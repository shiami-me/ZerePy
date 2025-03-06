import logging
import os
import uuid
from pydantic import BaseModel

from typing import Literal
import shutil
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from .text_generation import TextAgent
from .scheduler_agent import SchedulerAgent
from .price_agent import PriceAgent
from .email_agent import EmailAgent
from .image_agent import ImageAgent
from.rag_agent import RAGAgent
from.text_to_video_agent import TextToVideoAgent
from .portfolio_rebalance_agent import PortfolioRebalanceAgent
from ..tools.rag_tools import GraphRAG, load_documents_from_path ,Document

logger = logging.getLogger("agent/shiami")

class State(MessagesState):
    next: str
    documents: list[Document]
    file_paths: list[str]

class Router(BaseModel):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["text", "scheduler", "email", "price", "image", "rag","video", "portfolio", "FINISH"]

class Shiami:
    def __init__(self, agent ,agents: list[str], llm, prompt: str, prompts: dict[str, str], data: dict[str, str]):
        self._agents = agents
        self._system_prompt = (
            "You are a Shiami tasked with managing a conversation between the"
            f" following workers: {agents}. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When error occurs,"
            " respond with FINISH."
            " if some agent doesn't do it's task well, ask to redo. and if some agent replies with anything that is the job of other agent then retry by refining the prompt specific to that agent."
            " For video creation tasks, use the video agent which can convert text to video through PowerPoint."
            f" Here's some info for your current task: {prompt}"
            f" Prompts for workers: {prompts}"
        )
        self._data = data
        self._llm = llm
        self._prompts = prompts
        self._agent = agent
        self.graph = self.create_graph()
        self.rag_instance = None

        # Example configuration to add to your data dictionary
        self._data["video"] = {
            "name": "video",
            "next": "shiami"
        }

    async def handle_file_upload(self, file_paths: list[str]):
        """Handles the file upload process."""
        try:
            saved_file_paths = []
            for file_path in file_paths:
                # Generate a unique filename

                unique_filename = str(uuid.uuid4()) + os.path.splitext(file_path)[1]
                # Define the upload directory

                upload_dir = "uploads"  # Change this to your desired directory

                os.makedirs(upload_dir, exist_ok=True)  # Ensure directory exists
                saved_file_path = os.path.join(upload_dir, unique_filename)

                if not file_path.lower().endswith(('.pdf', '.txt', '.docx')):
                    return {"error": "Unsupported file type."}

                # to be implemented, read the file content from the uploaded file object and write it to the saved path.
                shutil.copy(file_path, saved_file_path) # Example using shutil

                saved_file_paths.append(saved_file_path) 

            return {"status": "files saved", "saved_file_paths": saved_file_paths}
        except Exception as e:
            logger.error(f"File upload error: {e}")
            return {"error": str(e)}

    def create_graph(self):
        builder = StateGraph(State)
        builder.add_edge(START, "shiami")
        builder.add_node("shiami", self.node)
        for agent in self._agents:
            if self._name_to_class(agent) is None:
                continue
            if agent == "scheduler":
                builder.add_node("scheduler", SchedulerAgent(
                    llm=self._llm, 
                    name=agent, 
                    prompt=self._prompts[agent], 
                    next=self._data["scheduler"]["next"], 
                    run_manager=self.execute_task).node)
                continue
            if agent == "rag":
                builder.add_node("rag", RAGAgent(
                    llm=self._llm,
                    name=agent,
                    next=self._data["rag"]["next"],
                    rag_instance=self.rag_instance
                ).node)
            if agent == "image":
                builder.add_node("image", ImageAgent(
                    llm=self._llm,
                    name=agent,
                    prompt=self._prompts[agent],
                    next=self._data["image"]["next"],
                    agent=self._agent
                ).node)
                continue
            agent_class = self._name_to_class(self._data[agent]["name"])
            logger.info(self._data[agent]["next"])
            logger.info(agent)
            builder.add_node(
                agent, 
                agent_class(
                    llm=self._llm, 
                    name=agent, 
                    prompt=self._prompts[agent],
                    next=self._data[agent]["next"]
                ).node
            )

        return builder.compile()


    def node(self, state: State):
        messages = [
            {"role": "system", "content": self._system_prompt},
        ] + state["messages"]

        response = self._llm.with_structured_output(
            Router).invoke(messages)
        goto = response.next
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    async def execute_task(self, message: str, file_paths: list[str] = None):
        try:
            if file_paths:
                file_status = await self.handle_file_upload(file_paths)
                if "error" in file_status:
                    return file_status

            async for s in self.graph.astream(
                {
                    "messages": [("user", message)],
                    "file_paths": file_paths or []
                },
                subgraphs=True,
            ):
                logger.info(s)
                if "messages" in s and s["messages"]:
                    last_msg = s["messages"][-1].content
                    if "recommendation" in last_msg.lower():
                        return s
            return {"status": "completed"}
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            return {"error": str(e)}

    @staticmethod
    def _name_to_class(class_name: str):
        if class_name == "text":
            return TextAgent
        elif class_name == "scheduler":
            return SchedulerAgent
        elif class_name == "email":
            return EmailAgent
        elif class_name == "price":
            return PriceAgent
        elif class_name == "image":
            return ImageAgent
        elif class_name == "rag":
            return RAGAgent
        elif class_name == "portfolio":
            return PortfolioRebalanceAgent
        elif class_name == "video":
            return TextToVideoAgent
        return None
