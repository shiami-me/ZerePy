import logging
from pydantic import BaseModel

from typing import Literal

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from .text_generation import TextAgent
from .scheduler_agent import SchedulerAgent
from .price_agent import PriceAgent
from .email_agent import EmailAgent
from .image_agent import ImageAgent
from ..utils.vector_store_utils import VectorStoreUtils

logger = logging.getLogger("agent/shiami")

class State(MessagesState):
    next: str

class Router(BaseModel):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["text", "scheduler", "email", "price", "image", "FINISH"]

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
            f" Here's some info for your current task: {prompt}"
            f" Prompts for workers: {prompts}"
        )
        self._data = data
        self._llm = llm
        self._prompts = prompts
        self._agent = agent
        self.graph = self.create_graph()

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

    def execute_task(self, message: str):
        try:
            for s in self.graph.stream(
                {
                    "messages": [("user", message)]
                },
                subgraphs=True,
            ):
                logger.info(s)
                logger.info("----")
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
        return None
