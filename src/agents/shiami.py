import logging
from pydantic import BaseModel

from typing import Literal

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from .python_repl import PythonReplAgent
from .text_generation import TextAgent
from .scheduler_agent import SchedulerAgent

logger = logging.getLogger("agent/shiami")

class State(MessagesState):
    next: str

class Router(BaseModel):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["python_repl", "text", "scheduler", "FINISH"]

class Shiami:
    def __init__(self, agents: list[str], llm, prompts: dict[str, str]):
        self._agents = agents
        self._system_prompt = (
            "You are a Shiami tasked with managing a conversation between the"
            f" following workers: {agents}. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When error occurs,"
            " respond with FINISH."
            f" Prompts for workers: {prompts}"
        )

        self._llm = llm
        self._prompts = prompts
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
                    llm=self._llm, name=agent, prompt=self._prompts[agent], run_manager=self.execute_task).node)
                continue
            agent_class = self._name_to_class(agent)
            builder.add_node(agent, agent_class(
                llm=self._llm, name=agent, prompt=self._prompts[agent]).node)

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
        if class_name == "python_repl":
            return PythonReplAgent
        elif class_name == "text":
            return TextAgent
        elif class_name == "scheduler":
            return SchedulerAgent
        return None
