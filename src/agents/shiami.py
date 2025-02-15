import logging

from typing import Literal, Annotated, Union
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from .python_repl import PythonReplAgent
from .text_generation import TextAgent

logger = logging.getLogger("agent/shiami")

class State(MessagesState):
    next: str


class Shiami:
    def __init__(self, agents: list[str], llm, prompts: dict[str, str]):
        self._agents = agents
        self._system_prompt = (
            "You are a Shiami tasked with managing a conversation between the"
            f" following workers: {agents}. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When finished,"
            " respond with FINISH."
        )

        class Router(TypedDict):
            """Worker to route to next. If no workers needed, route to FINISH."""

            next: Literal[*agents, "FINISH"]
        self.router = Router
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
            agent_class = self._name_to_class(agent)
            builder.add_node(agent, agent_class(
                llm=self._llm, name=agent, prompt=self._prompts[agent]).node)

        return builder.compile()

    def node(self, state: State):
        messages = [
            {"role": "system", "content": self._system_prompt},
        ] + state["messages"]

        response = self._llm.with_structured_output(
            self.router()).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    def execute_task(self, message: str):
        # TODO it just prints for now, have to modify it
        for s in self.graph.stream(
            {
                "messages": [
                    (
                "user",
                message
                        )
                ]
            },
            subgraphs=True,
        ):
            logger.info(s)
            logger.info("----")

    @staticmethod
    def _name_to_class(class_name: str):
        if class_name == "python_repl":
            return PythonReplAgent
        elif class_name == "text":
            return TextAgent
        return None
