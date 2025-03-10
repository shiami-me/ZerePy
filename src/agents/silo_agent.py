from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph import MessagesState, END
from src.tools.silo_tools import get_silo_tools
from ..utils.vector_store_utils import VectorStoreUtils
from ..utils.create_agent import Agent
import json
import logging

logger = logging.getLogger("agents.silo_agent")


class State(MessagesState):
    next: str


class SiloAgent:
    """Agent for handling Silo protocol operations with transaction processing capabilities."""

    def __init__(self, llm, name: str, prompt: str, next: str, agent):
        self._name = name
        self._agent = agent
        self._next = next

        # Initialize Silo-specific tools
        self.silo_tools = get_silo_tools(agent=self._agent, llm=llm)

        # Create the agent with the tools
        self.silo_agent = Agent(
            tools=self.silo_tools,
            vector_store=VectorStoreUtils(tools=self.silo_tools),
            llm=llm,
            prompt=prompt
        )._create_conversation_graph()

    def node(self, state: State):
        """Process the current state and return the next command."""
        try:
            result = self.silo_agent.invoke(state)
            logger.info(f"Silo agent response: {result['messages'][-1].content[:100]}...")
            
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=result["messages"][-1].content,
                            name=self._name
                        )
                    ]
                },
                goto="shiami"
            )
        except Exception as e:
            error_msg = f"Silo agent error: {str(e)}"
            logger.error(error_msg)
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=error_msg,
                            name=self._name
                        )
                    ]
                },
                goto="shiami",
            )