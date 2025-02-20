from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from src.tools.price_tools import CryptoPricePredictionTool
from ..utils.vector_store_utils import VectorStoreUtils
from ..utils.create_agent import Agent

class State(MessagesState):
    next: str

class PriceAgent:
    """Agent for handling cryptocurrency price predictions."""
    
    def __init__(self, llm, name: str, prompt: str, next: str):
        self._name = name
        self.price_tool = CryptoPricePredictionTool()
        self.price_agent = Agent(
            tools=[self.price_tool],
            vector_store=VectorStoreUtils(tools=[self.price_tool]),
            llm=llm,
            prompt=prompt
        )._create_conversation_graph()
        self.next = next

    def node(self, state: State):
        """Process the current state and return the next command."""
        try:
            result = self.price_agent.invoke(state)
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=result["messages"][-1].content,
                            name=self._name
                        )
                    ]
                },
#         # goto=self.next
            goto="shiami"
            )
        except Exception as e:
            error_msg = f"Price prediction agent error: {str(e)}"
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