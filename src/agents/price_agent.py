from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from src.tools.price_tools import CryptoPricePredictionTool, GetTokenPriceTool
from ..utils.vector_store_utils import VectorStoreUtils
from ..utils.create_agent import Agent

class State(MessagesState):
    next: str

class PriceAgent:
    """Agent for handling cryptocurrency price predictions."""
    
    def __init__(self, llm, name: str, prompt: str, next: str):
        self._name = name
        self.price_tools = [CryptoPricePredictionTool(), GetTokenPriceTool()]
        self.price_agent = Agent(
            tools=self.price_tools,
            vector_store=VectorStoreUtils(tools=self.price_tools),
            llm=llm,
            prompt=f"{prompt}\n\nYou have live token price data using get_token_price tool, and for predictions you can use the crypto_price_prediction tool. Do not make predictions unless asked to.\n Ex - 'What is the price of Bitcoin?' or 'Give a price report of EGGS token' or 'Predict the price of Bitcoin tomorrow'."
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