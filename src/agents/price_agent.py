from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from src.tools.price_tools import CryptoPricePredictionTool

class State(MessagesState):
    next: str

class PriceAgent:
    """Agent for handling cryptocurrency price predictions."""
    
    def __init__(self, llm, name: str, prompt: str):
        self._name = name
        self.price_tool = CryptoPricePredictionTool()
        self.price_agent = create_react_agent(
            llm,
            tools=[self.price_tool],
            prompt=prompt
        )

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
                goto="shiami",
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