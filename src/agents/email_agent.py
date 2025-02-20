from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph import MessagesState
from src.tools.email_tool import EmailTool
from ..utils.vector_store_utils import VectorStoreUtils
from ..utils.create_agent import Agent

class State(MessagesState):
    next: str

class EmailAgent:
    """Agent for handling email operations using yagmail."""
    
    def __init__(self, llm, name: str, prompt: str, next: str):
        self._name = name
        self.email_tool = EmailTool()
        self.email_agent = Agent(
            tools=[self.email_tool],
            vector_store=VectorStoreUtils(tools=[self.email_tool]),
            llm=llm,
            prompt=prompt
        )._create_conversation_graph()
        self.next = next

    def node(self, state: State):
        """Process the current state and return the next command."""
        try:
            result = self.email_agent.invoke(state)
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
            error_msg = f"Email agent error: {str(e)}"
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
