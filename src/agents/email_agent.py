from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from src.tools.email_tool import EmailTool

class State(MessagesState):
    next: str

class EmailAgent:
    """Agent for handling email operations using yagmail."""
    
    def __init__(self, llm, name: str, prompt: str):
        self._name = name
        self.email_tool = EmailTool()
        self.email_agent = create_react_agent(
            llm,
            tools=[self.email_tool],
            prompt=prompt
        )

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
                goto="shiami",
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
