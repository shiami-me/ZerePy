from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph import MessagesState

class State(MessagesState):
    next: str

class TextAgent:
    def __init__(self, llm, prompt: str, name: str):
        super().__init__()
        self.text_agent = create_react_agent(
            llm, tools=[], prompt=prompt
        )
        self._name = name
    
    def node(self, state: State):
        result = self.text_agent.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name=self._name)
                ]
            },
            goto="shiami",
        )
        