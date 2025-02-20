from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph import MessagesState
from src.tools.together_tools import TogetherImageGenerationTool
from ..utils.vector_store_utils import VectorStoreUtils
from ..utils.create_agent import Agent

class State(MessagesState):
    next: str

class ImageAgent:
    """Agent for handling image generation using Together AI."""
    
    def __init__(self, llm, name: str, prompt: str, next: str, agent):
        self._name = name
        self.image_tool = TogetherImageGenerationTool(agent)
        self.image_agent = Agent(
            tools=[self.image_tool],
            vector_store=VectorStoreUtils(tools=[self.image_tool]),
            llm=llm,
            prompt=prompt
        )._create_conversation_graph()
        self.next = next

    def node(self, state: State):
        """Process the current state and return the next command."""
        try:
            result = self.image_agent.invoke(state)
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
            error_msg = f"Image generation agent error: {str(e)}"
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
