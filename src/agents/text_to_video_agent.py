from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph import MessagesState, END
from ..utils.vector_store_utils import VectorStoreUtils
from ..utils.create_agent import Agent
from ..tools.text_to_video_tools import (
    generate_content,
    create_powerpoint,
    convert_to_images,
    generate_video,
    PresentationRequest,
    setup_logging
)

logger = setup_logging()

class State(MessagesState):
    next: str

class TextToVideoAgent:
    """Agent for handling text to video conversion tasks."""
    
    def __init__(self, llm, name: str, prompt: str, next: str, run_manager):
        self._name = name
        # Initialize all tools
        self.tools = [
            generate_content,
            create_powerpoint,
            convert_to_images,
            generate_video
        ]
        
        # Create the agent with all available tools
        self.video_agent = Agent(
            tools=self.tools,
            vector_store=VectorStoreUtils(tools=self.tools),
            llm=llm,
            prompt=prompt
        )._create_conversation_graph()
        self.next = next

    def node(self, state: State):
        """Process the current state and return the next command."""
        try:
            # Process the request through the agent
            result = self.video_agent.invoke(state)
            
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=result["messages"][-1].content,
                            name=self._name
                        )
                    ]
                },
                goto=END,
            )
        except Exception as e:
            error_msg = f"Text to video agent error: {str(e)}"
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