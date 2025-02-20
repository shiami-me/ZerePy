from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph import MessagesState, END
from src.tools.scheduler_tool import SchedulerTool
from ..utils.vector_store_utils import VectorStoreUtils
from ..utils.create_agent import Agent

class State(MessagesState):
    next: str

class SchedulerAgent:
    """Agent for handling scheduling tasks using cron expressions."""
    
    def __init__(self, llm, name: str, prompt: str, next: str, run_manager):
        self._name = name
        self.scheduler_tool = SchedulerTool(run_manager)
        self.scheduler_agent = Agent(
            tools=[self.scheduler_tool],
            vector_store=VectorStoreUtils(tools=[self.scheduler_tool]),
            llm=llm,
            prompt=prompt
        )._create_conversation_graph()
        self.next = next

    def node(self, state: State):
        """Process the current state and return the next command."""
        try:
            result = self.scheduler_agent.invoke(state)
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
            error_msg = f"Scheduler agent error: {str(e)}"
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
