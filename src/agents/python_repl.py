from typing import Type, Optional
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_experimental.utilities import PythonREPL
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph import MessagesState


class State(MessagesState):
    next: str


repl = PythonREPL()


class PythonReplInput(BaseModel):
    code: str = Field(description="The Python code to execute.")


class PythonReplTool(BaseTool):
    name: str = "python_repl_tool"
    description: str = "Executes Python code and returns the result."
    args_schema: Type[BaseModel] = PythonReplInput
    return_direct: bool = True

    def _run(self, code: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Execute the given Python code."""
        try:
            result = repl.run(code)
        except BaseException as e:
            return f"Failed to execute. Error: {repr(e)}"
        return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"

    async def _arun(self, code: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Async execution of Python code."""
        return self._run(code, run_manager=run_manager.get_sync() if run_manager else None)


class PythonReplAgent:
    def __init__(self, llm, name: str, prompt: str):
        self._name = name
        self.repl = PythonREPL()
        self.code_agent = create_react_agent(
            llm, tools=[PythonReplTool()], prompt=prompt
        )

    def node(self, state: State):
        result = self.code_agent.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=result["messages"][-1].content, name=self._name
                    )
                ]
            },
            goto="shiami",
        )
