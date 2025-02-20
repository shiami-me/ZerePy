import logging
import uuid

from langgraph.graph.message import add_messages
from langchain.schema import AIMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from typing import TypedDict, Annotated, Any

logger = logging.getLogger("graph.agent")


class State(TypedDict):
    """Type for tracking conversation state"""
    messages: Annotated[list, add_messages]
    selected_tools: list[str]


class Agent:
    def __init__(self, tools, vector_store, llm, prompt):
        self.tools = tools
        self.vector_store = vector_store
        self.tool_registry = {str(uuid.uuid4()): tool for tool in self.tools}

        self.graph = self._create_conversation_graph()
        self.llm = llm
        self.prompt = prompt

    def _create_conversation_graph(self) -> Any:
        """Create the conversation flow graph"""
        def chatbot(state: State):
            """Generate response using LLM"""
            try:
                llm = self.llm
                messages = state["messages"]

                # Safely get selected tools
                selected_tools = []
                for tool_id in state.get("selected_tools", []):
                    tool = self.tool_registry[tool_id]
                    if tool:
                        selected_tools.append(tool)
                if not selected_tools:
                    # Fallback to using all tools if none selected
                    selected_tools = self.tools
                llm_with_tools = llm.bind_tools(
                    tools=selected_tools, tool_choice="auto")
                response = llm_with_tools.invoke(
                    [{"role": "system", "content": self.prompt}] + messages)

                return {"messages": [response]}
            except Exception as e:
                logger.error(f"Error in chatbot node: {str(e)}")
                return {"messages": [AIMessage(content="I apologize, but I encountered an error processing your request.")]}

        def route_tools(state: State):
            """Route based on whether tools are needed"""
            try:
                last_user_message = state["messages"][-1]
                query = last_user_message.content
                selected_tool_ids = self.vector_store.route_tools(
                    query, self.tool_registry)

                return {"selected_tools": selected_tool_ids}
            except Exception as e:
                logger.error(f"Error in route_tools: {str(e)}")
                return {"selected_tools": list(self.tool_registry.keys())}

        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_node("route_tools", route_tools)
        tool_node = ToolNode(tools=self.tools)
        graph_builder.add_node("tools", tool_node)
        graph_builder.add_conditional_edges(
            "chatbot", tools_condition, path_map=["tools", "__end__"])
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge("route_tools", "chatbot")
        graph_builder.add_edge(START, "route_tools")

        return graph_builder.compile()
