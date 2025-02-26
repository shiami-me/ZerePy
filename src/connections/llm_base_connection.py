import logging
import os
import json
import uuid
import faiss
import asyncio
import functools

from typing import Dict, Any, Annotated, Optional, AsyncGenerator
from typing_extensions import TypedDict
from dotenv import load_dotenv, set_key
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import ToolMessage, AIMessageChunk, RemoveMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph, START, END
from src.tools.sonic_tools import get_sonic_tools
from src.tools.together_tools import get_together_tools
from src.tools.silo_tools import get_silo_tools
from src.tools.debridge_tools import get_debridge_tools
from src.tools.beets_tools import get_beets_tools
from src.connections.base_connection import BaseConnection, Action, ActionParameter
from langgraph.types import Command, interrupt
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.prompts import TAVILY_SEARCH_TOOL_PROMPT
from langchain_core.documents import Document
from langgraph.prebuilt import tools_condition, ToolNode
from src.utils.stores.docs_store import DocsStore

logger = logging.getLogger("connections.llm_base_connection")


class State(TypedDict):
    """Type for tracking conversation state"""
    messages: Annotated[list, add_messages]
    selected_tools: list[str]


class LLMBaseConnection(BaseConnection):
    """Base class for LLM connections with common functionality"""

    def __init__(self, config: Dict[str, Any], agent=Optional, tools: bool = True):
        super().__init__(config)
        self._client = None
        self.system_prompt = None
        if tools:
            self.search_tool = None
            self.tools = []
            self._agent = agent
            self.register_actions()
            self.setup_tools()
            self.graph_builder = self._create_conversation_graph()

    def setup_tools(self):
        """Initialize tools with FAISS for efficient similarity search"""
        try:
            load_dotenv()
            embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004")

            if self.config.get("tavily", False):
                tavily_api_key = os.getenv('TAVILY_API_KEY')
                if tavily_api_key:
                    self.search_tool = TavilySearchResults(
                        name="search_web",
                        description=TAVILY_SEARCH_TOOL_PROMPT,
                        api_key=tavily_api_key,
                        max_results=self.config.get("max_tavily_results", 2)
                    )
                    self.tools.append(self.search_tool)

            if "sonic" in self.config.get("plugins", []):
                self.tools.extend(get_sonic_tools(agent=self._agent, llm=self.get_llm_identifier()))

            if "silo" in self.config.get("plugins", []):
                self.tools.extend(get_silo_tools(agent=self._agent, llm=self.get_llm_identifier()))

            if "image" in self.config.get("plugins", []):
                self.tools.extend(get_together_tools(self._agent))
            
            if "debridge" in self.config.get("plugins", []):
                self.tools.extend(get_debridge_tools(self._agent))
            
            if "beets" in self.config.get("plugins", []):
                self.tools.extend(get_beets_tools(self._agent))
            
            self.tools.extend([DocsStore().retriever_tool])

            self.tool_registry = {str(uuid.uuid4()): tool for tool in self.tools}

            tool_documents = []
            for tool_id, tool in self.tool_registry.items():
                try:
                    doc = Document(
                        page_content=tool.description,
                        id=tool_id,
                        metadata={"tool_name": tool.name},
                    )
                    tool_documents.append(doc)
                except Exception as e:
                    logger.error(f"Error creating document for tool {tool.name}: {e}")
                    continue

            if tool_documents:
                # Initialize FAISS
                embedding_dim = len(embedding_model.embed_query("hello world"))
                index = faiss.IndexFlatL2(embedding_dim)
                self.vector_store = FAISS(
                    embedding_function=embedding_model,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                )
                self.vector_store.add_documents(tool_documents)
            else:
                raise ValueError("No valid tool documents created")

        except Exception as e:
            logger.error(f"Error in setup_tools: {e}")
            raise

    def get_llm_identifier(self) -> str:
        """Override this method to return the LLM identifier"""
        raise NotImplementedError

    def _create_conversation_graph(self) -> StateGraph:
        """Create the conversation flow graph"""
        def chatbot(state: State):
            """Generate response using LLM"""
            try:
                llm = self._get_client()
                messages = state["messages"][-7:]
                converted_messages = []
                for msg in messages:
                    if isinstance(msg, SystemMessage):
                        converted_messages.append(HumanMessage(
                            content=f"System: {msg.content}"))
                    elif isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
                        converted_messages.append(msg)

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
                response = llm_with_tools.invoke(converted_messages)

                return {"messages": [response]}
            except Exception as e:
                logger.error(f"Error in chatbot node: {str(e)}")
                return {"messages": [AIMessage(content="I apologize, but I encountered an error processing your request.")]}

        def route_tools(state: State):
            """Route based on whether tools are needed"""
            try:
                last_user_message = state["messages"][-1]
                query = last_user_message.content
                tool_documents = self.vector_store.similarity_search(
                    query, k=3)
                selected_tool_ids = [
                    doc.id for doc in tool_documents if doc.id in self.tool_registry]
                if not selected_tool_ids:
                    # Fallback to using all tools
                    selected_tool_ids = list(self.tool_registry.keys())

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

        return graph_builder

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate LLM configuration"""
        required_fields = ["model"]
        missing_fields = [
            field for field in required_fields if field not in config]

        if missing_fields:
            raise ValueError(
                f"Missing required configuration fields: {', '.join(missing_fields)}")

        if not isinstance(config["model"], str):
            raise ValueError("model must be a string")

        if "tavily" in config and not isinstance(config["tavily"], bool):
            raise ValueError("tavily configuration must be a boolean")

        return config

    def register_actions(self) -> None:
        """Register common LLM actions"""
        self.actions = {
            "generate-text": Action(
                name="generate-text",
                parameters=[
                    ActionParameter("prompt", True, str,
                                    "The input prompt for text generation"),
                    ActionParameter("system_prompt", True, str,
                                    "System prompt to guide the model"),
                    ActionParameter("thread", False, str,
                                    "Thread ID for conversation"),
                    ActionParameter("model", False, str,
                                    "Model to use for generation"),
                    ActionParameter("temperature", False, float,
                                    "Temperature for generation")
                ],
                description="Generate text using LLM models"
            ),
            "check-model": Action(
                name="check-model",
                parameters=[
                    ActionParameter("model", True, str,
                                    "Model name to check availability")
                ],
                description="Check if a specific model is available"
            ),
            "list-models": Action(
                name="list-models",
                parameters=[],
                description="List all available models"
            ),
            "continue-execution": Action(
                name="continue-execution",
                parameters=[
                    ActionParameter("data", True, str,
                                    "Data to continue execution"),
                    ActionParameter("thread", True, str,
                                    "Thread ID for conversation"),
                ],
                description="Continue execution with provided data"
            ),
            "get-message-history": Action(
                name="get-message-history",
                parameters=[
                    ActionParameter("thread", True, str,
                                    "Thread ID to retrieve message history")
                ],
                description="Retrieve message history for a specific thread"
            ),
            "delete-chat": Action(
                name="delete-chat",
                parameters=[
                    ActionParameter("thread", True, str,
                                    "Thread ID to delete chat history")
                ],
                description="Delete chat history for a specific thread"
            ),
        }

    async def continue_execution(self, data: str, thread: str) -> Any:
        """Continue execution based on provided data"""
        try:
            config = {"configurable": {"thread_id": thread}}

            db_uri = os.getenv('POSTGRES_DB_URI')
            if not db_uri:
                raise ValueError(
                    "PostgreSQL connection URI not found in environment")

            async with AsyncPostgresSaver.from_conn_string(db_uri) as checkpointer:
                await checkpointer.setup()
                graph = self.graph_builder.compile(checkpointer=checkpointer)
                await graph.ainvoke(
                    Command(resume={"data": data}),
                    config=config
                )

        except Exception as e:
            logger.error(f"Error in continue_execution: {str(e)}")
            raise

    def interrupt_chat(self, query: str) -> Any:
        """Interrupt the current chat flow"""
        response = interrupt({"query": query})
        return response["data"]

    def _get_client(self):
        """Override this method to return the specific LLM client"""
        raise NotImplementedError

    def configure(self) -> bool:
        """Override this method to handle specific LLM configuration"""
        raise NotImplementedError

    def is_configured(self, verbose: bool = False) -> bool:
        """Override this method to check specific LLM configuration"""
        raise NotImplementedError

    def check_model(self, model: str, **kwargs) -> bool:
        """Override this method to check model availability"""
        raise NotImplementedError

    def list_models(self, **kwargs) -> None:
        """Override this method to list available models"""
        raise NotImplementedError

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str,
        thread: str = None,
        model: str = None,
        stream: bool = True,
        **kwargs
    ) -> AsyncGenerator:
        """Generate text using the LLM"""
        try:
            enhanced_system_prompt = f"""
{system_prompt}

Behave like a normal assistant if there's no wallet.
You are a helpful assistant with access to various tools. When using tools:
1. Don't explain what you're doing, just do it
2. Don't ask for confirmation, just execute
3. Don't mention the tool names
4. Keep responses natural and concise
5. Use multiple tools when needed
6. When you need some external information or the user asks for internet search, use Tavily search tool when available.
7. Use connect wallet address whenever it's needed, for example - for sonic related tools. Ask user to connect wallet if needed.
"""
            if not self.system_prompt:
                self.system_prompt = enhanced_system_prompt

            messages = [
                HumanMessage(
                    content=f"Instructions for you: {enhanced_system_prompt}"),
                HumanMessage(content=f"Input: {prompt}")
            ]

            initial_state = {
                "messages": messages,
            }

            config = {"configurable": {"thread_id": thread}}

            db_uri = os.getenv('POSTGRES_DB_URI')
            if not db_uri:
                raise ValueError(
                    "PostgreSQL connection URI not found in environment")

            async with AsyncPostgresSaver.from_conn_string(db_uri) as checkpointer:
                await checkpointer.setup()
                graph = self.graph_builder.compile(checkpointer=checkpointer)
                response_stream = graph.astream(
                    initial_state,
                    config,
                    stream_mode="messages"
                )

                async for events in response_stream:

                    for event in events:
                        if hasattr(event, "content"):
                            yield event.content
                        if hasattr(event, "additional_kwargs"):
                            if isinstance(event.additional_kwargs, dict):
                                function_call = event.additional_kwargs.get("function_call")
                                
                                if isinstance(function_call, dict):
                                    function_name = function_call.get("name")

                                    if function_name:
                                        yield json.dumps({"tool": function_name})
                state = await graph.aget_state(config=config)

                if len(state.tasks) > 0:
                    task = state.tasks[-1]
                    if hasattr(task, "interrupts") and task.interrupts:
                        interrupt_value = task.interrupts[0].value
                        if isinstance(interrupt_value, dict) and "query" in interrupt_value:
                            yield interrupt_value["query"]


        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise
        
    async def get_message_history(self, thread: str) -> list:
        """
        Retrieve and format the message history for the frontend.

        Args:
            thread (str): Thread ID.

        Returns:
            list: A list of messages formatted as {id, text, sender}.
        """
        try:
            config = {"configurable": {"thread_id": thread}}
            formatted_messages = []
            db_uri = os.getenv('POSTGRES_DB_URI')
            
            async with AsyncPostgresSaver.from_conn_string(db_uri) as checkpointer:
                await checkpointer.setup()
                state = await self.graph_builder.compile(checkpointer=checkpointer).aget_state(config=config)
                messages = state.values.get("messages", [])
                i = 0
                for message in messages:
                    if isinstance(message, HumanMessage) and "Input: " in message.content:
                        content = message.content
                        sender = "user"

                        formatted_messages.append({
                            "id": i,
                            "text": content.replace("Input: ", ""),
                            "sender": sender
                        })
                        i += 1
                    elif isinstance(message, AIMessage):
                        content = message.content
                        sender = "bot"

                        formatted_messages.append({
                            "id": i,
                            "text": content,
                            "sender": sender
                        })
                        i += 1
            return formatted_messages

        except Exception as e:
            logger.error(f"Error fetching message history: {str(e)}")
            return []

    async def delete_chat(self, thread: str) -> None:
        """
        Delete the chat history for a specific thread.

        Args:
            thread (str): Thread ID.
        """
        try:
            db_uri = os.getenv('POSTGRES_DB_URI')
            async with AsyncPostgresSaver.from_conn_string(db_uri) as checkpointer:
                await checkpointer.setup()
                graph = self.graph_builder.compile(checkpointer=checkpointer)
                state = await graph.aget_state(config={"configurable": {"thread_id": thread}})
                messages = state.values.get("messages", [])
                for message in messages:
                    await graph.aupdate_state(
                        config={"configurable": {"thread_id": thread}},
                        values={"messages": [RemoveMessage(id=message.id)], "selected_tools": []},
                        as_node="chatbot"
                    )


        except Exception as e:
            logger.error(f"Error deleting chat history: {str(e)}")

    @property
    def is_llm_provider(self) -> bool:
        return True

    async def perform_action(self, action_name: str, kwargs) -> Any:
        """Execute an LLM action with validation"""
        if action_name not in self.actions:
            raise KeyError(f"Unknown action: {action_name}")

        load_dotenv()
        if not self.is_configured(verbose=True):
            raise ValueError(f"{self.__class__.__name__} is not properly configured")

        action = self.actions[action_name]
        errors = action.validate_params(kwargs)
        if errors:
            raise ValueError(f"Invalid parameters: {', '.join(errors)}")

        method_name = action_name.replace('-', '_')
        method = getattr(self, method_name)
        if asyncio.iscoroutinefunction(method):
            return await method(**kwargs)
        
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(method, **kwargs))