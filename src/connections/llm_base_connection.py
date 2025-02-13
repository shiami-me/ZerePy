import logging
import os
import json
import uuid
from typing import Dict, Any, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv, set_key
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, START, END
from src.tools.sonic_tools import get_sonic_tools
from src.tools.together_tools import get_together_tools
from src.tools.silo_tools import get_silo_tools
from src.connections.base_connection import BaseConnection, Action, ActionParameter
from langgraph.types import Command, interrupt
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.prompts import TAVILY_SEARCH_TOOL_PROMPT
from langchain_core.documents import Document
from langgraph.prebuilt import tools_condition, ToolNode

logger = logging.getLogger("connections.llm_base_connection")

class State(TypedDict):
    """Type for tracking conversation state"""
    messages: Annotated[list, add_messages]
    selected_tools: list[str]

class LLMBaseConnection(BaseConnection):
    """Base class for LLM connections with common functionality"""
    
    def __init__(self, config: Dict[str, Any], agent):
        super().__init__(config)
        self._client = None
        self.system_prompt = None
        self.search_tool = None
        self.tools = []
        self._agent = agent
        self.register_actions()
        self.setup_tools()
        self.graph_builder = self._create_conversation_graph()

    def setup_tools(self):
        """Initialize tools for the connection"""
        try:
            load_dotenv()
            
            if self.config.get("tavily", False):
                tavily_api_key = os.getenv('TAVILY_API_KEY')
                if tavily_api_key:
                    self.search_tool = TavilySearchResults(
                        description=TAVILY_SEARCH_TOOL_PROMPT,
                        api_key=tavily_api_key,
                        max_results=self.config.get("max_tavily_results", 2)
                    )
                    self.tools.append(self.search_tool)
            
            if "sonic" in self.config.get("plugins", []):
                sonic_tools = get_sonic_tools(agent=self._agent, llm=self.get_llm_identifier())
                self.tools.extend(sonic_tools)   
            
            if "silo" in self.config.get("plugins", []):
                silo_tools = get_silo_tools(agent=self._agent, llm=self.get_llm_identifier())
                self.tools.extend(silo_tools)       
        
            if "image" in self.config.get("plugins", []):
                image_tools = get_together_tools(self._agent)
                self.tools.extend(image_tools)
            self.tool_registry = {
                str(uuid.uuid4()): tool for tool in self.tools
            }
            db_uri = os.getenv('POSTGRES_DB_URI')
            if not db_uri:
                raise ValueError("PostgreSQL connection URI not found in environment")
            
            # Create tool documents with error handling
            tool_documents = []
            for id, tool in self.tool_registry.items():
                try:
                    doc = Document(
                        page_content=tool.description,
                        id=id,
                        metadata={"tool_name": tool.name},
                    )
                    tool_documents.append(doc)
                except Exception as e:
                    logger.error(f"Error creating document for tool {tool.name}: {e}")
                    continue

            if tool_documents:
                # self.vector_store = PGVector(
                #     embeddings=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
                #     collection_name=f"{self.get_llm_identifier()}-tools",
                #     connection=db_uri,
                #     use_jsonb=True,
                # )
                self.vector_store = InMemoryVectorStore(
                    embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
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
                messages = state["messages"]
                converted_messages = []
                for msg in messages:
                    if isinstance(msg, SystemMessage):
                        converted_messages.append(HumanMessage(content=f"System: {msg.content}"))
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

                llm_with_tools = llm.bind_tools(tools=selected_tools, tool_choice="auto")
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
                tool_documents = self.vector_store.similarity_search(query, k=3)
                selected_tool_ids = [doc.id for doc in tool_documents if doc.id in self.tool_registry]
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
        graph_builder.add_conditional_edges("chatbot", tools_condition, path_map=["tools", "__end__"])
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge("route_tools", "chatbot")
        graph_builder.add_edge(START, "route_tools")

        return graph_builder

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate LLM configuration"""
        required_fields = ["model"]
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {', '.join(missing_fields)}")
            
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
                    ActionParameter("prompt", True, str, "The input prompt for text generation"),
                    ActionParameter("system_prompt", True, str, "System prompt to guide the model"),
                    ActionParameter("model", False, str, "Model to use for generation"),
                    ActionParameter("temperature", False, float, "Temperature for generation")
                ],
                description="Generate text using LLM models"
            ),
            "check-model": Action(
                name="check-model",
                parameters=[
                    ActionParameter("model", True, str, "Model name to check availability")
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
                    ActionParameter("data", True, str, "Data to continue execution")
                ],
                description="Continue execution with provided data"
            )
        }

    def continue_execution(self, data: str) -> Any:
        """Continue execution based on provided data"""
        try:
            config = {"configurable": {"thread_id": "222"}}
            
            db_uri = os.getenv('POSTGRES_DB_URI')
            if not db_uri:
                raise ValueError("PostgreSQL connection URI not found in environment")
            
            with PostgresSaver.from_conn_string(db_uri) as checkpointer:
                checkpointer.setup()
                graph = self.graph_builder.compile(checkpointer=checkpointer)
                graph.invoke(
                    Command(resume={"data": data}), 
                    config=config
                )
                
        except Exception as e:
            logger.error(f"Error in continue_execution: {str(e)}")
            raise

    def interrupt_chat(self, query: str) -> Any:
        """Interrupt the current chat flow"""
        logger.info(query)
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

    def generate_text(
        self,
        prompt: str,
        system_prompt: str,
        model: str = None,
        stream: bool = True,
        **kwargs
    ) -> str:
        """Generate text using the LLM"""
        try:
            enhanced_system_prompt = f"""
{system_prompt}

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
                HumanMessage(content=f"Instructions for you: {enhanced_system_prompt}"),
                HumanMessage(content=f"Input: {prompt}")
            ]

            initial_state = {
                "messages": messages,
            }

            config = {"configurable": {"thread_id": "222"}}
            
            collected_response = []
            db_uri = os.getenv('POSTGRES_DB_URI')
            if not db_uri:
                raise ValueError("PostgreSQL connection URI not found in environment")
                
            with PostgresSaver.from_conn_string(db_uri) as checkpointer:
                checkpointer.setup()
                graph = self.graph_builder.compile(checkpointer=checkpointer)
                response_stream = graph.stream(
                    initial_state,
                    config,
                )
                
                for chunk in response_stream:
                    if isinstance(chunk, dict):
                        if "chatbot" in chunk and "messages" in chunk["chatbot"]:
                            messages = chunk["chatbot"]["messages"]
                            if messages and isinstance(messages[-1], (AIMessage, ToolMessage)):
                                collected_response.append(messages[-1].content)
                        elif "__interrupt__" in chunk:
                            interrupt_data = chunk["__interrupt__"][0]
                            if hasattr(interrupt_data, 'value'):
                                interrupt_value = interrupt_data.value
                                collected_response = [json.dumps(interrupt_value.get('query', ''))]
                        elif "tools" in chunk and "messages" in chunk["tools"]:
                            messages = chunk["tools"]["messages"]
                            for message in messages:
                                if isinstance(message, (ToolMessage, AIMessage)):
                                    collected_response.append(message.content)

            return "\n".join(filter(None, collected_response))
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise

    @property
    def is_llm_provider(self) -> bool:
        return True

    def perform_action(self, action_name: str, kwargs) -> Any:
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
        return method(**kwargs)
