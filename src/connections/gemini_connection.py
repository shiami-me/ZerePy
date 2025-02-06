import logging
import os
import json
import time
import uuid
from typing import Dict, Any, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv, set_key
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from src.tools.sonic_tools import get_sonic_tools
from src.tools.together_tools import get_together_tools
from src.connections.base_connection import BaseConnection, Action, ActionParameter
from langgraph.types import Command, interrupt

logger = logging.getLogger("connections.gemini_connection")

class State(TypedDict):
    """Type for tracking conversation state"""
    messages: Annotated[list, add_messages]
    context: Dict[str, Any]

class BasicToolNode:
    """Tool execution node for LangGraph"""
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}
        logger.debug(f"Initialized tools: {list(self.tools_by_name.keys())}")

    def __call__(self, inputs: dict):
        try:
            messages = inputs.get("messages", [])
            if not messages:
                raise ValueError("No message found in input")
            
            message = messages[-1]
            outputs = []
            
            # Add debugging
            logger.debug(f"Processing message in BasicToolNode: {message}")
            
            # Handle different tool call formats
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls is None and hasattr(message, "additional_kwargs"):
                tool_calls = message.additional_kwargs.get("tool_calls", [])
            
            if not tool_calls:
                logger.warning("No tool calls found in message")
                return {"messages": outputs}
            
            for tool_call in tool_calls:
                logger.debug(f"Processing tool call: {tool_call}")
                
                # Handle different tool call formats
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args")
                    tool_id = tool_call.get("id", "default_id")
                else:
                    tool_name = tool_call.name
                    tool_args = tool_call.args
                    tool_id = getattr(tool_call, "id", "default_id")
                
                if tool_name not in self.tools_by_name:
                    logger.error(f"Tool not found: {tool_name}")
                    continue
                
                try:
                    tool_result = self.tools_by_name[tool_name].invoke(tool_args)
                    logger.debug(f"Tool result: {tool_result}")
                    
                    tool_content = (
                        json.dumps(tool_result) 
                        if not isinstance(tool_result, str) 
                        else tool_result
                    )
                    
                    outputs.append(
                        ToolMessage(
                            content=tool_content,
                            name=tool_name,
                            tool_call_id=tool_id,
                        )
                    )
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}")
                    if tool_name == "sonic_request_transaction_data":
                        outputs.append(
                            ToolMessage(
                                content="Waiting for transaction data",
                                name=tool_name,
                                tool_call_id=tool_id,
                            )
                        )
                    else:
                        outputs.append(
                            ToolMessage(
                                content=json.dumps({"error": str(e)}),
                                name=tool_name,
                                tool_call_id=tool_id,
                            )
                        )
            
            return {"messages": outputs}
            
        except Exception as e:
            logger.error(f"Error in BasicToolNode: {e}")
            raise

class GeminiConnectionError(Exception):
    """Base exception for Gemini connection errors"""
    pass

class GeminiConfigurationError(GeminiConnectionError):
    """Raised when there are configuration/credential issues"""
    pass

class GeminiAPIError(GeminiConnectionError):
    """Raised when Gemini API requests fail"""
    pass

class GeminiConnection(BaseConnection):
    def __init__(self, config: Dict[str, Any], agent):
        super().__init__(config)
        self._client = None
        self.system_prompt = None
        self.search_tool = None
        self.register_actions()
        self._agent = agent
        self.setup_tools()
        self.graph_builder = self._create_conversation_graph()
        self.checkpointer = None

    
    def _create_conversation_graph(self) -> StateGraph:
        """Create the conversation flow graph"""
        
        def chatbot(state: State):
            """Generate response using Gemini"""
            llm = self._get_client()
            messages = state["messages"]
            
            # Convert messages to format Gemini expects
            converted_messages = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    # Convert system message to human message
                    converted_messages.append(HumanMessage(content=f"System: {msg.content}"))
                elif isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
                    converted_messages.append(msg)
            
            # If there are tool messages, create a more detailed summary prompt
            tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
            if tool_messages:
                context = "\n\nI found multiple relevant sources:\n" + "\n".join(
                    [msg.content for msg in tool_messages]
                ) + "\n\nPlease consider all these sources in your response."
                
                # Add context to the last user message
                for i in reversed(range(len(converted_messages))):
                    if isinstance(converted_messages[i], HumanMessage):
                        converted_messages[i].content += context
                        break
            
            llm_with_tools = llm.bind_tools(
                tools=self.tools,
                tool_choice="auto"
            )
            response = llm_with_tools.invoke(converted_messages)
            
            return {"messages": [response]}

        def route_tools(state: State):
            """Route based on whether tools are needed"""
            if isinstance(state, list):
                ai_message = state[-1]
            elif messages := state.get("messages", []):
                ai_message = messages[-1]
            else:
                raise ValueError(f"No messages found in input state: {state}")
            
            tool_calls = getattr(ai_message, "tool_calls", None)
            if tool_calls is None and hasattr(ai_message, "additional_kwargs"):
                tool_calls = ai_message.additional_kwargs.get("tool_calls", [])
            
            if tool_calls and len(tool_calls) > 0:
                logger.debug(f"Found tool calls: {tool_calls}")
                return "tools"
            return END

        # Create graph
        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node("chatbot", chatbot)
        tool_node = BasicToolNode(tools=self.tools)
        graph_builder.add_node("tools", tool_node)
        
        # Add edges
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_conditional_edges(
            "chatbot",
            route_tools,
            {"tools": "tools", END: END}
        )
        graph_builder.add_edge("tools", "chatbot")
        return graph_builder


    def setup_tools(self):
        """Initialize tools for the connection"""
        self.tools = []
        load_dotenv()
        
        if self.config.get("tavily", False):
            tavily_api_key = os.getenv('TAVILY_API_KEY')
            if tavily_api_key:
                self.search_tool = TavilySearchResults(
                    api_key=tavily_api_key,
                    max_results=self.config.get("max_tavily_results", 2)
                )
                self.tools.append(self.search_tool)
        
        if "sonic" in self.config.get("plugins", []):
            sonic_tools = get_sonic_tools(agent=self._agent, llm="gemini")
            self.tools.extend(sonic_tools)       
        
        if "image" in self.config.get("plugins", []):
            image_tools = get_together_tools(self._agent)
            self.tools.extend(image_tools)

    @property
    def is_llm_provider(self) -> bool:
        return True

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Gemini configuration from JSON"""
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
        """Register available Gemini actions"""
        self.actions = {
            "generate-text": Action(
                name="generate-text",
                parameters=[
                    ActionParameter("prompt", True, str, "The input prompt for text generation"),
                    ActionParameter("system_prompt", True, str, "System prompt to guide the model"),
                    ActionParameter("model", False, str, "Model to use for generation"),
                    ActionParameter("temperature", False, float, "Temperature for generation")
                ],
                description="Generate text using Gemini models"
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
                description="List all available Gemini models"
            ),
            "continue-execution": Action(
                name="continue-execution",
                parameters=[
                    ActionParameter("data", True, str, "Data to continue execution")
                ],
                description="Continue execution with provided data"
            )
        }

    def _get_client(self) -> ChatGoogleGenerativeAI:
        """Get or create Gemini client"""
        if not self._client:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise GeminiConfigurationError("Google API key not found in environment")
            
            self._client = ChatGoogleGenerativeAI(
                model=self.config["model"],
                temperature=self.config.get("temperature", 0.7),
                convert_system_message_to_human=True,
                google_api_key=api_key
            )
        return self._client

    def configure(self) -> bool:
        """Sets up Gemini and Tavily API authentication"""
        logger.info("\n🤖 API SETUP")

        if self.is_configured():
            logger.info("\nAPIs are already configured.")
            response = input("Do you want to reconfigure? (y/n): ")
            if response.lower() != 'y':
                return True

        logger.info("\n📝 To get your API credentials:")
        logger.info("Google AI: https://aistudio.google.com/app/apikey")
        
        google_api_key = input("\nEnter your Google API key: ")
        postgres_uri = input("\nEnter your PostgreSQL connection URI: ")

        tavily_api_key = None
        if self.config.get("tavily", False):
            logger.info("Tavily: https://tavily.com")
            tavily_api_key = input("\nEnter your Tavily API key: ")

        try:
            if not os.path.exists('.env'):
                with open('.env', 'w') as f:
                    f.write('')

            set_key('.env', 'GEMINI_API_KEY', google_api_key)
            set_key('.env', 'POSTGRES_DB_URI', postgres_uri)
            if tavily_api_key:
                set_key('.env', 'TAVILY_API_KEY', tavily_api_key)
            
            # Validate the configurations
            self._client = None
            self._get_client()
            
            if self.config.get("tavily", False) and tavily_api_key:
                self.search_tool = TavilySearchResults(
                    api_key=tavily_api_key,
                    max_results=self.config.get("max_tavily_results", 2)
                )

            logger.info("\n✅ API configuration successfully saved!")
            logger.info("Your API keys have been stored in the .env file.")
            return True

        except Exception as e:
            logger.error(f"Configuration failed: {e}")
            return False

    def is_configured(self, verbose = False) -> bool:
        """Check if required API keys and configurations are present"""
        try:
            load_dotenv()
            api_key = os.getenv('GEMINI_API_KEY')
            db_uri = os.getenv('POSTGRES_DB_URI')
            
            if not api_key or not db_uri:
                return False

            if self.config.get("tavily", False):
                tavily_api_key = os.getenv('TAVILY_API_KEY')
                if not tavily_api_key:
                    return False

            self._client = None
            self._get_client()
            return True
            
        except Exception as e:
            if verbose:
                logger.debug(f"Configuration check failed: {e}")
            return False

    def generate_text(
        self,
        prompt: str,
        system_prompt: str,
        model: str = None,
        stream: bool = True,
        **kwargs
    ) -> str:
        try:
            enhanced_system_prompt = f"""
    {system_prompt}

    You are a helpful assistant with access to various tools. When using tools:
    1. Don't explain what you're doing, just do it
    2. Don't ask for confirmation, just execute
    3. Don't mention the tool names
    4. Keep responses natural and concise
    5. Use multiple tools when needed
    6. When you need some external information or the user asks for it, use Tavily search tool when available.
    7. Use connect wallet address whenever it's needed, for example - for sonic related tools. Ask user to connect wallet if needed(only for sonic related tools except sonic_request_transaction_data).
    
    8. For Sonic transfers or swaps:
       - First execute the transfer/swap operation
       - Always use sonic_request_transaction_data in such transfer/swap/send operations.
       - Do not use sonic_request_transaction_data for anything other than transfers/swaps/send
    """
            if not self.system_prompt:
                self.system_prompt = enhanced_system_prompt

            messages = [
                HumanMessage(content=f"Instructions for you: {enhanced_system_prompt}"),
                HumanMessage(content=prompt)
            ]

            initial_state = {
                "messages": messages,
                "context": {}
            }

            config = {"configurable": {"thread_id": "24249321221"}}
            
            collected_response = []
            db_uri = os.getenv('POSTGRES_DB_URI')
            if not db_uri:
                raise GeminiConfigurationError("PostgreSQL connection URI not found in environment")
            # Compile graph with memory checkpointer
            with PostgresSaver.from_conn_string(db_uri) as checkpointer:
                checkpointer.setup()
                graph = self.graph_builder.compile(checkpointer=checkpointer)
                snapshot = graph.get_state(config)
                response_stream = graph.stream(
                    initial_state,
                    config,
                )
                for chunk in response_stream:
                    if isinstance(chunk, dict):
                        if "chatbot" in chunk and "messages" in chunk["chatbot"]:
                            messages = chunk["chatbot"]["messages"]
                            messages
                            if messages and isinstance(messages[-1], (AIMessage, ToolMessage)):
                                collected_response.append(messages[-1].content)
                        elif "tools" in chunk and "messages" in chunk["tools"]:
                            messages = chunk["tools"]["messages"]
                            for message in messages:
                                if isinstance(message, (ToolMessage, AIMessage)):
                                    collected_response.append(message.content)
            return "\n".join(filter(None, collected_response))
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise GeminiAPIError(f"Text generation failed: {str(e)}")
    def continue_execution(self, data: str) -> Any:
        """Continue execution based on provided data"""
        try:
            config = {"configurable": {"thread_id": "24249321221"}}
            
            db_uri = os.getenv('POSTGRES_DB_URI')
            if not db_uri:
                raise GeminiConfigurationError("PostgreSQL connection URI not found in environment")
            
            interrupted_tool_call_id = None
            
            with PostgresSaver.from_conn_string(db_uri) as checkpointer:
                checkpointer.setup()
                graph = self.graph_builder.compile(checkpointer=checkpointer)
                snapshot = graph.get_state(config)
                # Process messages to find interrupted tool call
                messages = snapshot.values["messages"]
                for i, message in enumerate(messages):
                    
                    if isinstance(message, ToolMessage):
                        try:
                            content = json.loads(message.content)
                            if content.get("status") == "interrupt":
                                # Look for the next ToolMessage
                                for next_msg in messages[i+1:]:
                                    if isinstance(next_msg, ToolMessage):
                                        interrupted_tool_call_id = next_msg.tool_call_id
                                        try:
                                            tool_message = AIMessage(
                                                data
                                            )
                                            tool_message.additional_kwargs = {
                                                "tool_call_id": interrupted_tool_call_id
                                            }
                                            response_command = Command(
                                                update={
                                                    "messages": [
                                                        tool_message
                                                    ],
                                                }
                                            )
                                            
                                            response_stream = graph.stream(
                                                response_command,
                                                config,
                                            )
                                        except Exception as e:
                                            logger.error(f"Error in stream: {str(e)}")
                                            raise
                                        break
                                break
                        except json.JSONDecodeError:
                            # Skip messages that can't be parsed as JSON
                            continue
                    
                logger.info(f"Found interrupted tool call ID: {interrupted_tool_call_id}")

                return data
        except Exception as e:
            logger.error(f"Error in continue_execution: {str(e)}")
            raise


    def check_model(self, model: str, **kwargs) -> bool:
        """Check if a specific model is available"""
        try:
            supported_models = [
                "gemini-1.5-pro",
                "gemini-1.5-flash",
            ]
            return model in supported_models
                
        except Exception as e:
            raise GeminiAPIError(f"Model check failed: {e}")

    def list_models(self, **kwargs) -> None:
        """List all available Gemini models"""
        try:
            supported_models = [
                "gemini-1.5-pro",
                "gemini-1.5-flash",
            ]

            logger.info("\nAVAILABLE MODELS:")
            for i, model_id in enumerate(supported_models, start=1):
                logger.info(f"{i}. {model_id}")
                    
        except Exception as e:
            raise GeminiAPIError(f"Listing models failed: {e}")
        
    def interrupt_chat(self, query: str) -> Any:
        logger.info(query)
        response = interrupt({query: "query"})
        return response["data"]

    def perform_action(self, action_name: str, kwargs) -> Any:
        """Execute a Gemini action with validation"""
        if action_name not in self.actions:
            raise KeyError(f"Unknown action: {action_name}")

        load_dotenv()
        
        if not self.is_configured(verbose=True):
            raise GeminiConfigurationError("Gemini is not properly configured")

        action = self.actions[action_name]
        errors = action.validate_params(kwargs)
        if errors:
            raise ValueError(f"Invalid parameters: {', '.join(errors)}")

        method_name = action_name.replace('-', '_')
        method = getattr(self, method_name)
        return method(**kwargs)