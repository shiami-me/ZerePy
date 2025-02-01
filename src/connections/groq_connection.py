import logging
import os
from typing import Dict, Any, List, Optional, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv, set_key
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

import json
from src.connections.base_connection import BaseConnection, Action, ActionParameter

logger = logging.getLogger("connections.groq_connection")

class State(TypedDict):
    """Type for tracking conversation state"""
    messages: Annotated[list, add_messages]
    context: Dict[str, Any]

class GroqConnectionError(Exception):
    """Base exception for Groq connection errors"""
    pass

class GroqConfigurationError(GroqConnectionError):
    """Raised when there are configuration/credential issues"""
    pass

class GroqAPIError(GroqConnectionError):
    """Raised when Groq API requests fail"""
    pass

class BasicToolNode:
    """Tool execution node for LangGraph"""
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        messages = inputs.get("messages", [])
        if not messages:
            raise ValueError("No message found in input")
        
        message = messages[-1]
        outputs = []
        
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            
            tool_content = json.dumps(tool_result)
                
            outputs.append(
                ToolMessage(
                    content=tool_content,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

class GroqConnection(BaseConnection):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._client = None
        self.system_prompt = None
        self.register_actions()
        self.setup_tools()
        # Initialize MemorySaver
        self.memory = MemorySaver()
        self.graph = self._create_conversation_graph()

    def setup_tools(self):
        """Initialize tools for the connection"""
        self.tools = []
        if self.config.get("tavily", False) and os.getenv('TAVILY_API_KEY'):
            load_dotenv()                                            
            max_results = self.config.get("max_tavily_results", 2)
            logger.info(f"Initializing Tavily search with max_results: {max_results}")
            self.search_tool = TavilySearchResults(api_key=os.getenv('TAVILY_API_KEY'), max_results=max_results)
            self.tools.append(self.search_tool)


    def _create_conversation_graph(self) -> StateGraph:
        """Create the conversation flow graph"""
        
        def chatbot(state: State):
            """Generate response using Groq"""
            llm = self._get_client()
            messages = state["messages"]
            
            # If there are tool messages, create a more detailed summary prompt
            tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
            if tool_messages:
                context = "\n\nI found multiple relevant sources:\n" + "\n".join(
                    [msg.content for msg in tool_messages]
                ) + "\n\nPlease consider all these sources in your response."
                
                # Add context to the last user message
                for i in reversed(range(len(messages))):
                    if isinstance(messages[i], HumanMessage):
                        messages[i].content += context
                        break
            
            llm_with_tools = llm.bind_tools(self.tools)
            response = llm_with_tools.invoke(messages)
            
            return {"messages": [response]}

        def route_tools(state: State):
            """Route based on whether tools are needed"""
            if isinstance(state, list):
                ai_message = state[-1]
            elif messages := state.get("messages", []):
                ai_message = messages[-1]
            else:
                raise ValueError(f"No messages found in input state: {state}")
            
            if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
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
        
        # Compile graph with memory checkpointer
        return graph_builder.compile(checkpointer=self.memory)


    @property
    def is_llm_provider(self) -> bool:
        return True

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Groq configuration from JSON"""
        required_fields = ["model"]
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {', '.join(missing_fields)}")
            
        if not isinstance(config["model"], str):
            raise ValueError("model must be a string")
            
        # Ensure tavily is boolean if present
        if "tavily" in config and not isinstance(config["tavily"], bool):
            raise ValueError("tavily configuration must be a boolean")
                
        return config

    def register_actions(self) -> None:
        """Register available Groq actions"""
        self.actions = {
            "generate-text": Action(
                name="generate-text",
                parameters=[
                    ActionParameter("prompt", True, str, "The input prompt for text generation"),
                    ActionParameter("system_prompt", True, str, "System prompt to guide the model"),
                    ActionParameter("model", False, str, "Model to use for generation"),
                    ActionParameter("temperature", False, float, "A decimal number that determines the degree of randomness in the response.")
                ],
                description="Generate text using Groq models"
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
                description="List all available Groq models"
            )
        }

    def _get_client(self) -> ChatGroq:
        """Get or create Groq client"""
        if not self._client:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise GroqConfigurationError("Groq API key not found in environment")
            
            self._client = ChatGroq(
                api_key=api_key,
                model_name=self.config["model"],
                temperature=self.config.get("temperature", 0.7)
            )
        return self._client

    def configure(self) -> bool:
        """Sets up Groq and Tavily API authentication"""
        logger.info("\n🤖 API SETUP")

        if self.is_configured():
            logger.info("\nAPIs are already configured.")
            response = input("Do you want to reconfigure? (y/n): ")
            if response.lower() != 'y':
                return True

        logger.info("\n📝 To get your API credentials:")
        logger.info("Groq: https://console.groq.com")
        
        groq_api_key = input("\nEnter your Groq API key: ")
        
        # Only ask for Tavily if enabled in config
        tavily_api_key = None
        if self.config.get("tavily", False):
            logger.info("Tavily: https://tavily.com")
            tavily_api_key = os.getenv('TAVILY_API_KEY')

        try:
            if not os.path.exists('.env'):
                with open('.env', 'w') as f:
                    f.write('')

            set_key('.env', 'GROQ_API_KEY', groq_api_key)
            
            # Validate the API keys
            self._client = None  # Reset client
            self._get_client()
            
            # Test Tavily only if enabled
            if self.config.get("tavily", False) and tavily_api_key:
                self.search_tool = TavilySearchResults(api_key=tavily_api_key, max_results=self.config.get("max_tavily_results", 2))

            logger.info("\n✅ API configuration successfully saved!")
            logger.info("Your API keys have been stored in the .env file.")
            return True

        except Exception as e:
            logger.error(f"Configuration failed: {e}")
            return False

    def is_configured(self, verbose = False) -> bool:
        """Check if Groq API key is configured and valid"""
        try:
            load_dotenv()
            api_key = os.getenv('GROQ_API_KEY')
            tavily_api_key = os.getenv('TAVILY_API_KEY')
            if not api_key or not tavily_api_key:
                return False

            # Try to initialize client
            self._client = None  # Reset client
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
            # Store or update system prompt if it's new
            if not self.system_prompt:
                self.system_prompt = system_prompt
            
            # Convert message history to the format expected by LangGraph
            messages = []
            
            # Always include system prompt at the start
            messages.append(SystemMessage(content=self.system_prompt))

            # Add the current prompt
            messages.append(HumanMessage(content=prompt))
            
            # Initialize state with conversation history
            initial_state = {
                "messages": messages,
                "context": {}
            }
            
            # Process through graph with thread_id for memory persistence
            config = {"configurable": {"thread_id": str(id(self))}}
            response_stream = self.graph.stream(
                initial_state,
                config,
                stream_mode="values"
            )
            
            collected_response = []
            for event in response_stream:
                if "messages" in event:
                    message = event["messages"][-1]
                    if isinstance(message, (AIMessage, ToolMessage)):
                        collected_response.append(message.content)
            
            generated_text = "\n".join(collected_response)

            return generated_text
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise GroqAPIError(f"Text generation failed: {str(e)}")

    def check_model(self, model: str, **kwargs) -> bool:
        """Check if a specific model is available"""
        try:
            # List of supported Groq models
            supported_models = [
                "mixtral-8x7b-32768",
                "distil-whisper-large-v3-en",
                "gemma2-9b-it",
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama-guard-3-8b",
                "llama3-70b-8192",
                "llama3-8b-8192",
                "whisper-large-v3",
                "whisper-large-v3-turbo",
                # Preview Models
                "deepseek-r1-distill-llama-70b",
                "llama-3.3-70b-specdec",
                "llama-3.2-1b-preview",
                "llama-3.2-3b-preview",
                "llama-3.2-11b-vision-preview",
                "llama-3.2-90b-vision-preview"
            ]

            return model in supported_models
                
        except Exception as e:
            raise GroqAPIError(f"Model check failed: {e}")

    def list_models(self, **kwargs) -> None:
        """List all available Groq models"""
        try:
            # List supported models
            supported_models = [
                "mixtral-8x7b-32768",
                "distil-whisper-large-v3-en",
                "gemma2-9b-it",
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama-guard-3-8b",
                "llama3-70b-8192",
                "llama3-8b-8192",
                "whisper-large-v3",
                "whisper-large-v3-turbo",
                # Preview Models
                "deepseek-r1-distill-llama-70b",
                "llama-3.3-70b-specdec",
                "llama-3.2-1b-preview",
                "llama-3.2-3b-preview",
                "llama-3.2-11b-vision-preview",
                "llama-3.2-90b-vision-preview"
            ]

            logger.info("\nAVAILABLE MODELS:")
            for i, model_id in enumerate(supported_models, start=1):
                logger.info(f"{i}. {model_id}")
                    
        except Exception as e:
            raise GroqAPIError(f"Listing models failed: {e}")
    
    def perform_action(self, action_name: str, kwargs) -> Any:
        """Execute a Groq action with validation"""
        if action_name not in self.actions:
            raise KeyError(f"Unknown action: {action_name}")

        # Explicitly reload environment variables
        load_dotenv()
        
        if not self.is_configured(verbose=True):
            raise GroqConfigurationError("Groq is not properly configured")

        action = self.actions[action_name]
        errors = action.validate_params(kwargs)
        if errors:
            raise ValueError(f"Invalid parameters: {', '.join(errors)}")

        # Call the appropriate method based on action name
        method_name = action_name.replace('-', '_')
        method = getattr(self, method_name)
        return method(**kwargs)
