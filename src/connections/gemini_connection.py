import logging
import os
from typing import Dict, Any, List, Optional, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv, set_key
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from src.tools.sonic_tools import SONIC_SYSTEM_PROMPT, get_sonic_tools
from src.tools.together_tools import TOGETHER_SYSTEM_PROMPT, get_together_tools

import json
from src.connections.base_connection import BaseConnection, Action, ActionParameter

logger = logging.getLogger("connections.gemini_connection")

class State(TypedDict):
    """Type for tracking conversation state"""
    messages: Annotated[list, add_messages]
    context: Dict[str, Any]

class GeminiConnectionError(Exception):
    """Base exception for Gemini connection errors"""
    pass

class GeminiConfigurationError(GeminiConnectionError):
    """Raised when there are configuration/credential issues"""
    pass

class GeminiAPIError(GeminiConnectionError):
    """Raised when Gemini API requests fail"""
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

class GeminiConnection(BaseConnection):
    def __init__(self, config: Dict[str, Any], agent):
        super().__init__(config)
        self._client = None
        self.system_prompt = None
        self.register_actions()
        self._agent = agent
        self.setup_tools()
        # Initialize MemorySaver
        self.memory = MemorySaver()
        self.graph = self._create_conversation_graph()

    def setup_tools(self):
        """Initialize tools for the connection"""
        self.tools = []
        load_dotenv()
        if "sonic" in self.config.get("plugins", []):
            sonic_tools = get_sonic_tools()
            self.tools.extend(sonic_tools)       
        
        if "image" in self.config.get("plugins", []):
            image_tools = get_together_tools(self._agent)
            self.tools.extend(image_tools)                      
        if self.config.get("tavily", False) and os.getenv('TAVILY_API_KEY'):
            max_results = self.config.get("max_tavily_results", 2)
            logger.info(f"Initializing Tavily search with max_results: {max_results}")
            self.search_tool = TavilySearchResults(api_key=os.getenv('TAVILY_API_KEY'), max_results=max_results)
            self.tools.append(self.search_tool)

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
            
            llm_with_tools = llm.bind_tools(self.tools)
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
        """Validate Gemini configuration from JSON"""
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
        
        # Only ask for Tavily if enabled in config
        tavily_api_key = None
        if self.config.get("tavily", False):
            logger.info("Tavily: https://tavily.com")
            tavily_api_key = os.getenv('TAVILY_API_KEY')

        try:
            if not os.path.exists('.env'):
                with open('.env', 'w') as f:
                    f.write('')

            set_key('.env', 'GEMINI_API_KEY', google_api_key)
            
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
        """Check if Gemini API key is configured and valid"""
        try:
            load_dotenv()
            api_key = os.getenv('GEMINI_API_KEY')
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
            # Combine system prompts if Sonic tools are available
            has_sonic_tools = any(tool.name.startswith('sonic_') for tool in self.tools)
            has_together_tools = any(tool.name.startswith('together_') for tool in self.tools)
            
            enhanced_system_prompt = system_prompt
            if has_sonic_tools:
                enhanced_system_prompt = f"""
    {system_prompt}
    
    Even if you have the plugins, don't forget that you can also work like a normal chatbot.
    Also give emojis in your outputs when necessary. Keep it well formatted.
    Use the plugins when it's needed - 
    {SONIC_SYSTEM_PROMPT}
    
    """
            if has_together_tools:
                enhanced_system_prompt = f"""
    {system_prompt}

    Even if you have the plugins, don't forget that you can also work like a normal chatbot.
    Also give emojis in your outputs when necessary. Keep it well formatted.
    Use the plugins when it's needed -
    {TOGETHER_SYSTEM_PROMPT}

    """
            
            # Store or update system prompt if it's new
            if not self.system_prompt:
                self.system_prompt = enhanced_system_prompt
                
            # Convert system prompt and user prompt into a single human message
            combined_prompt = f"""System: {self.system_prompt}"""
            
            # Convert message history to the format expected by LangGraph
            messages = []
            
            # Always include system prompt at the start
            messages.append(SystemMessage(content=combined_prompt))

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
            raise GeminiAPIError(f"Text generation failed: {str(e)}")

    def check_model(self, model: str, **kwargs) -> bool:
        """Check if a specific model is available"""
        try:
            # List of supported Gemini models
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
            # List supported models
            supported_models = [
                "gemini-1.5-pro",
                "gemini-1.5-flash",
            ]

            logger.info("\nAVAILABLE MODELS:")
            for i, model_id in enumerate(supported_models, start=1):
                logger.info(f"{i}. {model_id}")
                    
        except Exception as e:
            raise GeminiAPIError(f"Listing models failed: {e}")
    
    def perform_action(self, action_name: str, kwargs) -> Any:
        """Execute a Gemini action with validation"""
        if action_name not in self.actions:
            raise KeyError(f"Unknown action: {action_name}")

        # Explicitly reload environment variables
        load_dotenv()
        
        if not self.is_configured(verbose=True):
            raise GeminiConfigurationError("Gemini is not properly configured")

        action = self.actions[action_name]
        errors = action.validate_params(kwargs)
        if errors:
            raise ValueError(f"Invalid parameters: {', '.join(errors)}")

        # Call the appropriate method based on action name
        method_name = action_name.replace('-', '_')
        method = getattr(self, method_name)
        return method(**kwargs)
