import logging
import os
from typing import Dict, Any, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv, set_key
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.prebuilt import create_react_agent
from src.tools.sonic_tools import get_sonic_tools
from src.tools.together_tools import get_together_tools
from src.connections.base_connection import BaseConnection, Action, ActionParameter
import asyncio

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

class GroqConnection(BaseConnection):
    def __init__(self, config: Dict[str, Any], agent):
        super().__init__(config)
        self._client = None
        self.system_prompt = None
        self.search_tool = None
        self.register_actions()
        self._agent = agent
        self.setup_tools()

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
            sonic_tools = get_sonic_tools(self._agent, llm="qroq")
            self.tools.extend(sonic_tools)       
        
        if "image" in self.config.get("plugins", []):
            image_tools = get_together_tools(self._agent)
            self.tools.extend(image_tools)

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
                    ActionParameter("temperature", False, float, "Temperature for generation")
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
        postgres_uri = input("\nEnter your PostgreSQL connection URI: ")
        
        # Only ask for Tavily if enabled in config
        tavily_api_key = None
        if self.config.get("tavily", False):
            logger.info("Tavily: https://tavily.com")
            tavily_api_key = input("\nEnter your Tavily API key: ")

        try:
            if not os.path.exists('.env'):
                with open('.env', 'w') as f:
                    f.write('')

            set_key('.env', 'GROQ_API_KEY', groq_api_key)
            set_key('.env', 'POSTGRES_URI', postgres_uri)
            if tavily_api_key:
                set_key('.env', 'TAVILY_API_KEY', tavily_api_key)
            
            # Validate the configurations
            self._client = None
            self._get_client()
            
            # Test Tavily if enabled
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
            api_key = os.getenv('GROQ_API_KEY')
            db_uri = os.getenv('POSTGRES_DB_URI')
            
            if not api_key or not db_uri:
                return False

            # Check Tavily if enabled
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
        
    async def generate_text(
        self,
        prompt: str,
        system_prompt: str,
        model: str = None,
        stream: bool = True,
        **kwargs
    ) -> str:
        try:
            # Combine all system prompts
            enhanced_system_prompt = f"""
{system_prompt}

You are a helpful assistant with access to various tools. When using tools:
1. Don't explain what you're doing, just do it
2. Don't ask for confirmation, just execute
3. Don't mention the tool names
4. Keep responses natural and concise
5. Use multiple tools when needed
6. When you need some external information or the user asks for it, use Tavily search tool when available.
7. Use connect wallet address whenever it's needed, for example - for sonic related tools. Ask user to connect wallet if needed(only for sonic related tools).

"""
            # Store or update system prompt if it's new
            if not self.system_prompt:
                self.system_prompt = enhanced_system_prompt

            # Create messages for the agent
            messages = [
                SystemMessage(content=enhanced_system_prompt),
                HumanMessage(content=prompt)
            ]

            # Initialize state with conversation history
            initial_state = {
                "messages": messages,
                "context": {}
            }

            config = {"configurable": {"thread_id": "1"}}
            
            collected_response = []
            db_uri = os.getenv('POSTGRES_DB_URI')
            if not db_uri:
                raise GroqConfigurationError("PostgreSQL connection URI not found in environment")
            async with AsyncPostgresSaver.from_conn_string(db_uri) as checkpointer:
                agent = create_react_agent(
                    self._get_client(),
                    self.tools,
                    checkpointer=checkpointer
                )
                async for chunk in agent.astream(initial_state, config):
                    if isinstance(chunk, dict):
                        if "agent" in chunk and "messages" in chunk["agent"]:
                            messages = chunk["agent"]["messages"]
                            if messages and isinstance(messages[-1], (AIMessage, ToolMessage)):
                                collected_response.append(messages[-1].content)
                        elif "tools" in chunk and "messages" in chunk["tools"]:
                            messages = chunk["tools"]["messages"]
                            if messages and isinstance(messages[-1], (AIMessage, ToolMessage)):
                                collected_response.append(messages[-1].content)
                                
                    checkpoint_tuples = [c async for c in checkpointer.alist(config)]
                    logger.debug(f"Checkpoint history: {checkpoint_tuples}")

            return "\n".join(filter(None, collected_response))
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise GroqAPIError(f"Text generation failed: {str(e)}")

    async def check_model(self, model: str, **kwargs) -> bool:
        """Check if a specific model is available"""
        try:
            supported_models = [
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama-guard-3-8b",
                "llama3-70b-8192",
                "llama3-8b-8192",
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

    async def list_models(self, **kwargs) -> None:
        """List all available Groq models"""
        try:
            supported_models = [
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama-guard-3-8b",
                "llama3-70b-8192",
                "llama3-8b-8192",
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

        load_dotenv()
        
        if not self.is_configured(verbose=True):
            raise GroqConfigurationError("Groq is not properly configured")

        action = self.actions[action_name]
        errors = action.validate_params(kwargs)
        if errors:
            raise ValueError(f"Invalid parameters: {', '.join(errors)}")

        method_name = action_name.replace('-', '_')
        method = getattr(self, method_name)
                
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(method(**kwargs))
        except Exception as e:
            raise GroqAPIError(f"Action failed: {str(e)}")


