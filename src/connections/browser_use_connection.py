import logging
from typing import Dict, Any, Optional
from browser_use import Agent, Browser, BrowserConfig, Controller
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
import os
import asyncio

from dotenv import load_dotenv, set_key

from src.connections.base_connection import BaseConnection, Action, ActionParameter

logger = logging.getLogger("agent")

class BrowserConnectionError(Exception):
    """Base exception for Browser connection errors"""
    pass

class BrowserConfigurationError(BrowserConnectionError):
    """Raised when there are configuration/credential issues"""
    pass

class BrowserAPIError(BrowserConnectionError):
    """Raised when Browser operations fail"""
    pass

class BrowserUseConnection(BaseConnection):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._agent: Optional[Agent] = None
        self._browser: Optional[Browser] = None
        self._controller: Optional[Controller] = None
        self._headless = config.get("headless", True)
        self._client = None

    @property
    def is_llm_provider(self) -> bool:
        return False

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate browser-use configuration"""
     
        if not isinstance(config.get("headless", True), bool):
            raise BrowserConfigurationError("'headless' must be a boolean value")

        return config

    def register_actions(self):
        """Register available browser-use actions"""
        self.actions = {
            "browse": Action(
                name="browse",
                description="Execute a browsing task using the agent",
                parameters=[
                    ActionParameter(
                        "task",
                        True,
                        str,
                        "The task to execute",
                    ),
                ],
            )
        }

    def configure(self) -> bool:
        """Sets up Browser and Gemini API authentication"""
        logger.info("\n🤖 API SETUP")

        if self.is_configured():
            logger.info("\nAPIs are already configured.")
            response = input("Do you want to reconfigure? (y/n): ")
            if response.lower() != 'y':
                return True

        logger.info("\n📝 To get your API credentials:")
        logger.info("Gemini: https://aistudio.google.com/app/apikey")
        
        gemini_api_key = input("\nEnter your Gemini API key: ")

        try:
            if not os.path.exists('.env'):
                with open('.env', 'w') as f:
                    f.write('')

            # Save API key
            set_key('.env', 'GEMINI_API_KEY', gemini_api_key)
            
            # Validate the configuration by initializing components
            self._client = None  # Reset client
            
            # Test configuration by initializing
            self._get_client()

            logger.info("\n✅ Configuration successfully saved!")
            logger.info("Your API key has been stored in the .env file.")
            return True

        except Exception as e:
            logger.error(f"Configuration failed: {e}")
            return False

    def is_configured(self, verbose=False) -> bool:
        """Check if Gemini API key and browser configuration are valid"""
        try:
            load_dotenv()
            api_key = os.getenv('GEMINI_API_KEY')
            
            if not api_key:
                if verbose:
                    logger.debug("Gemini API key not found")
                return False

            # Try to initialize client and browser
            self._client = None  # Reset client
            
            self._get_client()
            return True
            
        except Exception as e:
            if verbose:
                logger.debug(f"Configuration check failed: {e}")
            return False

    def _get_client(self):
        """Initialize and return the Gemini client"""
        if not self._client:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise BrowserConfigurationError("GEMINI API key not found in environment")
            
            try:
                self._client = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    api_key=SecretStr(api_key)
                )
            except Exception as e:
                raise BrowserConfigurationError(f"Failed to initialize Gemini client: {str(e)}")
        
        return self._client

    def perform_action(self, action_name: str, kwargs: Dict[str, Any]) -> Any:
        """Execute the specified action with given parameters"""
        if action_name not in self.actions:
            raise KeyError(f"Unknown action: {action_name}")
        action = self.actions[action_name]
        errors = action.validate_params(kwargs)
        if errors:
            raise ValueError(f"Invalid parameters: {', '.join(errors)}")

        try:
            method_name = action_name.replace("-", "_")
            method = getattr(self, method_name)
            return method(**kwargs)
        except Exception as e:
            logger.error(f"Action '{action_name}' failed: {str(e)}")
            raise BrowserAPIError(f"Action execution failed: {str(e)}")

    def browse(self, task: str) -> Dict[str, Any]:
        """Execute a browsing task using the agent"""
        if not self.is_configured():
            raise BrowserConfigurationError("Browser agent is not configured")
        logger.info(f"Executing task: {task}")
        try:
                        
            # Configure browser using config headless value
            self._browser = Browser(
                config=BrowserConfig(
                    headless=self._headless,  # Using the value from config
                    disable_security=True
                )
            )

            self._agent = Agent(
                task = (
                    task
                ),
                llm = self._client,
                browser = self._browser
            )
            result = asyncio.run(self._agent.run())
            return {
                "status": "success",
                "result": result
            }
        except Exception as e:
            logger.error(f"Browse operation failed: {str(e)}")
            raise BrowserAPIError(f"Browse operation failed: {str(e)}")
