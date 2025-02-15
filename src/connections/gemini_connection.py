import logging
import os
from typing import Dict, Any
from dotenv import load_dotenv, set_key
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from src.connections.llm_base_connection import LLMBaseConnection

logger = logging.getLogger("connections.gemini_connection")

class GeminiConnectionError(Exception):
    """Base exception for Gemini connection errors"""
    pass

class GeminiConfigurationError(GeminiConnectionError):
    """Raised when there are configuration/credential issues"""
    pass

class GeminiAPIError(GeminiConnectionError):
    """Raised when Gemini API requests fail"""
    pass

class GeminiConnection(LLMBaseConnection):
    def __init__(self, config: Dict[str, Any], agent, tools: bool = True):
        super().__init__(config, agent, tools)

    def get_llm_identifier(self) -> str:
        return "gemini"

    def _get_client(self) -> ChatGoogleGenerativeAI:
        """Get or create Gemini client"""
        if not self._client:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise GeminiConnectionError("Google API key not found in environment")
            
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
            
            # Validate configurations
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

    def check_model(self, model: str, **kwargs) -> bool:
        """Check if a specific model is available"""
        try:
            supported_models = [
                "gemini-1.5-pro",
                "gemini-1.5-flash",
            ]
            return model in supported_models
                
        except Exception as e:
            raise GeminiConnectionError(f"Model check failed: {e}")

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
            raise GeminiConnectionError(f"Listing models failed: {e}")