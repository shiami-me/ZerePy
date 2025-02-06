import logging
import os
from typing import Dict, Any
from dotenv import load_dotenv, set_key
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from src.connections.llm_base_connection import LLMBaseConnection

logger = logging.getLogger("connections.groq_connection")

class GroqConnectionError(Exception):
    """Base exception for Groq connection errors"""
    pass

class GroqConnection(LLMBaseConnection):
    def __init__(self, config: Dict[str, Any], agent):
        super().__init__(config, agent)
        self.is_async = True  # Indicate this is an async implementation

    def get_llm_identifier(self) -> str:
        return "groq"

    def _get_client(self) -> ChatGroq:
        """Get or create Groq client"""
        if not self._client:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise GroqConnectionError("Groq API key not found in environment")
            
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
        try:
            load_dotenv()
            api_key = os.getenv('GROQ_API_KEY')
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

    def list_models(self, **kwargs) -> None:
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


