import logging
import os
from typing import Dict, Any, List
from dotenv import load_dotenv, set_key
from together import Together
import base64
from PIL import Image
import io
import asyncio
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models.image import Base, GeneratedImage
from src.helpers.pinata import PinataStorage
from src.connections.base_connection import BaseConnection, Action, ActionParameter

logger = logging.getLogger("connections.together_ai_connection")

class TogetherAIConnectionError(Exception):
    """Base exception for Together AI connection errors"""
    pass

class TogetherAIConfigurationError(TogetherAIConnectionError):
    """Raised when there are configuration/credential issues"""
    pass

class TogetherAIAPIError(TogetherAIConnectionError):
    """Raised when Together AI API requests fail"""
    pass

class TogetherAIConnection(BaseConnection):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._client = None
        self._db_engine = None
        self._Session = None
        self.setup_database()
        self.register_actions()
        self.pinata = PinataStorage()

    @property
    def is_llm_provider(self) -> bool:
        return True

    def setup_database(self):
        """Initialize database connection"""
        database_url = os.getenv('POSTGRES_DB_URI')
        if not database_url:
            raise TogetherAIConfigurationError("POSTGRES_DB_URI not found in environment")
        
        self._db_engine = create_engine(database_url)
        Base.metadata.create_all(self._db_engine)
        self._Session = sessionmaker(bind=self._db_engine)

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Together AI configuration"""
        return config

    def register_actions(self) -> None:
        """Register available Together AI actions"""
        self.actions = {
            "generate-image": Action(
                name="generate-image",
                parameters=[
                    ActionParameter("prompt", True, str, "The text prompt for image generation"),
                    ActionParameter("model", False, str, "Model to use for image generation"),
                    ActionParameter("width", False, int, "Image width (default: 768)"),
                    ActionParameter("height", False, int, "Image height (default: 768)"),
                    ActionParameter("steps", False, int, "Number of inference steps (default: 4)"),
                    ActionParameter("n", False, int, "Number of images to generate (default: 1)")
                ],
                description="Generate images using Together AI models"
            )
        }

    def _get_client(self) -> Together:
        """Get or create Together AI client"""
        if not self._client:
            api_key = os.getenv("TOGETHER_API_KEY")
            if not api_key:
                raise TogetherAIConfigurationError("Together AI API key not found in environment")
            
            self._client = Together(api_key=api_key)
        return self._client

    def configure(self) -> bool:
        """Sets up Together AI authentication"""
        logger.info("\n🤖 API SETUP")

        if self.is_configured():
            logger.info("\nAPI is already configured.")
            response = input("Do you want to reconfigure? (y/n): ")
            if response.lower() != 'y':
                return True

        logger.info("\n📝 To get your API credentials:")
        logger.info("Together AI: https://www.together.ai/")
        
        together_api_key = input("\nEnter your Together AI API key: ")

        try:
            if not os.path.exists('.env'):
                with open('.env', 'w') as f:
                    f.write('')

            set_key('.env', 'TOGETHER_API_KEY', together_api_key)
            
            # Validate the API key
            self._client = None  # Reset client
            self._get_client()

            logger.info("\n✅ API configuration successfully saved!")
            logger.info("Your API key has been stored in the .env file.")
            return True

        except Exception as e:
            logger.error(f"Configuration failed: {e}")
            return False

    def is_configured(self, verbose = False) -> bool:
        """Check if Together AI API key is configured and valid"""
        try:
            load_dotenv()
            api_key = os.getenv('TOGETHER_API_KEY')
            if not api_key:
                return False

            # Try to initialize client
            self._client = None  # Reset client
            self._get_client()
            return True
            
        except Exception as e:
            if verbose:
                logger.debug(f"Configuration check failed: {e}")
            return False

    async def generate_image(
        self,
        prompt: str,
        model: str = "black-forest-labs/FLUX.1-schnell-Free",
        width: int = 768,
        height: int = 768,
        steps: int = 4,
        n: int = 1,
        **kwargs
    ) -> List[str]:
        """Generate images using Together AI and store in database"""
        try:
            client = self._get_client()
            
            response = client.images.generate(
                prompt=prompt,
                model=model,
                width=width,
                height=height,
                steps=steps,
                n=n,
                response_format="b64_json",
                update_at="2025-02-03T07:18:13.665Z"
            )

            session = self._Session()
            ipfs_hash = None
            try:
                for idx, image_data in enumerate(response.data):
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"together_ai_{timestamp}_{idx}.png"
                    result = await self.pinata.pin_image(image_data.b64_json, filename)
                    logger.info(result)
                    ipfs_hash = result["IpfsHash"]
                    # Store in database
                    db_image = GeneratedImage(
                        filename=filename,
                        ipfs_hash=result["IpfsHash"],
                        prompt=prompt,
                        model=model
                    )
                    session.add(db_image)
                if ipfs_hash:
                    session.commit()
                    return ipfs_hash
                else:
                    raise TogetherAIAPIError("Image generation failed: No IPFS hash returned")

            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()

        except Exception as e:
            raise TogetherAIAPIError(f"Image generation failed: {str(e)}")
    
    def perform_action(self, action_name: str, kwargs) -> Any:
        """Execute a Together AI action with validation"""
        if action_name not in self.actions:
            raise KeyError(f"Unknown action: {action_name}")

        # Explicitly reload environment variables
        load_dotenv()
        
        if not self.is_configured(verbose=True):
            raise TogetherAIConfigurationError("Together AI is not properly configured")

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
            raise TogetherAIAPIError(f"Action failed: {str(e)}")
