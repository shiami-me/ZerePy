import logging
import os
from typing import Dict, Any, List
from dotenv import load_dotenv, set_key
from together import Together
import base64
from PIL import Image
import io
from datetime import datetime

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
        self.register_actions()

    @property
    def is_llm_provider(self) -> bool:
        return True

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

    def generate_image(
        self,
        prompt: str,
        model: str = "black-forest-labs/FLUX.1-schnell-Free",
        width: int = 768,
        height: int = 768,
        steps: int = 4,
        n: int = 1,
        **kwargs
    ) -> List[str]:
        """Generate images using Together AI"""
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

            # Create output directory if it doesn't exist
            output_dir = "generated_images"
            os.makedirs(output_dir, exist_ok=True)

            image_paths = []
            
            for idx, image_data in enumerate(response.data):
                # Decode base64 image
                image_bytes = base64.b64decode(image_data.b64_json)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(output_dir, f"together_ai_{timestamp}_{idx}.png")
                image.save(image_path)
                image_paths.append(image_path)

            return image_paths

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

        # Call the appropriate method based on action name
        method_name = action_name.replace('-', '_')
        method = getattr(self, method_name)
        return method(**kwargs)
