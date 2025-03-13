from langchain.tools import BaseTool
import json
import logging
from typing import Optional
from src.action_handler import execute_action

logger = logging.getLogger("tools.together_tools")


class TogetherImageGenerationTool(BaseTool):
    name: str = "generate_image"
    description: str = """
    Generate images using Together AI's image generation models.
    Output only the image. Do not output the tool you're using.
    Input should be a JSON string with:
    - prompt: Text description of the image to generate
    - width: (optional) Image width in pixels (default: 768)
    - height: (optional) Image height in pixels (default: 768)
    
    Examples:
    - "Generate a realistic photo of a cat"
    - "Create digital art of a futuristic city"
    - "Make an image of mountains at sunset"
    
    Return Image URL: https://ipfs.io/ipfs/<ipfs_hash>
    """
    return_direct: bool = True
    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def _run(
        self,
        prompt: str,
        width: Optional[int] = 768,
        height: Optional[int] = 768,
    ) -> str:
        """
        Generate images using Together AI
        
        Args:
            prompt (str): Text description of the image to generate
            width (int, optional): Image width in pixels. Defaults to 768.
            height (int, optional): Image height in pixels. Defaults to 768.
            steps (int, optional): Number of inference steps. Defaults to 4.
            
        Returns:
            str: JSON string containing status and image paths
        """
        try:
            logger.info(f"Generating image(s) with prompt: {prompt}")
            
            # Generate images using execute_action
            response = execute_action(
                agent=self._agent,
                action_name="together-generate-image",
                prompt = prompt,
                model = "black-forest-labs/FLUX.1-schnell-Free",
                width = width,
                height = height,
                steps = 3,
                n = 1
            )
            logger.info(f"Response: {response}")
            if not response or "error" in response:
                raise Exception(response.get("error", "Unknown error occurred"))

            
            result = {
                "status": "success",
                "message": f"Generated image",
                "type": "image",
                "ipfs_hash": response,
                "parameters": {
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "steps": 3
                }
            }
            
            logger.info(f"Successfully generated image")
            return json.dumps(result)
            
        except Exception as e:
            error_msg = f"Failed to generate image: {str(e)}"
            logger.error(error_msg)
            return json.dumps({
                "status": "error",
                "message": error_msg,
                "parameters": {
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "steps": 3
                }
            })

def get_together_tools(agent) -> list:
    """
    Initialize and return Together AI tools
    
    Args:
        execute_action: Function to execute actions
    
    Returns:
        list: List of initialized Together AI tools
    """
    return [TogetherImageGenerationTool(agent)]
