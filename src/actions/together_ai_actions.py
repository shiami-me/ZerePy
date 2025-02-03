from src.action_handler import register_action
import logging

logger = logging.getLogger("actions.together_ai_actions")

@register_action("together-generate-image")
def together_generate_image(agent, **kwargs):
    """Generate images using Together AI models"""
    agent.logger.info("\n🎨 GENERATING IMAGE WITH TOGETHER AI")
    try:
        # Extract parameters with defaults
        prompt = kwargs.get('prompt')
        model = kwargs.get('model', "black-forest-labs/FLUX.1-schnell-Free")
        width = kwargs.get('width', 768)
        height = kwargs.get('height', 768)
        steps = kwargs.get('steps', 4)
        n = kwargs.get('n', 1)

        if not prompt:
            raise ValueError("Prompt is required for image generation")

        result = agent.connection_manager.perform_action(
            connection_name="together",
            action_name="generate-image",
            params = [
                prompt, model, width, height, steps, n
            ]
        )
        
        if result:
            agent.logger.info("✅ Image generation completed!")
            agent.logger.info(f"📁 Images saved to: {', '.join(result)}")
        return result
    except Exception as e:
        agent.logger.error(f"❌ Image generation failed: {str(e)}")
        return None
