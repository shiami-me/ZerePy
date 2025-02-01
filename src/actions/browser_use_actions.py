import time
import asyncio
from src.action_handler import register_action
from src.helpers import print_h_bar
from src.prompts import BROWSE_WEB_PROMPT  # You'll need to create this prompt

@register_action("browse")
def browse(agent, **kwargs):
    """Execute a web browsing task using browser-use agent"""
    agent.logger.info("\n🌐 EXECUTING WEB BROWSING TASK")
    print_h_bar()

    task = kwargs.get("task")
    if not task:
        agent.logger.error("Task description is required")
        return False

    # Generate browsing task using LLM if needed
    if kwargs.get("use_llm", False):
        prompt = BROWSE_WEB_PROMPT.format(agent_name=agent.name, task=task)
        task = agent.prompt_llm(prompt)
        if not task:
            agent.logger.error("Failed to generate browsing task")
            return False

    agent.logger.info(f"\n🔍 Executing task: '{task}'")
    
    try:
        result = agent.connection_manager.perform_action(
            connection_name="browser_use",
            action_name="browse",
            params={"task": task}
        )
        
        if result.get("status") == "success":
            agent.logger.info("\n✅ Browsing task completed successfully!")
            # Store results in agent state if needed
            agent.state["last_browse_result"] = result.get("result")
            return True
        else:
            agent.logger.error(f"\n❌ Browsing task failed: {result.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        agent.logger.error(f"\n❌ Error during browsing task: {str(e)}")
        return False
