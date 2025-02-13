from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from ..tools import tool_registry

router = APIRouter(prefix="/api")

class NaturalLanguageRequest(BaseModel):
    prompt: str

def parse_agent_request(prompt: str) -> tuple[str, List[str]]:
    """Parse natural language prompt to determine needed tools.
    For example: "Create an AI agent that creates memecoins"
    would return ("MemeBot", ["twitter-search", "image-gen", "token-creator"])
    """
    # Basic keyword matching for demo
    tools = []
    if any(word in prompt.lower() for word in ["meme", "coin", "memecoin", "token"]):
        tools.extend(["twitter-search", "image-gen", "token-creator"])
    if "image" in prompt.lower() or "generate" in prompt.lower():
        tools.append("image-gen")
    if "tweet" in prompt.lower() or "twitter" in prompt.lower():
        tools.append("twitter-search")
    
    # Generate a name from the prompt
    name_keywords = ["agent", "bot", "that"]
    for keyword in name_keywords:
        if keyword in prompt.lower():
            name = prompt.lower().split(keyword)[1].strip().split()[0].title() + "Bot"
            break
    else:
        name = "CustomBot"
    
    return name, list(set(tools))

@router.post("/natural-language")
async def create_agent_from_prompt(request: NaturalLanguageRequest):
    """Create an agent from natural language description"""
    try:
        name, tool_ids = parse_agent_request(request.prompt)
        
        # Convert tool IDs to full tool configs with defaults
        tools = []
        for tool_id in tool_ids:
            tool = tool_registry.get_tool(tool_id)
            if tool:
                tool_config = {
                    "id": tool.id,
                    "name": tool.name,
                    "config": {opt.name: opt.default for opt in tool.config_options if opt.default}
                }
                tools.append(tool_config)
        
        # Create agent config
        from .agents import create_agent, AgentCreateRequest
        return await create_agent(AgentCreateRequest(name=name, tools=tools))
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))