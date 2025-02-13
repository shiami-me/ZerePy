import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

class AgentCreateRequest(BaseModel):
    name: str
    tools: list

router = APIRouter(prefix="/api")

@router.post("/agents")
async def create_agent(request: AgentCreateRequest):
    """Create a new agent from tools configuration"""
    try:
        agent_config = {
            "name": request.name,
            "bio": [f"You are {request.name}, an AI agent created with specific tools."],
            "traits": ["Helpful", "Efficient"],
            "config": [],
            "tasks": []
        }

        # Add tool configurations
        for tool in request.tools:
            tool_config = {
                "name": tool["id"],
                **(tool.get("config") or {})
            }
            agent_config["config"].append(tool_config)

            # Add related tasks
            if tool["id"] == "twitter-search":
                agent_config["tasks"].extend([
                    {"name": "search-twitter", "weight": 1},
                    {"name": "analyze-tweets", "weight": 1}
                ])
            elif tool["id"] == "image-gen":
                agent_config["tasks"].append(
                    {"name": "generate-image", "weight": 1}
                )
            elif tool["id"] == "token-creator":
                agent_config["tasks"].append(
                    {"name": "create-token", "weight": 1}
                )

        # Save agent configuration
        agents_dir = Path("agents")
        agents_dir.mkdir(exist_ok=True)
        
        agent_path = agents_dir / f"{request.name.lower()}.json"
        if agent_path.suffix == '.json':
            with open(agent_path, "w") as f:
                json.dump(agent_config, f, indent=2)
        else:
            raise Exception("Invalid file extension. Only .json files are allowed.")

        return {
            "status": "success",
            "message": f"Agent {request.name} created successfully",
            "agent": agent_config
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))