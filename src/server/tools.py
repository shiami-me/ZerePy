from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel

class ToolType(str, Enum):
    SOCIAL = "social"
    CONTENT = "content"
    BLOCKCHAIN = "blockchain"
    RESEARCH = "research"

class ToolConfigOption(BaseModel):
    name: str
    type: str  # "string" | "number" | "boolean"
    description: str
    required: bool = True
    default: Optional[str] = None

class Tool(BaseModel):
    id: str
    name: str
    type: ToolType
    description: str
    config_options: List[ToolConfigOption]
    module: str  # Python module path
    class_name: str  # Class name within module

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {
            "twitter-search": Tool(
                id="twitter-search",
                name="Twitter Search",
                type=ToolType.SOCIAL,
                description="Search and analyze Twitter content",
                config_options=[
                    ToolConfigOption(
                        name="searchTerms",
                        type="string",
                        description="Search terms to monitor"
                    ),
                    ToolConfigOption(
                        name="resultCount",
                        type="number",
                        description="Number of results to return",
                        default="10"
                    )
                ],
                module="ZerePy.src.actions.twitter_actions",
                class_name="TwitterActions"
            ),
            "image-gen": Tool(
                id="image-gen",
                name="Image Generator",
                type=ToolType.CONTENT,
                description="Generate images using AI",
                config_options=[
                    ToolConfigOption(
                        name="model",
                        type="string",
                        description="AI model to use",
                        default="together-ai"
                    ),
                    ToolConfigOption(
                        name="style",
                        type="string",
                        description="Image style",
                        default="meme"
                    )
                ],
                module="ZerePy.src.actions.together_ai_actions",
                class_name="TogetherAIActions"
            ),
            "token-creator": Tool(
                id="token-creator",
                name="Token Creator",
                type=ToolType.BLOCKCHAIN,
                description="Create and deploy ERC20 tokens",
                config_options=[
                    ToolConfigOption(
                        name="name",
                        type="string",
                        description="Token name"
                    ),
                    ToolConfigOption(
                        name="symbol",
                        type="string", 
                        description="Token symbol"
                    ),
                    ToolConfigOption(
                        name="supply",
                        type="string",
                        description="Initial supply"
                    )
                ],
                module="ZerePy.src.actions.ethereum_actions",
                class_name="EthereumActions"
            )
        }

    def list_tools(self) -> List[Tool]:
        return list(self.tools.values())

    def get_tool(self, tool_id: str) -> Optional[Tool]:
        return self.tools.get(tool_id)

tool_registry = ToolRegistry()