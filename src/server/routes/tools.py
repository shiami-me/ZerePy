from fastapi import APIRouter
from ..tools import tool_registry

router = APIRouter(prefix="/api")

@router.get("/tools")
async def list_tools():
    """List all available tools/actions"""
    return {"tools": tool_registry.list_tools()}