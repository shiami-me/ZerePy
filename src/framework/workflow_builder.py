"""
Workflow Builder for Natural Language Agent Creation.

This module handles natural language understanding for automated workflow creation.
"""
from typing import List, Dict, Any, Optional
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

from ..server.tools import Tool, tool_registry
from .react_agent import ReActAgent

# Prompt for task understanding
TASK_ANALYSIS_PROMPT = """Analyze the following task and identify required tools and workflow:

Task: {task}

Available tools:
{tool_descriptions}

Provide output in the following format:
{
    "tools": ["tool1_id", "tool2_id"],
    "workflow": {
        "steps": [
            {"tool": "tool1_id", "purpose": "why this tool"},
            {"tool": "tool2_id", "purpose": "why this tool"}
        ],
        "transitions": [
            ["tool1_id", "tool2_id", "transition condition"]
        ]
    }
}

Your analysis:"""

class WorkflowBuilder:
    """Builds agent workflows from natural language descriptions."""
    
    def __init__(self):
        """Initialize workflow builder."""
        self.tools = tool_registry.list_tools()
        self.tool_descriptions = self._format_tool_descriptions()
    
    def _format_tool_descriptions(self) -> str:
        """Format available tools for prompts."""
        descriptions = []
        for tool in self.tools:
            desc = f"{tool.name} ({tool.id}):\\n"
            desc += f"Type: {tool.type}\\n"
            desc += f"Description: {tool.description}"
            descriptions.append(desc)
        return "\\n\\n".join(descriptions)
    
    def analyze_task(self, task: str) -> Dict[str, Any]:
        """Analyze task to determine required tools and workflow."""
        # Get LLM response
        prompt = PromptTemplate(
            template=TASK_ANALYSIS_PROMPT,
            input_variables=["task", "tool_descriptions"]
        )
        response = self.llm(prompt.format(
            task=task,
            tool_descriptions=self.tool_descriptions
        ))
        
        # Parse response
        try:
            analysis = json.loads(response)
            return self._validate_analysis(analysis)
        except:
            return {
                "error": "Failed to analyze task",
                "raw_response": response
            }
    
    def _validate_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean up task analysis."""
        if not isinstance(analysis, dict):
            return {"error": "Invalid analysis format"}
            
        # Validate tools exist
        tools = []
        for tool_id in analysis.get("tools", []):
            tool = tool_registry.get_tool(tool_id)
            if tool:
                tools.append(tool)
            else:
                return {"error": f"Invalid tool: {tool_id}"}
                
        # Validate workflow steps
        workflow = analysis.get("workflow", {})
        steps = workflow.get("steps", [])
        for step in steps:
            if not tool_registry.get_tool(step.get("tool")):
                return {"error": f"Invalid tool in workflow: {step.get('tool')}"}
                
        return {
            "tools": tools,
            "workflow": workflow
        }
    
    def create_agent(self, task: str) -> Optional[ReActAgent]:
        """Create ReAct agent from task description."""
        # Analyze task
        analysis = self.analyze_task(task)
        if "error" in analysis:
            return None
            
        # Create agent with selected tools
        return ReActAgent(analysis["tools"])

    def create_agent_with_tools(self, tools: List[Tool]) -> ReActAgent:
        """Create ReAct agent with specific tools."""
        return ReActAgent(tools)