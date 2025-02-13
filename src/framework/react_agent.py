"""
LangGraph ReAct Agent Implementation.

This module implements a ReAct agent framework using LangGraph for workflow transitions.
"""
from typing import List, Dict, Any, Optional, Tuple
from langgraph.graph import StateGraph, Graph
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from ..server.tools import Tool, tool_registry

# Define state schema
class AgentState(BaseModel):
    """State management for ReAct agent."""
    messages: List[Union[AIMessage, HumanMessage]]
    steps: List[Dict[str, Any]]
    tool_outputs: Dict[str, Any]
    current_tool: Optional[str] = None
    final_output: Optional[str] = None

# Define ReAct prompts
REACT_SYSTEM_PROMPT = """You are an AI agent that can use tools to complete tasks. Follow this format:

Thought: Analyze what to do
Action: Use a tool in the format: {"tool": "tool_name", "params": {...}}
Observation: Tool output
... (repeat until task is done)
Thought: Task is complete
Final Answer: Final response

Available tools:
{tool_descriptions}

Current objective: {objective}
Previous steps: {previous_steps}
"""

class ReActAgent:
    """Implementation of ReAct agent using LangGraph."""
    
    def __init__(self, tools: List[Tool]):
        """Initialize ReAct agent with available tools."""
        self.tools = tools
        self.tool_descriptions = self._format_tool_descriptions()
        self.graph = self._build_graph()
    
    def _format_tool_descriptions(self) -> str:
        """Format tool descriptions for prompt."""
        descriptions = []
        for tool in self.tools:
            params = [f"- {opt.name}: {opt.description}" 
                     for opt in tool.config_options]
            desc = f"{tool.name} ({tool.id}):\\n{tool.description}\\n"
            if params:
                desc += "Parameters:\\n" + "\\n".join(params)
            descriptions.append(desc)
        return "\\n\\n".join(descriptions)
    
    def _build_graph(self) -> Graph:
        """Build LangGraph workflow."""
        # Create state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("generate_thought", self._generate_thought)
        workflow.add_node("execute_action", self._execute_action)
        workflow.add_node("process_observation", self._process_observation)
        
        # Add edges
        workflow.add_edge("generate_thought", "execute_action")
        workflow.add_edge("execute_action", "process_observation")
        workflow.add_edge("process_observation", "generate_thought")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "process_observation",
            self._should_continue,
            {
                True: "generate_thought",
                False: "end"
            }
        )
        
        # Compile graph
        return workflow.compile()
    
    def _generate_thought(self, state: AgentState) -> AgentState:
        """Generate next thought based on current state."""
        # Format prompt
        prompt = PromptTemplate(
            template=REACT_SYSTEM_PROMPT,
            input_variables=["tool_descriptions", "objective", "previous_steps"]
        )
        
        # Get LLM response
        formatted_steps = "\\n".join(
            [f"Step {i+1}: {step}" for i, step in enumerate(state.steps)]
        )
        response = self.llm(prompt.format(
            tool_descriptions=self.tool_descriptions,
            objective=state.messages[0].content,
            previous_steps=formatted_steps
        ))
        
        # Update state
        state.steps.append(f"Thought: {response}")
        return state
    
    def _execute_action(self, state: AgentState) -> AgentState:
        """Execute tool action based on thought."""
        # Parse action from last thought
        action = self._parse_action(state.steps[-1])
        if not action:
            return state
            
        tool_id = action["tool"]
        tool = tool_registry.get_tool(tool_id)
        if not tool:
            state.steps.append(f"Error: Tool {tool_id} not found")
            return state
            
        # Execute tool
        try:
            result = self._execute_tool(tool, action["params"])
            state.tool_outputs[tool_id] = result
            state.steps.append(f"Action: {action}\\nObservation: {result}")
        except Exception as e:
            state.steps.append(f"Error executing {tool_id}: {str(e)}")
            
        return state
    
    def _process_observation(self, state: AgentState) -> AgentState:
        """Process tool execution observation."""
        # Check for final answer
        last_step = state.steps[-1]
        if "Final Answer:" in last_step:
            state.final_output = last_step.split("Final Answer:")[-1].strip()
        return state
    
    def _should_continue(self, state: AgentState) -> bool:
        """Determine if agent should continue or stop."""
        return state.final_output is None
    
    def run(self, objective: str) -> str:
        """Run ReAct agent workflow."""
        # Initialize state
        state = AgentState(
            messages=[HumanMessage(content=objective)],
            steps=[],
            tool_outputs={}
        )
        
        # Execute graph
        final_state = self.graph.run(state)
        return final_state.final_output or "Failed to complete task"

# Utility functions
def _parse_action(thought: str) -> Optional[Dict[str, Any]]:
    """Parse action from thought text."""
    try:
        if "Action:" not in thought:
            return None
        action_text = thought.split("Action:")[-1].split("Observation:")[0]
        return json.loads(action_text)
    except:
        return None