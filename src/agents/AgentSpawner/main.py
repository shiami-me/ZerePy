from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Tuple, Annotated
import os
import json
import re
import logging
import traceback
import sys

from textwrap import dedent
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format=' %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AIAgentGenerator")

load_dotenv()

# Configuration setup
class AgentConfig:
    def __init__(self, api_key=None):
        # Use provided key or get from environment
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Please provide it or set GOOGLE_API_KEY environment variable.")
        
        # Default model settings
        self.model_name = "gemini-2.0-flash"
        self.temperature = 0.7
        self.max_output_tokens = 2048

# State definitions for our agent system
class AgentState(BaseModel):
    user_prompt: str = Field(description="The original user prompt")
    agent_specs: List[Dict] = Field(default_factory=list, description="Specifications for agents that have been defined")
    workflow_spec: Optional[Dict] = Field(default=None, description="Specification for the workflow")
    current_output: str = Field(default="", description="Current output to return to the user")
    error: Optional[str] = Field(default=None, description="Error message if any")

    # Make sure to use model_copy instead of directly modifying state
    def update_output(self, output: str) -> 'AgentState':
        """Helper method to update the current output"""
        logger.debug(f"Updating output to new content of length {len(output)}")
        return self.model_copy(update={"current_output": output})

    def update_error(self, error: str) -> 'AgentState':
        """Helper method to update the error"""
        logger.debug(f"Updating error: {error}")
        return self.model_copy(update={"error": error})


# Debug helper function to inspect state objects
def debug_state(state, location=""):
    """Print debug info about state object"""
    logger.debug(f"STATE INSPECTION at {location}")
    logger.debug(f"Type: {type(state)}")
    logger.debug(f"Dir: {dir(state)}")
    if hasattr(state, 'model_dump'):
        logger.debug(f"Model dump: {state.model_dump()}")
    return state


# Main Agent Generator class
class AIAgentGenerator:
    def __init__(self, api_key=None):
        # Initialize configuration
        self.config = AgentConfig(api_key)
        logger.info("Initializing AIAgentGenerator")
        
        # Initialize LLM
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.model_name,
                google_api_key=self.config.api_key,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens
            )
            logger.info(f"Initialized LLM model: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _is_complete_agent(self, code: str) -> bool:
        """Check if the generated code contains minimum expected structure."""
        # For example, require presence of a main interface and at least one class definition.
        return ("if __name__" in code) and ("class " in code) and ("def " in code)

    def _build_workflow(self) -> StateGraph:
        """Build the main agent generation workflow"""
        logger.info("Building workflow graph")
        # Create the workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes to the graph
        workflow.add_node("analyze_prompt", self._wrap_with_debug(self.analyze_prompt, "analyze_prompt"))
        workflow.add_node("design_agents", self._wrap_with_debug(self.design_agents, "design_agents"))
        workflow.add_node("create_workflow", self._wrap_with_debug(self.create_workflow, "create_workflow"))
        workflow.add_node("generate_code", self._wrap_with_debug(self.generate_code, "generate_code"))
        
        # Add edges to define the flow
        workflow.add_edge("analyze_prompt", "design_agents")
        workflow.add_edge("design_agents", "create_workflow")
        workflow.add_edge("create_workflow", "generate_code")
        workflow.add_edge("generate_code", END)
        
        # Set the entry point
        workflow.set_entry_point("analyze_prompt")
        
        # Compile the workflow
        logger.info("Workflow graph compiled")
        return workflow.compile()
    
    def _wrap_with_debug(self, func, name):
        """Wrap node functions with debugging"""
        def wrapped(state):
            logger.debug(f"ENTERING NODE: {name}")
            logger.debug(f"Input state type: {type(state)}")
            
            if hasattr(state, "model_dump"):
                logger.debug(f"Input state fields: {list(state.model_dump().keys())}")
            else:
                logger.debug(f"Input state dir: {dir(state)}")
                logger.debug(f"Input state dict: {state.__dict__ if hasattr(state, '__dict__') else 'No __dict__'}")
            
            try:
                logger.debug(f"Calling function: {name}")
                result = func(state)
                logger.debug(f"Function {name} completed")
                
                logger.debug(f"Output state type: {type(result)}")
                if hasattr(result, "model_dump"):
                    logger.debug(f"Output state fields: {list(result.model_dump().keys())}")
                else:
                    logger.debug(f"Output state dir: {dir(result)}")
                
                return result
            except Exception as e:
                logger.error(f"ERROR in {name}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Return state with error if possible
                if hasattr(state, "update_error"):
                    return state.update_error(f"Error in {name}: {str(e)}")
                else:
                    # If we can't update state properly, reraise
                    raise
        
        return wrapped
    
    def analyze_prompt(self, state: AgentState) -> AgentState:
        """Analyze the user prompt to understand the requirements"""
        logger.info("Starting analyze_prompt node")
        
        # Verify state has expected attributes
        logger.debug(f"State type: {type(state)}")
        if not isinstance(state, AgentState):
            logger.error(f"Expected AgentState, got {type(state)}")
            # Try to convert if possible
            if hasattr(state, "user_prompt"):
                logger.debug("Attempting to convert state to AgentState")
                try:
                    state_dict = {k: getattr(state, k) for k in AgentState.model_fields.keys() if hasattr(state, k)}
                    state = AgentState(**state_dict)
                except Exception as e:
                    logger.error(f"Failed to convert state: {str(e)}")
        
        prompt = dedent(f"""
        You are a system that analyzes user requirements to build AI agent workflows.
        
        TASK: Analyze the following user prompt and extract key requirements for building an AI agent system.
        
        USER PROMPT: "{state.user_prompt}"
        
        Please identify:
        1. The main goal or purpose of the requested agent system
        2. Key functionalities that would be needed
        3. Types of agents that might be required
        4. Any specific technologies or constraints mentioned
        5. The expected inputs and outputs of the system
        
        Format your response as a structured analysis that will be used in the next steps of building the agent system.
        """)
        
        try:
            logger.debug("Sending prompt to LLM")
            response = self.llm.invoke(prompt)
            logger.debug(f"LLM response type: {type(response)}")
            logger.debug(f"LLM response dir: {dir(response)}")
            
            # Check what attributes the response has
            if hasattr(response, "content"):
                logger.debug("Response has 'content' attribute")
                content = response.content
            elif hasattr(response, "text"):
                logger.debug("Response has 'text' attribute")
                content = response.text
            else:
                logger.debug(f"Response doesn't have common attributes, converting to string")
                content = str(response)
            
            logger.debug(f"Extracted content length: {len(content)}")
            
            # Create new state with updated output
            new_state = state.update_output(content)
            logger.debug(f"New state created with updated output, type: {type(new_state)}")
            return new_state
        except Exception as e:
            logger.error(f"Error in analyze_prompt: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return state.update_error(f"Error Analyzing prompt: {str(e)}")
    
    def design_agents(self, state: AgentState) -> AgentState:
        """Design the individual agents based on the analysis"""
        logger.info("Starting design_agents node")
        logger.debug(f"Input state has current_output: {hasattr(state, 'current_output')}")
        
        # Verify state has expected attributes
        if not hasattr(state, "current_output"):
            logger.error("State missing 'current_output' attribute")
            # Try to access as a dict
            if hasattr(state, "get"):
                logger.debug("Attempting to access state as dict")
                current_output = state.get("current_output", "")
            else:
                logger.error("Cannot access state attributes")
                return AgentState(user_prompt=getattr(state, "user_prompt", "")).update_error(
                    "Error: State object structure is invalid"
                )
        else:
            current_output = state.current_output
        
        prompt = dedent(f"""
        You are a system that designs specialized AI agents based on requirements analysis.
        
        PREVIOUS ANALYSIS: 
        {current_output}
        
        TASK: Design a set of specialized agents to fulfill the requirements. For each agent, define:
        1. Agent name and purpose
        2. Input and output specifications
        3. Primary functionality
        4. Required tools or APIs
        5. How it will integrate with other agents
        
        Format your response as a structured list of agent specifications in JSON-compatible format.
        Each agent should have a clear, focused responsibility within the overall system.
        """)
        
        try:
            logger.debug("Sending agent design prompt to LLM")
            response = self.llm.invoke(prompt)
            
            # Extract content from response
            if hasattr(response, "content"):
                response_content = response.content
            else:
                response_content = str(response)
            
            logger.debug(f"Received design response of length: {len(response_content)}")
            
            # Create a new state with the updated output
            updated_state = state.update_output(response_content)
            logger.debug(f"Updated state with design response, type: {type(updated_state)}")
            
            # Have the LLM extract actual JSON data for agent_specs
            extraction_prompt = dedent(f"""
            Based on the agent designs described below, extract a JSON list of agent specifications.
            Each agent should be a JSON object with fields for name, purpose, inputs, outputs, functionality, and tools.
            
            Agent designs:
            {updated_state.current_output}
            
            Return ONLY valid JSON with no explanation or additional text.
            """)
            
            logger.debug("Sending extraction prompt to LLM")
            json_response = self.llm.invoke(extraction_prompt)
            
            # Extract content
            if hasattr(json_response, "content"):
                json_text = json_response.content
            else:
                json_text = str(json_response)
                
            logger.debug(f"Received JSON extraction response of length: {len(json_text)}")
            
            # Try parsing the direct JSON response
            try:
                logger.debug("Attempting to parse JSON directly")
                specs = json.loads(json_text)
            except json.JSONDecodeError:
                # If direct parsing fails, extract JSON from a code block
                logger.debug("Direct JSON parsing failed, trying code block extraction")
                json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
                json_match = re.search(json_pattern, json_text, re.DOTALL)
                if json_match:
                    logger.debug("Found JSON in code block")
                    json_str = json_match.group(1).strip()
                    specs = json.loads(json_str)
                else:
                    logger.error("Could not extract valid JSON from LLM response")
                    raise ValueError("Could not extract valid JSON from LLM response")
            
            logger.debug(f"Successfully parsed JSON with {len(specs)} agent specifications")
            
            # Create a new copy with all updates at once
            result_state = updated_state.model_copy(update={
                "agent_specs": specs,
                "error": None
            })
            
            logger.debug(f"Returning result state of type: {type(result_state)}")
            return result_state
        except Exception as e:
            logger.error(f"Error in design_agents: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return state.update_error(f"Error designing agents: {str(e)}")
    
    def create_workflow(self, state: AgentState) -> AgentState:
        """Create a workflow connecting the designed agents"""
        logger.info("Starting create_workflow node")
        logger.debug(f"Input state has agent_specs: {hasattr(state, 'agent_specs')}")
        logger.debug(f"Input state has user_prompt: {hasattr(state, 'user_prompt')}")
        
        # Safely access agent_specs
        agent_specs = getattr(state, "agent_specs", [])
        user_prompt = getattr(state, "user_prompt", "")
        
        # Create a safe JSON string for the prompt
        try:
            agent_specs_json = json.dumps(agent_specs, indent=2)
            logger.debug(f"Serialized agent_specs to JSON of length: {len(agent_specs_json)}")
        except Exception as e:
            logger.error(f"Error serializing agent_specs: {str(e)}")
            agent_specs_json = "[]"
        
        prompt = dedent(f"""
        You are a system that designs workflows connecting AI agents together.
        
        AGENT SPECIFICATIONS:
        {agent_specs_json}
        
        TASK: Design a workflow that connects these agents to achieve the original goal:
        "{user_prompt}"
        
        For the workflow specification, include:
        1. The entry point agent
        2. The connections between agents (which agent outputs go to which agent inputs)
        3. Conditional logic for routing between agents if needed
        4. The final output format and which agent produces it
        
        Return ONLY a JSON object containing:
        - "entry_point": The starting agent name
        - "connections": Array of objects describing agent connections
        - "routing_logic": Object containing conditional routing rules
        - "final_output": Object describing the final output format
        
        Format as valid JSON with no additional text or explanation.
        """)
        
        try:
            logger.debug("Sending workflow prompt to LLM")
            response = self.llm.invoke(prompt)
            
            # Extract content
            if hasattr(response, "content"):
                workflow_text = response.content
            else:
                workflow_text = str(response)
                
            logger.debug(f"Received workflow response of length: {len(workflow_text)}")
            
            # Try to parse the JSON response
            try:
                # First try direct JSON parsing
                logger.debug("Attempting direct JSON parsing of workflow response")
                workflow_dict = json.loads(workflow_text)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from code block
                logger.debug("Direct parsing failed, trying code block extraction")
                json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
                json_match = re.search(json_pattern, workflow_text, re.DOTALL)
                if json_match:
                    logger.debug("Found JSON in code block")
                    json_str = json_match.group(1).strip()
                    workflow_dict = json.loads(json_str)
                else:
                    logger.warning("Could not extract valid JSON, using fallback workflow")
                    workflow_dict = {
                        "entry_point": agent_specs[0]["name"] if agent_specs else "main_agent",
                        "connections": [],
                        "routing_logic": {},
                        "final_output": {"format": "text", "producer": "last_agent"}
                    }
            
            logger.debug(f"Parsed workflow dict with keys: {workflow_dict.keys()}")
            
            # Create new state with both updates
            result_state = state.model_copy(update={
                "workflow_spec": workflow_dict, 
                "current_output": workflow_text
            })
            
            logger.debug(f"Created new state with workflow spec, type: {type(result_state)}")
            return result_state
        except Exception as e:
            logger.error(f"Error in create_workflow: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return state.update_error(f"Error creating workflow: {str(e)}")
    
    def generate_code(self, state: AgentState) -> AgentState:
        """Generate the final Python code for the agent system"""
        logger.info("Starting generate_code node")
        logger.debug(f"Input state has workflow_spec: {hasattr(state, 'workflow_spec')}")
        logger.debug(f"Input state has agent_specs: {hasattr(state, 'agent_specs')}")
        
        # Safely access required fields
        user_prompt = getattr(state, "user_prompt", "")
        
        # Create safe JSON for prompts
        try:
            agent_specs_json = json.dumps(getattr(state, "agent_specs", []), indent=2)
            workflow_spec_json = json.dumps(getattr(state, "workflow_spec", {}), indent=2)
        except Exception as e:
            logger.error(f"Error serializing specs: {str(e)}")
            agent_specs_json = "[]"
            workflow_spec_json = "{}"
        
        prompt = dedent(f"""
        You are a system that generates Python code for AI agent systems using LangGraph and LangChain.
        
        USER REQUIREMENTS: "{user_prompt}"
        
        AGENT SPECIFICATIONS:
        {agent_specs_json}
        
        WORKFLOW SPECIFICATION:
        {workflow_spec_json}
        
        TASK: Generate complete, executable Python code that implements this agent system using LangGraph and LangChain with Google's Generative AI. The code should:
        1. Define all necessary classes and functions
        2. Implement each agent as specified
        3. Build the workflow graph connecting the agents
        4. Include a simple interface to run the system
        5. Handle errors appropriately
        6. Include clear comments explaining the implementation
        
        The code should be complete and ready to run with minimal additional setup (aside from API keys).
        """)
        
        try:
            logger.debug("Sending code generation prompt to LLM")
            response = self.llm.invoke(prompt)
            
            # Extract content
            if hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)
                
            logger.debug(f"Received code generation response of length: {len(content)}")
            
            # Create new state with updated output
            result_state = state.update_output(content)
            logger.debug(f"Created new state with generated code, type: {type(result_state)}")
            return result_state
        except Exception as e:
            logger.error(f"Error in generate_code: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return state.update_error(f"Error generating code: {str(e)}")
    
    def run(self, user_prompt: str) -> Tuple[str, Optional[str]]:
        """Run the agent generator with a user prompt"""
        logger.info(f"Starting agent generator run with prompt: {user_prompt[:50]}...")
        try:
            # Initialize the state with the user prompt
            initial_state = AgentState(user_prompt=user_prompt)
            logger.debug(f"Created initial state: {type(initial_state)}")
            
            # Execute the workflow
            logger.info("Invoking workflow")
            final_state = self.workflow.invoke(initial_state)
            logger.debug(f"Workflow complete, final state type: {type(final_state)}")
            
            # If final_state is not an instance of AgentState, try converting it
            if not isinstance(final_state, AgentState):
                try:
                    final_state = AgentState(
                        user_prompt=final_state.get("user_prompt", ""),
                        agent_specs=final_state.get("agent_specs", []),
                        workflow_spec=final_state.get("workflow_spec", None),
                        current_output=final_state.get("current_output", ""),
                        error=final_state.get("error", None)
                    )
                    logger.debug("Converted final state to AgentState")
                except Exception as e:
                    logger.error(f"Conversion to AgentState failed: {str(e)}")
                    return "", "System error: Invalid state returned from workflow"
            
            # Check if final_state has expected attributes
            if not hasattr(final_state, "current_output") or not hasattr(final_state, "error"):
                logger.error(f"Final state missing expected attributes: {dir(final_state)}")
                return "", "System error: Invalid state returned from workflow"
            
            # Validate that the generated code is complete
            if not self._is_complete_agent(final_state.current_output):
                logger.error("Generated code is incomplete.")
                return "", "Incomplete agent code generated. Please re-run the agent generator."
            
            logger.info("Run completed successfully")
            return final_state.current_output, final_state.error
        except Exception as e:
            logger.error(f"Error in run method: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return "", f"System error: {str(e)}"

# Main interface
def create_agent_system(prompt: str, api_key: str = None) -> str:
    """
    Create a custom agent system based on a user prompt.
    
    Args:
        prompt: User's description of the desired agent system
        api_key: Google API key (optional if set in environment)
        
    Returns:
        Generated code for the custom agent system
    """
    logger.info("Starting create_agent_system")
    try:
        generator = AIAgentGenerator(api_key)
        result, error = generator.run(prompt)
        
        if error:
            logger.error(f"Error occurred: {error}")
            return f"Error: {error}"
        else:
            logger.info("Successfully generated agent system")
            return result
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return f"Critical Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Example prompt
    logger.info("Starting example run")
    example_prompt = """
    Build me an AI agent that can reccomend me recipies based on the ingredients I have in my kitchen.
    """
    try:
        generated_system = create_agent_system(example_prompt)
        # Save the generated code to a file
        output_file = "generated_agent_system.py"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(generated_system)
        logger.info(f"Generated system saved to {output_file}")
    except Exception as e:
        logger.critical(f"Example run failed: {str(e)}")
        logger.critical(f"Traceback: {traceback.format_exc()}")