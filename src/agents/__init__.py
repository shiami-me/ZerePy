from typing import Dict, Type, Any
import os

from .price_prediction_agent import PricePredictionAgent

# Dictionary to map agent types to their respective classes
AGENT_REGISTRY: Dict[str, Type[Any]] = {
    "price_prediction": PricePredictionAgent
}

def get_agent_config_path(agent_type: str) -> str:
    """
    Retrieve the configuration path for a specific agent type.
    
    Args:
        agent_type (str): Type of agent to get config for.
    
    Returns:
        str: Path to the agent's configuration file.
    """
    base_config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'agents')
    config_map = {
        "price_prediction": "price_prediction.json"
    }
    
    config_filename = config_map.get(agent_type)
    if not config_filename:
        raise ValueError(f"No configuration found for agent type: {agent_type}")
    
    return os.path.join(base_config_dir, config_filename)

def create_agent(agent_type: str, **kwargs) -> Any:
    """
    Factory method to create an agent based on its type.
    
    Args:
        agent_type (str): Type of agent to create.
        **kwargs: Additional arguments for agent initialization.
    
    Returns:
        Agent instance
    """
    agent_class = AGENT_REGISTRY.get(agent_type)
    if not agent_class:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    config_path = get_agent_config_path(agent_type)
    return agent_class(config_path, **kwargs)

__all__ = [
    'PricePredictionAgent',
    'AGENT_REGISTRY',
    'get_agent_config_path',
    'create_agent'
]