import logging
import requests
import os
from typing import Dict, Any, List, Optional
from .base_connection import BaseConnection, Action, ActionParameter

logger = logging.getLogger("connections.tx_connection")

class TxConnectionError(Exception):
    """Base exception for transaction connection errors"""
    pass

class TxConnection(BaseConnection):
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base_url = config.get("api_base_url", "https://api.sonicscan.org/api")
        self.api_key = config.get("api_key", os.environ.get("SONICSCAN_API_KEY", ""))
        self._initialize()

    def _initialize(self):
        """Initialize Transaction connection"""
        if not self.api_key:
            logger.warning("No API key provided for SonicScan API. Some requests may be rate-limited.")
        logger.info(f"Initialized Transaction connection with API URL: {self.api_base_url}")

    def is_llm_provider(self) -> bool:
        return False

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the configuration parameters"""
        # Add API URL if not provided
        if "api_base_url" not in config:
            config["api_base_url"] = "https://api.sonicscan.org/api"
            
        # Check for API key in config or environment
        if "api_key" not in config:
            config["api_key"] = os.environ.get("SONICSCAN_API_KEY", "")
            
        return config

    def register_actions(self) -> None:
        # Get transaction list action
        self.actions['get_tx_list'] = Action(
            name='get_tx_list',
            description='Get a list of normal transactions for an address',
            parameters=[
                ActionParameter(name='address', type=str, required=True, description='Address to get transactions for'),
                ActionParameter(name='startblock', type=int, required=False, description='Starting block number'),
                ActionParameter(name='endblock', type=int, required=False, description='Ending block number'),
                ActionParameter(name='page', type=int, required=False, description='Page number'),
                ActionParameter(name='offset', type=int, required=False, description='Max records to return'),
                ActionParameter(name='sort', type=str, required=False, description='Sort order (asc/desc)'),
            ]
        )

    def configure(self) -> bool:
        """Configure the Transaction connection"""
        try:
            self._initialize()
            return True
        except Exception as e:
            logger.error(f"Failed to configure Transaction connection: {str(e)}")
            return False

    def is_configured(self, verbose: bool = False) -> bool:
        """Check if the connection is properly configured"""
        if not self.api_base_url:
            if verbose:
                logger.error("API base URL is not configured")
            return False
        return True

    def get_tx_list(self, address: str, startblock: int = 0, endblock: int = 99999999, 
                  page: int = 1, offset: int = 10, sort: str = "asc") -> Dict:
        """
        Get a list of normal transactions by address
        """
        try:
            params = {
                "module": "account",
                "action": "txlist",
                "address": address,
                "startblock": startblock,
                "endblock": endblock,
                "page": page,
                "offset": offset,
                "sort": sort,
            }
            
            if self.api_key:
                params["apikey"] = self.api_key
                
            response = requests.get(self.api_base_url, params=params)
            response.raise_for_status()
            
            result = response.json()
            if result.get("status") == "0" and result.get("message") == "No transactions found":
                return {
                    "status": "success", 
                    "message": "No transactions found", 
                    "result": []
                }
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise TxConnectionError(f"API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to get transaction list: {str(e)}")
            raise TxConnectionError(f"Failed to get transaction list: {str(e)}")

    def perform_action(self, action_name: str, kwargs) -> Any:
        """Execute a transaction action with validation"""
        if action_name not in self.actions:
            raise KeyError(f"Unknown action: {action_name}")
        logger.info(kwargs)
        if not self.is_configured(verbose=True):
            raise TxConnectionError("Transaction service is not properly configured")

        action = self.actions[action_name]
        validation_errors = action.validate_params(kwargs)
        if validation_errors:
            raise TxConnectionError(f"Invalid parameters: {', '.join(validation_errors)}")

        method = getattr(self, action_name)
        return method(**kwargs)
