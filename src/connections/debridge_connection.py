import logging
import requests
from typing import Dict, Any
import os
from dotenv import load_dotenv
from .base_connection import BaseConnection, Action, ActionParameter

logger = logging.getLogger("connections.debridge_connection")

class DebridgeConnectionError(Exception):
    """Base exception for Debridge connection errors"""
    pass

class DebridgeConnection(BaseConnection):
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_url = "https://dln.debridge.finance/v1.0"
        self._initialize()

    def _initialize(self):
        """Initialize Debridge connection"""
        try:
            load_dotenv()
            self.access_key = os.getenv("DEBRIDGE_ACCESS_KEY")
            if not self.access_key:
                raise DebridgeConnectionError("DEBRIDGE_ACCESS_KEY not found in environment variables")
        except Exception as e:
            raise DebridgeConnectionError(f"Failed to initialize Debridge connection: {str(e)}")

    def is_llm_provider(self) -> bool:
        return False

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the configuration parameters"""
        return config

    def register_actions(self) -> None:
        self.actions['bridge'] = Action(
            name='bridge',
            description='Bridge assets using Debridge',
            parameters=[
                ActionParameter(name='srcChainTokenIn', type=str, required=True, description='Source chain token address'),
                ActionParameter(name='srcChainTokenInAmount', type=str, required=True, description='Amount of source chain token'),
                ActionParameter(name='dstChainId', type=int, required=True, description='Destination chain ID'),
                ActionParameter(name='dstChainTokenOut', type=str, required=True, description='Destination chain token address'),
                ActionParameter(name='dstChainTokenOutAmount', type=str, required=True, description='Amount of destination chain token'),
                ActionParameter(name='dstChainTokenOutRecipient', type=str, required=True, description='Recipient address on destination chain'),
            ]
        )
        self.actions['get_token_address'] = Action(
            name='get_token_address',
            description='Get token address on Debridge',
            parameters=[
                ActionParameter(name='chainId', type=int, required=True, description='Chain ID'),
                ActionParameter(name='tokenSymbol', type=str, required=True, description='Token symbol'),
            ]
        )

    def configure(self) -> bool:
        """Configure the Debridge connection"""
        try:
            self._initialize()
            return True
        except Exception as e:
            logger.error(f"Failed to configure Debridge connection: {str(e)}")
            return False

    def is_configured(self, verbose: bool = False) -> bool:
        """Check if the connection is properly configured"""
        try:
            # Add any additional checks if needed
            return True
        except Exception as e:
            if verbose:
                logger.error(f"Configuration check failed: {str(e)}")
            return False

    def bridge(self, srcChainTokenIn: str, srcChainTokenInAmount: str, 
               dstChainId: int, dstChainTokenOut: str, dstChainTokenOutAmount: str, 
               dstChainTokenOutRecipient: str) -> Dict:
        """
        Bridge assets using Debridge
        
        Args:
            srcChainId: Source chain ID
            srcChainTokenIn: Source chain token address
            srcChainTokenInAmount: Amount of source chain token
            dstChainId: Destination chain ID
            dstChainTokenOut: Destination chain token address
            dstChainTokenOutAmount: Amount of destination chain token
            dstChainTokenOutRecipient: Recipient address on destination chain
            srcChainOrderAuthorityAddress: Order authority address on source chain
            dstChainOrderAuthorityAddress: Order authority address on destination chain
        """
        try:
            params = {
                "srcChainId": 100000014,
                "srcChainTokenIn": srcChainTokenIn,
                "srcChainTokenInAmount": srcChainTokenInAmount,
                "dstChainId": dstChainId,
                "dstChainTokenOut": dstChainTokenOut,
                "dstChainTokenOutAmount": dstChainTokenOutAmount,
                "prependOperatingExpense": True,
                "dstChainTokenOutRecipient": dstChainTokenOutRecipient,
                "srcChainOrderAuthorityAddress": dstChainTokenOutRecipient,
                "dstChainOrderAuthorityAddress": dstChainTokenOutRecipient,
                "accesstoken": self.access_key
            }
            
            response = requests.get(f"{self.api_url}/dln/order/create-tx", params=params)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            raise DebridgeConnectionError(f"Failed to bridge assets: {str(e)}")
        
    def get_token_address(self, chainId: int, token: str) -> str:
        """Get the token address on a specific chain by partial match of name or symbol"""
        try:
            params = {"chainId": chainId}
            response = requests.get(f"{self.api_url}/token-list", params=params)
            response.raise_for_status()
            tokens = response.json().get("tokens", {})

            token = token.lower()
            for address, details in tokens.items():
                if token in details["symbol"].lower() or token in details["name"].lower():
                    return address

            return None

        except Exception as e:
            raise DebridgeConnectionError(f"Failed to get token address: {str(e)}")

    def perform_action(self, action_name: str, kwargs) -> Any:
        """Execute a Debridge action with validation"""
        if action_name not in self.actions:
            raise KeyError(f"Unknown action: {action_name}")

        load_dotenv()
        
        if not self.is_configured(verbose=True):
            raise DebridgeConnectionError("Debridge is not properly configured")

        action = self.actions[action_name]
        validation_errors = action.validate_params(kwargs)
        if validation_errors:
            raise DebridgeConnectionError(f"Invalid parameters: {', '.join(validation_errors)}")

        method_name = action_name.replace('-', '_')
        method = getattr(self, method_name)
        return method(**kwargs)
