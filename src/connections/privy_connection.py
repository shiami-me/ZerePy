import logging
import os
import base64
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import requests
from .base_connection import BaseConnection, Action, ActionParameter
from src.helpers.privy_helper import get_authorization_signature
from src.helpers.auth_helper import verify_signature, get_wallet_from_private_key

logger = logging.getLogger("connections.privy_connection")

class PrivyConnectionError(Exception):
    """Base exception for Privy connection errors"""
    pass

class PrivyConnection(BaseConnection):
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._initialize()
        self.register_actions()

    def _initialize(self):
        """Initialize Privy connection"""
        try:
            load_dotenv()
            self.app_id = os.getenv("PRIVY_APP_ID")
            self.app_secret = os.getenv("PRIVY_APP_SECRET")
            self.base_url = "https://auth.privy.io/api/v1"
            self.rpc_url = f"{self.base_url}/wallets/rpc"
            
            # Get wallet address from private key
            self.wallet_address, _ = get_wallet_from_private_key()
            
            if not self.app_id or not self.app_secret:
                raise PrivyConnectionError("PRIVY_APP_ID or PRIVY_APP_SECRET not found in environment variables")
            
            if not self.wallet_address:
                raise PrivyConnectionError("ETH_PRIVATE_KEY not found or invalid in environment variables")
                
        except Exception as e:
            raise PrivyConnectionError(f"Failed to initialize Privy connection: {str(e)}")

    def is_llm_provider(self) -> bool:
        return False

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the configuration parameters"""
        return config

    def register_actions(self) -> None:
        """Register transaction signing and sending actions"""
        self.actions['sign_transaction'] = Action(
            name='sign_transaction',
            description='Sign a transaction with the Privy wallet',
            parameters=[
                ActionParameter(name='message', type=str, required=True, description='Message that was signed to authorize this action'),
                ActionParameter(name='signature', type=str, required=True, description='Signature of the message to verify authorization'),
                ActionParameter(name='transaction', type=dict, required=True, description='Transaction details (to, value, chain_id, nonce, gas_limit, data, etc.)'),
                ActionParameter(name='wallet',  type=str, required=True, description='Wallet address to sign the transaction with')
            ]
        )
        
        self.actions['send_transaction'] = Action(
            name='send_transaction',
            description='Send a transaction from the Privy wallet',
            parameters=[
                ActionParameter(name='message', type=str, required=True, description='Message that was signed to authorize this action'),
                ActionParameter(name='signature', type=str, required=True, description='Signature of the message to verify authorization'),
                ActionParameter(name='transaction', type=dict, required=True, description='Transaction details (to, value, data)'),
                ActionParameter(name='chain_id', type=int, required=True, description='Chain ID for caip2 format eip155:{chain_id}'),
                ActionParameter(name='wallet',  type=str, required=True, description='Wallet address to send the transaction from')
            ]
        )

    def configure(self) -> bool:
        """Configure the Privy connection"""
        try:
            self._initialize()
            return True
        except Exception as e:
            logger.error(f"Failed to configure Privy connection: {str(e)}")
            return False

    def is_configured(self, verbose: bool = False) -> bool:
        """Check if the connection is properly configured"""
        try:
            if not self.app_id or not self.app_secret or not self.wallet_address:
                if verbose:
                    logger.error("Privy credentials and wallet not properly configured")
                return False
            return True
        except Exception as e:
            if verbose:
                logger.error(f"Configuration check failed: {str(e)}")
            return False

    def _get_auth_headers(self, body: Dict) -> Dict[str, str]:
        """
        Generate authorization headers for Privy API requests
        
        Args:
            body: Request body
        
        Returns:
            Dict[str, str]: Headers dictionary with appropriate authorization
        """
        headers = {
            'Content-Type': 'application/json',
            'privy-app-id': self.app_id,
            'Authorization': f"Basic {base64.b64encode(f'{self.app_id}:{self.app_secret}'.encode()).decode()}"
        }
        
        # Add authorization signature
        auth_signature = get_authorization_signature(self.rpc_url, body, self.app_id)
        if auth_signature:
            headers['privy-authorization-signature'] = auth_signature
        else:
            logger.warning("Authorization signature could not be generated. Request may fail.")
            
        return headers

    def sign_transaction(self, message: str, signature: str, transaction: Dict, wallet: str) -> Dict:
        """
        Sign a transaction with the Privy wallet
        
        Args:
            message: Message that was signed to authorize this action
            signature: Signature to verify authorization
            transaction: Transaction details dictionary containing:
                - to: Recipient address
                - value: Amount to send (in wei)
                - chain_id: Chain ID
                - nonce: Transaction nonce
                - gas_limit: (optional) Gas limit
                - max_fee_per_gas: (optional) Max fee per gas
                - max_priority_fee_per_gas: (optional) Max priority fee per gas
                - data: (optional) Contract calldata for contract interactions
                - type: (optional) Transaction type (defaults to 2 for EIP-1559)
        """
        # First verify the signature
        if not verify_signature(message, signature):
            raise PrivyConnectionError("Invalid signature. Operation not authorized.")
        
        # Validate required transaction fields
        required_fields = ['to', 'value', 'chain_id', 'nonce']
        for field in required_fields:
            if field not in transaction:
                raise PrivyConnectionError(f"Transaction missing required field: {field}")
                
        # If there's no data field and it's meant for a contract, add empty data
        if 'data' not in transaction:
            transaction['data'] = "0x"
        
        # Build the request body
        body = {
            "address": wallet,
            "chain_type": "ethereum",
            "method": "eth_signTransaction",
            "params": {
                "transaction": transaction
            }
        }
        
        try:
            headers = self._get_auth_headers(body)
            response = requests.post(self.rpc_url, headers=headers, json=body)
            
            if not response.ok:
                error_message = response.json() if response.content else response.reason
                raise PrivyConnectionError(f"API request failed: {error_message}")
                
            return response.json()
        except requests.RequestException as e:
            logger.info(f"Failed to sign transaction: {str(e)}")
            raise PrivyConnectionError(f"Failed to sign transaction: {str(e)}")
        except Exception as e:
            logger.info(f"Failed to sign transaction: {str(e)}")

    def send_transaction(self, message: str, signature: str, transaction: Dict, chain_id: int, wallet: str) -> Dict:
        """
        Send a transaction from the Privy wallet
        
        Args:
            message: Message that was signed to authorize this action
            signature: Signature to verify authorization
            transaction: Transaction details dictionary containing:
                - to: Recipient address
                - value: Amount to send (in wei)
                - data: (optional) Contract calldata for contract interactions
            chain_id: Chain ID for caip2 format
            wallet: Wallet address to send from
        """
        # First verify the signature
        if not verify_signature(message, signature):
            raise PrivyConnectionError("Invalid signature. Operation not authorized.")
        
        # Validate required transaction fields
        required_fields = ['to', 'value']
        for field in required_fields:
            if field not in transaction:
                raise PrivyConnectionError(f"Transaction missing required field: {field}")
        
        # If there's no data field and it's meant for a contract, add empty data
        if 'data' not in transaction:
            transaction['data'] = "0x"
        
        # Build the request body
        body = {
            "address": wallet,
            "chain_type": "ethereum",
            "method": "eth_sendTransaction",
            "caip2": f"eip155:{chain_id}",
            "params": {
                "transaction": transaction
            }
        }
        
        try:
            headers = self._get_auth_headers(body)
            response = requests.post(self.rpc_url, headers=headers, json=body)
            
            if not response.ok:
                error_message = response.json() if response.content else response.reason
                raise PrivyConnectionError(f"API request failed: {error_message}")
                
            return response.json()
            
        except requests.RequestException as e:
            raise PrivyConnectionError(f"Failed to send transaction: {str(e)}")

    def perform_action(self, action_name: str, kwargs: Dict[str, Any]) -> Any:
        """Execute a Privy action with validation"""
        if action_name not in self.actions:
            raise KeyError(f"Unknown action: {action_name}")
        
        if not self.is_configured(verbose=True):
            raise PrivyConnectionError("Privy is not properly configured")

        action = self.actions[action_name]
        validation_errors = action.validate_params(kwargs)
        if validation_errors:
            raise PrivyConnectionError(f"Invalid parameters: {', '.join(validation_errors)}")

        method_name = action_name.replace('-', '_')
        method = getattr(self, method_name)
        return method(**kwargs)
