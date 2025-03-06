import logging
import os
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from .base_connection import BaseConnection, Action, ActionParameter
from src.helpers.privy_helper import get_authorization_signature
from src.helpers.auth_helper import verify_signature, create_auth_token, verify_auth_token
from src.models.wallet import UserWallet
from src.database import SessionLocal

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
            self.base_url = "https://api.privy.io/v1"
            self.wallets_url = f"{self.base_url}/wallets"
            if not self.app_id or not self.app_secret:
                raise PrivyConnectionError("PRIVY_APP_ID or PRIVY_APP_SECRET not found in environment variables")
        except Exception as e:
            raise PrivyConnectionError(f"Failed to initialize Privy connection: {str(e)}")

    def is_llm_provider(self) -> bool:
        return False

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the configuration parameters"""
        return config

    def register_actions(self) -> None:
        """Register wallet and transaction management actions"""
        self.actions['verify_and_create_wallet'] = Action(
            name='verify_and_create_wallet',
            description='Verify user signature and create a new Privy wallet',
            parameters=[
                ActionParameter(name='message', type=str, required=True, description='Message that was signed'),
                ActionParameter(name='signature', type=str, required=True, description='Ethereum signature'),
                ActionParameter(name='user_address', type=str, required=True, description='User\'s Ethereum address')
            ]
        )
        
        self.actions['get_user_wallet'] = Action(
            name='get_user_wallet',
            description='Get wallet for a user using token or address',
            parameters=[
                ActionParameter(name='token', type=str, required=False, description='JWT token'),
                ActionParameter(name='user_address', type=str, required=False, description='User\'s Ethereum address')
            ]
        )
        
        self.actions['create_wallet'] = Action(
            name='create_wallet',
            description='Create a new Privy wallet',
            parameters=[]
        )
        
        self.actions['sign_transaction'] = Action(
            name='sign_transaction',
            description='Sign a transaction with a Privy wallet',
            parameters=[
                ActionParameter(name='wallet_id', type=str, required=True, description='Wallet ID'),
                ActionParameter(name='transaction', type=dict, required=True, description='Transaction details (to, value, chain_id, nonce)')
            ]
        )
        
        self.actions['send_transaction'] = Action(
            name='send_transaction',
            description='Send a transaction from a Privy wallet',
            parameters=[
                ActionParameter(name='wallet_id', type=str, required=True, description='Wallet ID'),
                ActionParameter(name='caip2', type=str, required=True, description='CAIP-2 chain identifier'),
                ActionParameter(name='transaction', type=dict, required=True, description='Transaction details (to, value)')
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
            if not self.app_id or not self.app_secret:
                if verbose:
                    logger.error("Privy credentials are not properly configured")
                return False
            return True
        except Exception as e:
            if verbose:
                logger.error(f"Configuration check failed: {str(e)}")
            return False

    def _get_auth_headers(self, url: Optional[str] = None, body: Optional[Dict] = None) -> Dict[str, str]:
        """
        Generate authorization headers for Privy API requests
        
        Args:
            url: Full URL for the request (required for RPC endpoints)
            body: Request body (required for RPC endpoints)
        
        Returns:
            Dict[str, str]: Headers dictionary with appropriate authorization
        """
        headers = {
            'Content-Type': 'application/json',
            'privy-app-id': self.app_id,
            'Authorization': f"Basic {base64.b64encode(f'{self.app_id}:{self.app_secret}'.encode()).decode()}"
        }
        
        # For RPC endpoints and wallet updates, we need auth signatures
        if url and body and ("/rpc" in url or "/wallets/" in url):
            auth_signature = get_authorization_signature(url, body, self.app_id)
            if auth_signature:
                headers['privy-authorization-signature'] = auth_signature
            else:
                logger.warning("Authorization signature could not be generated. Some requests may fail.")
            
        return headers

    def _make_request(self, method: str, url: str, data: Any = None, headers: Optional[Dict[str, str]] = None) -> Dict:
        """Make an HTTP request to the Privy API"""
        import requests
        
        if headers is None:
            headers = self._get_auth_headers(url=url, body=data)
            
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=data)
            else:
                raise PrivyConnectionError(f"Unsupported HTTP method: {method}")
                
            if not response.ok:
                error_message = response.json() if response.content else response.reason
                raise PrivyConnectionError(f"API request failed: {error_message}")
                
            return response.json()
            
        except requests.RequestException as e:
            raise PrivyConnectionError(f"Request failed: {str(e)}")

    def create_wallet(self) -> Dict:
        """
        Create a new Privy wallet
        """
        wallet_data = {
            "chain_type": "ethereum"
        }
        
        try:
            return self._make_request('POST', self.wallets_url, wallet_data)
        except Exception as e:
            raise PrivyConnectionError(f"Failed to create wallet: {str(e)}")
        
    def sign_transaction(self, wallet_id: str, transaction: Dict) -> Dict:
        """
        Sign a transaction with a Privy wallet
        
        Args:
            wallet_id: Wallet ID
            transaction: Transaction details (to, value, chain_id, etc.)
        """
        required_fields = ['to', 'value', 'chain_id', 'nonce']
        for field in required_fields:
            if field not in transaction:
                raise PrivyConnectionError(f"Transaction missing required field: {field}")
        
        body = {
            "method": "eth_signTransaction",
            "params": {
                "transaction": transaction
            }
        }
        
        try:
            rpc_url = f"{self.wallets_url}/{wallet_id}/rpc"
            # We pass the URL and body so the auth signature can be generated
            return self._make_request('POST', rpc_url, body)
        except Exception as e:
            raise PrivyConnectionError(f"Failed to sign transaction: {str(e)}")

    def send_transaction(self, wallet_id: str, caip2: str, transaction: Dict) -> Dict:
        """
        Send a transaction from a Privy wallet
        
        Args:
            wallet_id: Wallet ID
            caip2: CAIP-2 chain identifier
            transaction: Transaction details (to, value)
        """
        required_fields = ['to', 'value']
        for field in required_fields:
            if field not in transaction:
                raise PrivyConnectionError(f"Transaction missing required field: {field}")
        
        body = {
            "method": "eth_sendTransaction",
            "caip2": caip2,
            "params": {
                "transaction": transaction
            }
        }
        
        try:
            rpc_url = f"{self.wallets_url}/{wallet_id}/rpc"
            # We pass the URL and body so the auth signature can be generated
            return self._make_request('POST', rpc_url, body)
        except Exception as e:
            raise PrivyConnectionError(f"Failed to send transaction: {str(e)}")

    def verify_and_create_wallet(self, message: str, signature: str, user_address: str) -> Dict:
        """
        Verify user's signature and create a Privy wallet
        
        Args:
            message: Message that was signed
            signature: Ethereum signature
            user_address: User's Ethereum address
            
        Returns:
            Dict: Contains wallet info and auth token if successful
        """
        # First verify the signature
        if not verify_signature(message, signature, user_address):
            raise PrivyConnectionError("Invalid signature")
        
        # Check if user already has a wallet
        db = SessionLocal()
        try:
            existing_wallet = db.query(UserWallet).filter(
                UserWallet.user_address == user_address,
                UserWallet.is_active == True
            ).first()
            
            if existing_wallet:
                # Generate new token
                token, expiry = create_auth_token(user_address, existing_wallet.wallet_id)
                
                # Update session token
                existing_wallet.session_token = token
                existing_wallet.session_expires_at = expiry
                existing_wallet.last_accessed_at = datetime.utcnow()
                db.commit()
                
                return {
                    "wallet_id": existing_wallet.wallet_id,
                    "wallet_address": existing_wallet.wallet_address,
                    "chain_type": existing_wallet.chain_type,
                    "token": token,
                    "expires_at": expiry.isoformat(),
                    "created": False
                }
            
            # Create a new wallet
            wallet_data = self.create_wallet()
            
            # Create auth token
            token, expiry = create_auth_token(user_address, wallet_data["id"])
            
            # Store wallet info in database
            user_wallet = UserWallet(
                id=wallet_data["id"],
                user_address=user_address,
                wallet_address=wallet_data["address"],
                wallet_id=wallet_data["id"],
                chain_type=wallet_data.get("chain_type", "ethereum"),
                session_token=token,
                session_expires_at=expiry
            )
            
            db.add(user_wallet)
            db.commit()
            
            return {
                "wallet_id": wallet_data["id"],
                "wallet_address": wallet_data["address"],
                "chain_type": wallet_data.get("chain_type", "ethereum"),
                "token": token,
                "expires_at": expiry.isoformat(),
                "created": True
            }
            
        except Exception as e:
            db.rollback()
            raise PrivyConnectionError(f"Failed to create wallet: {str(e)}")
        finally:
            db.close()

    def get_user_wallet(self, token: Optional[str] = None, user_address: Optional[str] = None) -> Dict:
        """
        Get wallet for a user using token or address
        
        Args:
            token: JWT token (preferred method)
            user_address: User's Ethereum address (fallback)
            
        Returns:
            Dict: User wallet information
        """
        if not token and not user_address:
            raise PrivyConnectionError("Either token or user_address must be provided")
            
        db = SessionLocal()
        try:
            if token:
                # Verify the token
                payload = verify_auth_token(token)
                if not payload:
                    raise PrivyConnectionError("Invalid or expired token")
                    
                user_wallet = db.query(UserWallet).filter(
                    UserWallet.wallet_id == payload["wallet_id"],
                    UserWallet.is_active == True
                ).first()
            else:
                # Look up by address
                user_wallet = db.query(UserWallet).filter(
                    UserWallet.user_address == user_address,
                    UserWallet.is_active == True
                ).first()
                
            if not user_wallet:
                raise PrivyConnectionError("Wallet not found")
                
            # Update last accessed time
            user_wallet.last_accessed_at = datetime.utcnow()
            db.commit()
            
            return {
                "wallet_id": user_wallet.wallet_id,
                "wallet_address": user_wallet.wallet_address,
                "chain_type": user_wallet.chain_type,
                "user_address": user_wallet.user_address,
                "created_at": user_wallet.created_at
            }
            
        except Exception as e:
            db.rollback()
            if isinstance(e, PrivyConnectionError):
                raise
            raise PrivyConnectionError(f"Failed to get wallet: {str(e)}")
        finally:
            db.close()

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
