import os
from jwt import JWT, exceptions
from datetime import datetime, timedelta
from eth_account.messages import encode_defunct
from web3 import Web3
from dotenv import load_dotenv
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger("helpers.auth_helper")

load_dotenv()

jwt = JWT()

# Get JWT secret from environment or use a default for development
JWT_SECRET = os.getenv("JWT_SECRET", "your-super-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 240  # Token expires after 10 days



def verify_signature(message: str, signature: str, address: str) -> bool:
    """
    Verify an Ethereum signature
    
    Args:
        message: The message that was signed
        signature: The signature to verify
        address: The Ethereum address that supposedly signed the message
        
    Returns:
        bool: True if the signature is valid, False otherwise
    """
    try:
        w3 = Web3()
        message_hash = encode_defunct(text=message)
        recovered_address = w3.eth.account.recover_message(message_hash, signature=signature)
        return recovered_address.lower() == address.lower()
    except Exception as e:
        logger.error(f"Error verifying signature: {str(e)}")
        return False


def create_auth_token(user_address: str, wallet_id: str) -> Tuple[str, datetime]:
    """
    Create a JWT token for user authentication
    
    Args:
        user_address: Ethereum address of the user
        wallet_id: ID of the user's Privy wallet
        
    Returns:
        tuple: (token, expiry datetime)
    """
    expiry = datetime.now() + timedelta(hours=JWT_EXPIRATION_HOURS)
    
    payload = {
        "sub": user_address,
        "wallet_id": wallet_id,
        "exp": expiry.timestamp(),
        "iat": datetime.now().timestamp()
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    return token, expiry


def verify_auth_token(token: str) -> Optional[Dict]:
    """
    Verify a JWT token
    
    Args:
        token: The JWT token to verify
        
    Returns:
        dict: The decoded token payload if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except exceptions.JWTDecodeError:
        logger.error("Token has expired")
        return None
    except exceptions.InvalidKeyTypeError:
        logger.error("Invalid token")
        return None
