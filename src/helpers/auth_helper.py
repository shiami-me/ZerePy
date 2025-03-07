import os
from eth_account.messages import encode_defunct
from eth_account import Account
from web3 import Web3
from dotenv import load_dotenv
import logging

logger = logging.getLogger("helpers.auth_helper")

load_dotenv()

# Get JWT secret from environment or use a default for development
JWT_SECRET = os.getenv("JWT_SECRET", "your-super-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 240  # Token expires after 10 days

def get_wallet_from_private_key():
    """
    Get wallet address from private key in environment
    
    Returns:
        tuple: (address, private_key) or (None, None) if not found
    """
    private_key = os.getenv("ETH_PRIVATE_KEY")
    if not private_key:
        logger.error("ETH_PRIVATE_KEY not found in environment variables")
        return None, None
        
    if not private_key.startswith("0x"):
        private_key = f"0x{private_key}"
        
    try:
        account = Account.from_key(private_key)
        return account.address, private_key
    except Exception as e:
        logger.error(f"Failed to get wallet from private key: {str(e)}")
        return None, None

def verify_signature(message: str, signature: str) -> bool:
    """
    Verify that a signature was signed by the wallet from ETH_PRIVATE_KEY
    
    Args:
        message: The message that was signed
        signature: The signature to verify
        
    Returns:
        bool: True if the signature is valid and from our wallet, False otherwise
    """
    try:
        wallet_address, _ = get_wallet_from_private_key()
        if not wallet_address:
            return False
            
        w3 = Web3()
        message_hash = encode_defunct(text=message)
        recovered_address = w3.eth.account.recover_message(message_hash, signature=signature)
        
        return recovered_address.lower() == wallet_address.lower()
    except Exception as e:
        logger.error(f"Error verifying signature: {str(e)}")
        return False

def sign_message(message: str) -> str:
    """
    Sign a message with the wallet from ETH_PRIVATE_KEY
    
    Args:
        message: The message to sign
        
    Returns:
        str: The signature
    """
    try:
        _, private_key = get_wallet_from_private_key()
        if not private_key:
            raise ValueError("ETH_PRIVATE_KEY not available")
            
        w3 = Web3()
        message_hash = encode_defunct(text=message)
        signed_message = w3.eth.account.sign_message(message_hash, private_key=private_key)
        
        return signed_message.signature.hex()
    except Exception as e:
        logger.error(f"Error signing message: {str(e)}")
        raise ValueError(f"Failed to sign message: {str(e)}")

