import json
import base64
import os
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from dotenv import load_dotenv
import logging

logger = logging.getLogger("helpers.privy_helper")

def load_auth_key():
    """Load the Privy authorization key from environment variables"""
    load_dotenv()
    auth_key = os.getenv("PRIVY_AUTHORIZATION_KEY")
    if not auth_key:
        logger.warning("PRIVY_AUTHORIZATION_KEY not found in environment variables")
        return None
    return auth_key


def canonicalize(obj):
    """
    Simple JSON canonicalization function.
    Sorts dictionary keys and ensures consistent formatting.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def get_authorization_signature(url, body, app_id):
    """
    Generate authorization signature for Privy API requests using ECDSA and hashlib.
    
    Args:
        url (str): The full URL for the API request
        body (dict): The request body
        app_id (str): The Privy app ID
        
    Returns:
        str: Base64-encoded signature or None if authorization key is not available
    """
    try:
        # Get authorization key
        auth_key = load_auth_key()
        if not auth_key:
            return None
            
        # Construct the payload
        payload = {
            "version": 1,
            "method": "POST" if "/rpc" in url else "PATCH",
            "url": url,
            "body": body,
            "headers": {"privy-app-id": app_id},
        }

        # Serialize the payload to JSON
        serialized_payload = canonicalize(payload)

        # Create ECDSA P-256 signing key from private key
        private_key_string = auth_key.replace("wallet-auth:", "")
        private_key_pem = (
            f"-----BEGIN PRIVATE KEY-----\n{private_key_string}\n-----END PRIVATE KEY-----"
        )

        # Load the private key from PEM format
        private_key = serialization.load_pem_private_key(
            private_key_pem.encode("utf-8"), password=None
        )

        # Sign the message using ECDSA with SHA-256
        signature = private_key.sign(
            serialized_payload.encode("utf-8"), ec.ECDSA(hashes.SHA256())
        )

        # Convert the signature to base64 for easy transmission
        return base64.b64encode(signature).decode("utf-8")
        
    except Exception as e:
        logger.error(f"Failed to generate authorization signature: {str(e)}")
        return None
