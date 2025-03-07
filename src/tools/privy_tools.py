from langchain.tools import BaseTool
import json
import logging
from src.helpers.auth_helper import sign_message
from typing import Optional

logger = logging.getLogger("tools.privy_tools")

class PrivySignTransactionTool(BaseTool):
    name: str = "privy_sign_transaction"
    description: str = """
    privy_sign_transaction: Sign a transaction with a Privy wallet
    
    This tool signs a transaction using the Privy wallet without sending it.
    You can use this for preparing transactions to be sent later.
    
    Input should be a JSON string with:
    - wallet: The Privy wallet address to use for signing
    - to: Recipient address
    - value: Amount to send (in wei)
    - nonce: Transaction nonce
    - gas_limit: (Optional) Gas limit (default: 21000)
    - max_fee_per_gas: (Optional) Max fee per gas
    - max_priority_fee_per_gas: (Optional) Max priority fee per gas
    - data: (Optional) Contract calldata for contract interactions (function selector + encoded parameters)
    """

    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def _run(self, query: str) -> str:
        try:
            # Parse the input
            params = json.loads(query)
            required_fields = ['wallet', 'to', 'value', 'chain_id', 'nonce']
            
            for field in required_fields:
                if field not in params:
                    return json.dumps({"error": f"Missing required field: {field}"})
            
            # Create more descriptive auth message based on whether it's a contract call or not
            is_contract_call = params.get('data') is not None and params.get('data') != '0x' and params.get('data') != ''
            
            if is_contract_call:
                auth_message = f"I authorize signing a contract transaction to {params['to']} with {params['value']} wei and calldata on chain {params['chain_id']}"
            else:
                auth_message = f"I authorize signing a transaction to {params['to']} for {params['value']} wei on chain {params['chain_id']}"
                
            auth_signature = sign_message(auth_message)
            
            # Prepare the transaction
            transaction = {
                "to": params["to"],
                "value": int(params["value"]),
                "chain_id": 146,
                "nonce": int(params["nonce"]),
                "type": 2  # EIP-1559 transaction
            }
            
            # Add optional parameters
            if "gas_limit" in params:
                transaction["gas_limit"] = params["gas_limit"]
                
            if "max_fee_per_gas" in params:
                transaction["max_fee_per_gas"] = params["max_fee_per_gas"]
                
            if "max_priority_fee_per_gas" in params:
                transaction["max_priority_fee_per_gas"] = params["max_priority_fee_per_gas"]
                
            # Add contract data if present
            if "data" in params and params["data"]:
                transaction["data"] = params["data"]
            
            # Call the Privy connection
            response = self._agent.connection_manager.connections["privy"].sign_transaction(
                message=auth_message,
                signature=auth_signature,
                transaction=transaction,
                wallet=params["wallet"]
            )
            
            return json.dumps(response)
            
        except Exception as e:
            logger.error(f"Sign transaction operation failed: {str(e)}")
            return json.dumps({"error": str(e)})


class PrivySendTransactionTool(BaseTool):
    name: str = "privy_send_transaction"
    description: str = """
    privy_send_transaction: Send a transaction from a Privy wallet
    
    This tool sends a transaction using the Privy wallet.
    
    Input should be a JSON string with:
    - wallet: The Privy wallet address to use for sending
    - to: Recipient address
    - value: Amount to send (in wei)
    - data: (Optional) Contract calldata for contract interactions (function selector + encoded parameters)
    """

    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def _run(self, query: str) -> str:
        try:
            # Parse the input
            params = json.loads(query)
            required_fields = ['wallet', 'to', 'value']
            
            for field in required_fields:
                if field not in params:
                    return json.dumps({"error": f"Missing required field: {field}"})
            
            # Create more descriptive auth message based on whether it's a contract call or not
            is_contract_call = params.get('data') is not None and params.get('data') != '0x' and params.get('data') != ''
            
            if is_contract_call:
                auth_message = f"I authorize sending a contract transaction to {params['to']} with {params['value']} wei and calldata on chain {params['chain_id']}"
            else:
                auth_message = f"I authorize sending a transaction to {params['to']} for {params['value']} wei on chain {params['chain_id']}"
                
            auth_signature = sign_message(auth_message)
            
            # Prepare the transaction
            transaction = {
                "to": params["to"],
                "value": int(params["value"])
            }
            
            # Add contract data if present
            if "data" in params and params["data"]:
                transaction["data"] = params["data"]
            
            # Call the Privy connection
            response = self._agent.connection_manager.connections["privy"].send_transaction(
                message=auth_message,
                signature=auth_signature,
                transaction=transaction,
                chain_id=146,
                wallet=params["wallet"]
            )
            
            return json.dumps(response)
            
        except Exception as e:
            logger.error(f"Send transaction operation failed: {str(e)}")
            return json.dumps({"error": str(e)})


def get_privy_tools(agent) -> list:
    """Return a list of all Privy-related tools."""
    return [
        PrivySignTransactionTool(agent),
        PrivySendTransactionTool(agent)
    ]
