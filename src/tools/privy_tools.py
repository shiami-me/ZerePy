from langchain.tools import BaseTool
import json
import logging
from src.helpers.auth_helper import sign_message
from typing import Optional
from web3 import Web3

logger = logging.getLogger("tools.privy_tools")

class PrivySendTransactionTool(BaseTool):
    name: str = "privy_send_transaction"
    description: str = """
    privy_send_transaction: Send a transaction from a Privy wallet
    chain_id = 146 (Sonic mainnet)
    This tool signs and sends a transaction using the Privy wallet.
    
    Input should be a JSON string with transaction details such as(not strict):
    - sender: The Privy wallet address to use for sending
    - to: Recipient address (contract address)
    - value: Amount to send (in wei)
    - data: Contract calldata for contract interactions
    - gas: Gas limit
    - maxFeePerGas: Maximum fee per gas
    - maxPriorityFeePerGas: Maximum priority fee per gas
    - nonce: Transaction nonce
    - tokenAddress: (optional) For token transactions, the address of the token
    - amount: (optional) For token transactions, the amount in token units
    - type: (optional) Transaction type identifier (e.g., "deposit", "withdraw")
    """

    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def _check_and_approve_token(self, sender: str, token_address: str, spender: str, amount: int) -> Optional[dict]:
        try:
            # Define ERC20 ABI for allowance and approve functions
            erc20_abi = [
                {
                    "constant": True,
                    "inputs": [
                        {"name": "_owner", "type": "address"},
                        {"name": "_spender", "type": "address"}
                    ],
                    "name": "allowance",
                    "outputs": [{"name": "", "type": "uint256"}],
                    "payable": False,
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "constant": False,
                    "inputs": [
                        {"name": "_spender", "type": "address"},
                        {"name": "_value", "type": "uint256"}
                    ],
                    "name": "approve",
                    "outputs": [{"name": "", "type": "bool"}],
                    "payable": False,
                    "stateMutability": "nonpayable",
                    "type": "function"
                }
            ]
            
            # Get web3 connection
            w3 = Web3(Web3.HTTPProvider("https://rpc.soniclabs.com"))
            
            # Create token contract instance
            token_contract = w3.eth.contract(address=token_address, abi=erc20_abi)
            
            # Check current allowance
            current_allowance = token_contract.functions.allowance(sender, spender).call()
            
            # If allowance is less than amount, create approval transaction
            if current_allowance < amount:
                logger.info(f"Insufficient allowance. Current: {current_allowance}, Required: {amount}")
                
                # Use high gas values in hex format as per Privy API requirements
                max_priority_fee = w3.to_wei('5', 'gwei')
                max_fee = w3.to_wei('150', 'gwei')
                
                hex_max_priority_fee = hex(max_priority_fee)
                hex_max_fee = hex(max_fee)
                
                logger.info(f"Using fixed gas prices - Max priority fee: {hex_max_priority_fee}, Max fee: {hex_max_fee}")
                
                # Create approval function
                approve_function = token_contract.functions.approve(spender, amount)
                
                # Estimate gas for the transaction
                estimated_gas = approve_function.estimate_gas({'from': sender})
                gas_with_buffer = int(estimated_gas * 1.2)
                hex_gas = hex(gas_with_buffer)
                
                # Get current nonce
                nonce = w3.eth.get_transaction_count(sender)
                
                # Get the approval call data
                call_data = approve_function.build_transaction({
                    'from': sender,
                    'gas': gas_with_buffer,
                    'nonce': nonce,
                    'maxFeePerGas': max_fee,
                    'maxPriorityFeePerGas': max_priority_fee,
                })['data']
                
                # Build transaction in the exact format expected by Privy API
                return {
                    "to": token_address,
                    "value": 0,
                    "data": call_data,
                    "chain_id": 146,  # Sonic chain
                    "gas_limit": hex_gas,  # Use gas_limit instead of gas
                    "max_fee_per_gas": hex_max_fee,  # Use hex string format
                    "max_priority_fee_per_gas": hex_max_priority_fee,  # Use hex string format
                    "nonce": nonce,
                    "type": 2
                }
            return None
        except Exception as e:
            logger.error(f"Error checking token allowance: {str(e)}")
            return None
    def _run(self, query: str) -> str:
        try:
            # Parse the input
            params = json.loads(query)
            
            # Extract required parameters
            if 'sender' not in params:
                return json.dumps({"error": "Missing required field: sender"})
                
            # Initialize web3 connection
            w3 = Web3(Web3.HTTPProvider("https://rpc.soniclabs.com"))
                
            # Check if we need to handle token approval first
            if 'tokenAddress' in params and params.get('tokenAddress') and 'amount' in params and 'to' in params:
                approval_tx = self._check_and_approve_token(
                    sender=params['sender'],
                    token_address=params['tokenAddress'],
                    spender=params['to'],
                    amount=int(params['amount'])
                )
                
                # If approval is needed, sign the approval transaction first
                if approval_tx:
                    auth_message = f"I authorize signing a token approval transaction for {params['to']}"
                    auth_signature = sign_message(auth_message)
                    
                    # Sign the approval transaction using Privy
                    approval_response = self._agent.connection_manager.connections["privy"].sign_transaction(
                        message=auth_message,
                        signature=auth_signature,
                        transaction=approval_tx,
                        wallet=params['sender']
                    )
                    
                    # Extract the signed transaction
                    logger.info(f"Sign approval transaction response: {approval_response}")
                    if "data" in approval_response and "signed_transaction" in approval_response["data"]:
                        raw_tx = approval_response["data"]["signed_transaction"]
                        
                        # Send the raw transaction
                        tx_hash = w3.eth.send_raw_transaction(raw_tx)
                        logger.info(f"Approval transaction sent with hash: {tx_hash.hex()}")
                        
                        # Wait for receipt
                        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
                        logger.info(f"Approval transaction receipt: {receipt}")
                        
                        # Add approval context to the response
                        approval_response["tx_hash"] = tx_hash.hex()
                        approval_response["receipt"] = {
                            "blockNumber": receipt.blockNumber,
                            "gasUsed": receipt.gasUsed,
                            "status": receipt.status
                        }
                        approval_response["requiresApproval"] = True
                        approval_response["approvalSent"] = True
                        approval_response["approvalConfirmed"] = (receipt.status == 1)
                        
                        # If approval failed, return error
                        if receipt.status != 1:
                            return json.dumps({
                                "error": "Token approval transaction failed",
                                "approvalTxHash": tx_hash.hex(),
                                "receipt": approval_response["receipt"]
                            })
                    else:
                        return json.dumps({
                            "error": "Failed to sign approval transaction",
                            "privy_response": approval_response
                        })
            
            # Create more descriptive auth message based on transaction type
            tx_type = params.get('type', 'transaction')
            tx_value = params.get('value', 0)
            tx_to = params.get('to', '')
            
            auth_message = f"I authorize signing a {tx_type} transaction to {tx_to} with {tx_value} wei"
            auth_signature = sign_message(auth_message)
            
            # Prepare the transaction object with all the fields from the input
            transaction = {}
            for key in ['to', 'value', 'data', 'gas', 'nonce', 'maxFeePerGas', 'maxPriorityFeePerGas']:
                if key in params:
                    transaction[key] = params[key]
            
            # Ensure transaction has type field (default to EIP-1559)
            if 'type' not in transaction:
                transaction['type'] = 2
            logger.info(f"Transaction: {transaction}")
            transaction = {
                "to": transaction["to"],
                "value": transaction["value"],
                "data": transaction["data"],
                "chain_id": 146,  # Sonic chain
                "gas_limit": transaction["gas"],  # Use gas_limit instead of gas
                "max_fee_per_gas": transaction["maxFeePerGas"],  # Use hex string format
                "max_priority_fee_per_gas": transaction["maxPriorityFeePerGas"],  # Use hex string format
                "nonce": w3.eth.get_transaction_count(params["sender"]),
                "type": transaction["type"]
            }
            # Sign the transaction using Privy
            signed_response = self._agent.connection_manager.connections["privy"].sign_transaction(
                message=auth_message,
                signature=auth_signature,
                transaction=transaction,
                wallet=params["sender"]
            )
            
            logger.info(f"Sign transaction response: {signed_response}")
            
            # Check if we have a signed transaction
            if "data" not in signed_response or "signed_transaction" not in signed_response["data"]:
                return json.dumps({
                    "error": "Failed to sign transaction",
                    "privy_response": signed_response
                })
                
            # Get the raw signed transaction
            raw_tx = signed_response["data"]["signed_transaction"]
            
            # Send the raw transaction
            tx_hash = w3.eth.send_raw_transaction(raw_tx)
            logger.info(f"Transaction sent with hash: {tx_hash.hex()}")
            
            # Wait for receipt
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            logger.info(f"Transaction receipt: {receipt}")
            
            # Prepare response
            response = {
                "tx_hash": tx_hash.hex(),
                "receipt": {
                    "blockNumber": receipt.blockNumber,
                    "gasUsed": receipt.gasUsed,
                    "status": receipt.status
                },
                "success": (receipt.status == 1)
            }
            
            # Add helpful context to the response
            if 'tokenAddress' in params:
                response["tokenAddress"] = params["tokenAddress"]
            if 'type' in params:
                response["transactionType"] = params["type"]
                
            return json.dumps(response)
            
        except Exception as e:
            logger.error(f"Send transaction operation failed: {str(e)}")
            return json.dumps({"error": str(e)})


def get_privy_tools(agent) -> list:
    """Return a list of all Privy-related tools."""
    return [
        PrivySendTransactionTool(agent),
    ]
