import logging
from web3 import Web3
from src.helpers.auth_helper import sign_message

logger = logging.getLogger("tools.privy.helpers")

def execute_privy_transaction(privy, transaction, wallet_address, auth_message=None):
    """
    Execute a transaction using Privy connection
    
    Args:
        privy: Privy connection instance
        transaction: Transaction details dictionary
        wallet_address: The wallet address to sign and send from
        auth_message: Custom authorization message (optional)
        
    Returns:
        Transaction execution result dict
    """
    try:
        if auth_message is None:
            auth_message = f"I authorize signing a transaction to {transaction.get('to')} with {transaction.get('value')} wei"
        auth_signature = sign_message(auth_message)

        w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))

        gas_price_info = w3.eth.fee_history(1, 'latest', [10, 50, 90])

        base_fee = gas_price_info.baseFeePerGas[-1]

        priority_fee = gas_price_info.reward[-1][2]

        max_fee = base_fee * 2 + priority_fee

        if not max_fee or max_fee == 0:
            priority_fee = w3.to_wei('2', 'gwei')
            max_fee = w3.to_wei('150', 'gwei')

        gas_limit = transaction.get("gas_limit")
        if not gas_limit:
            try:
                tx_for_estimate = {
                    "from": wallet_address,
                    "to": transaction["to"],
                    "value": transaction.get("value", 0),
                    "data": transaction.get("data", "0x")
                }
                estimated_gas = int(
                    w3.eth.estimate_gas(tx_for_estimate) * 1.2)
                gas_limit = hex(estimated_gas)
            except Exception as e:
                logger.warning(
                    f"Gas estimation failed: {str(e)}. Using default gas limit.")

        privy_tx = {
            "to": transaction["to"],
            "value": transaction.get("value", 0),
            "data": transaction.get("data", "0x"),
            "gas_limit": gas_limit,
            "max_fee_per_gas": hex(max_fee),
            "max_priority_fee_per_gas": hex(priority_fee),
            "nonce": transaction.get("nonce", w3.eth.get_transaction_count(wallet_address)),
            "chain_id": 146
        }

        logger.info(
            f"Transaction gas parameters: limit={gas_limit}, max_fee={hex(max_fee)}, priority_fee={hex(priority_fee)}")

        signed_response = privy.sign_transaction(
            message=auth_message,
            signature=auth_signature,
            transaction=privy_tx,
            wallet=wallet_address
        )

        logger.info(f"Sign transaction response: {signed_response}")

        if "data" not in signed_response or "signed_transaction" not in signed_response["data"]:
            return {
                "error": "Failed to sign transaction",
                "privy_response": signed_response
            }

        raw_tx = signed_response["data"]["signed_transaction"]

        tx_hash = w3.eth.send_raw_transaction(raw_tx)
        logger.info(f"Transaction sent with hash: {tx_hash.hex()}")

        receipt = None
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                receipt = w3.eth.wait_for_transaction_receipt(
                    tx_hash, timeout=120)
                break
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt+1}/{max_attempts} waiting for receipt failed: {str(e)}")
                if attempt == max_attempts - 1:
                    return {
                        "tx_hash": tx_hash.hex(),
                        "status": "pending",
                        "error": "Transaction was sent but receipt not available yet"
                    }

        logger.info(f"Transaction receipt: {receipt}")

        return {
            "tx_hash": tx_hash.hex(),
            "receipt": {
                "blockNumber": receipt.blockNumber,
                "gasUsed": receipt.gasUsed,
                "status": receipt.status,
                "effectiveGasPrice": receipt.effectiveGasPrice
            },
            "success": (receipt.status == 1),
            "gas_stats": {
                "gas_limit_used": gas_limit,
                "gas_used": receipt.gasUsed,
                "effective_gas_price": receipt.effectiveGasPrice,
                "cost_wei": receipt.gasUsed * receipt.effectiveGasPrice,
                "cost_eth": w3.from_wei(receipt.gasUsed * receipt.effectiveGasPrice, 'ether')
            }
        }

    except Exception as e:
        logger.error(f"Failed to execute Privy transaction: {str(e)}")
        return {"error": str(e)}

def approve_token(privy, token_address, spender_address, amount, sender):
    """
    Approve token spending for a specific contract
    
    Args:
        privy: Privy connection instance
        token_address: Address of the token contract
        spender_address: Address of the contract that will spend tokens
        amount: Amount to approve (in token units)
        sender: Address of the sender (wallet address)
        
    Returns:
        Approval transaction result
    """
    try:
        w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))

        from src.constants.abi import ERC20_ABI
        token_contract = w3.eth.contract(
            address=token_address, abi=ERC20_ABI)

        approve_call_data = token_contract.encodeABI(
            fn_name="approve",
            args=[spender_address, amount]
        )

        gas_estimate = None
        try:
            estimate_tx = {
                "from": sender,
                "to": token_address,
                "data": approve_call_data
            }
            gas_estimate = w3.eth.estimate_gas(estimate_tx)
            gas_with_buffer = int(gas_estimate * 1.3)
        except Exception as e:
            logger.warning(
                f"Gas estimation for approval failed: {str(e)}. Using default.")
            gas_with_buffer = 100000

        nonce = w3.eth.get_transaction_count(sender)

        approval_tx = {
            "to": token_address,
            "value": 0,
            "data": approve_call_data,
            "gas_limit": hex(gas_with_buffer),
            "nonce": nonce,
            "type": 2
        }

        auth_message = f"I authorize approving {token_address} to spend {amount} tokens to {spender_address}"

        logger.info(
            f"Executing token approval transaction with gas limit: {hex(gas_with_buffer)}")
        return execute_privy_transaction(privy, approval_tx, sender, auth_message)

    except Exception as e:
        logger.error(f"Failed to approve token: {str(e)}")
        return {"error": f"Token approval failed: {str(e)}"}

def execute(privy, from_address, to_address, data, value=0, action_name="transaction", auth_message=None):
    """
    Generic function to execute any transaction through Privy
    
    Args:
        privy: Privy connection instance
        from_address: Sender address
        to_address: Target contract address
        data: Encoded function call data
        value: ETH value to send (in wei)
        action_name: Human-readable name of the action (for logging and auth message)
        auth_message: Custom authorization message (optional)
        
    Returns:
        Transaction execution result
    """
    try:
        w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
        
        gas_estimate = None
        try:
            estimate_tx = {
                "from": from_address,
                "to": to_address,
                "value": value,
                "data": data
            }
            gas_estimate = w3.eth.estimate_gas(estimate_tx)
            gas_with_buffer = int(gas_estimate * 1.5)
        except Exception as e:
            logger.warning(
                f"Gas estimation for {action_name} failed: {str(e)}. Using default.")
            gas_with_buffer = 300000
        
        # Get nonce
        nonce = w3.eth.get_transaction_count(from_address)
        
        tx = {
            "to": to_address,
            "value": value,
            "data": data,
            "gas_limit": hex(gas_with_buffer),
            "nonce": nonce,
            "type": 2  # EIP-1559
        }
        
        if auth_message is None:
            auth_message = f"I authorize {action_name}"
        
        logger.info(f"Executing {action_name} transaction with gas limit: {hex(gas_with_buffer)}")
        
        # Execute the transaction
        return execute_privy_transaction(privy, tx, from_address, auth_message)
    
    except Exception as e:
        logger.error(f"Failed to execute {action_name}: {str(e)}")
        return {"error": f"{action_name.capitalize()} failed: {str(e)}"}
