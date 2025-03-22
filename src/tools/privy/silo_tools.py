from src.tools.silo_tools import SiloDepositTool, SiloBorrowTool, SiloRepayTool, SiloWithdrawTool, SiloClaimRewardsTool, SiloLoopingStrategyTool, get_silo_config_address, SiloBorrowSharesTool
from .sonic_tools import SonicSwapToolPrivy
import json
import logging
from web3 import Web3
from src.constants.abi import ERC20_ABI, SILO_ABI
from .helpers import approve_token, execute

logger = logging.getLogger("tools.privy.silo_tools")

# Multicall contract details
MULTICALL_ADDRESS = "0x1ca3352a55A3D69cA8e5aB06006e08aaF2bD09f1"
MULTICALL_ABI = [
    {
        "inputs": [
            {
                "components": [
                    {"internalType": "address", "name": "target", "type": "address"},
                    {"internalType": "bytes", "name": "callData", "type": "bytes"}
                ],
                "internalType": "struct PendleMulticallV2.Call[]",
                "name": "calls",
                "type": "tuple[]"
            }
        ],
        "name": "aggregate",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function"
    }
]

class SiloDepositToolPrivy(SiloDepositTool):
    description: str = """
    privy_silo_deposit: Deposit tokens into a Silo smart contract using Privy wallet.
    Ex - deposit Collateral 1000 USDC into Sonic/USDC pair. Then token_0: USDC, token_1: Sonic, collateral_type: 1 (collateral), amount: 1000.0, sender: "0xYourWalletAddress"
        deposit Protected 100 Sonic into Sonic/USDC pair. Then token_0: Sonic, token_1: USDC, collateral_type: 0 (protected), amount: 100.0, sender: "0xYourWalletAddress"
    Args:
        token_0: Symbol of the token to deposit
        token_1: Symbol of the other token in the Silo pair
        amount: Amount of assets to deposit
        collateral_type: Type of collateral (0 for Protected, 1 for Collateral)
        sender: Address of the sender
        id: Optional market ID to specify a specific market
        return_tx_only: If true, only return transaction data without executing (optional)
    """
    
    def _run(self, token_0: str, token_1: str, amount: float,
             collateral_type: int = 0, sender: str = None, id: int = None, return_tx_only: bool = False):
        try:
            # Call the parent method to get response data
            response = super()._run(token_0, token_1, amount, collateral_type, sender, id)
            
            if isinstance(response, str) and "Error" in response:
                return {"error": response}
            
            w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
            
            logger.info(response)
            token_contract = w3.eth.contract(
                address=response["tokenAddress"], abi=ERC20_ABI)
            current_allowance = token_contract.functions.allowance(
                sender, response["to"]).call()
            
            # Create transaction array for multicall
            transactions = []
            
            # Prepare approval transaction if needed
            if current_allowance < response["amount"]:
                logger.info(
                    f"Preparing approval for {token_0}. Current allowance: {current_allowance}, Required: {response['amount']}")
                
                # Build approval transaction data
                approval_tx_data = token_contract.functions.approve(
                    response["to"], response["amount"]
                ).build_transaction({
                    'from': sender,
                    'nonce': w3.eth.get_transaction_count(sender),
                    'gas': 200000,
                    'gasPrice': w3.eth.gas_price
                })['data']
                
                transactions.append({
                    "to": response["tokenAddress"],
                    "data": approval_tx_data,
                    "value": 0
                })
            
            # Add the deposit transaction
            transactions.append({
                "to": response["to"],
                "data": response["data"],
                "value": 0
            })
            
            # If return_tx_only is True, just return the transaction data without executing
            if return_tx_only:
                return json.dumps({
                    "type": "deposit_transaction_details",
                    "token": token_0,
                    "token_address": response["tokenAddress"],
                    "amount": amount,
                    "collateral_type": collateral_type,
                    "silo_address": response["to"],
                    "transactions": transactions
                })
            
            # Execute the transactions directly if not returning transactions only
            approval_result = None
            
            # Execute approval if needed
            if current_allowance < response["amount"]:
                approval_result = approve_token(
                    self._agent.connection_manager.connections["privy"],
                    response["tokenAddress"], response["to"], response["amount"], sender)
                
                if "error" in approval_result:
                    return {"error": approval_result["error"]}
                
                if not approval_result.get("success", False):
                    return {"error": "Token approval transaction failed", "tx_hash": approval_result.get("tx_hash")}
                
                logger.info(f"Token approval successful: {approval_result.get('tx_hash')}")
            
            # Execute deposit transaction
            auth_message = f"I authorize Silo deposit of {amount} {token_0} tokens (type: {collateral_type})"
            deposit_result = execute(
                self._agent.connection_manager.connections["privy"],
                sender,
                response["to"],
                response["data"],
                0,
                "deposit",
                auth_message
            )
            
            if "error" in deposit_result:
                return {"error": deposit_result["error"]}
            
            result = {
                "type": "deposit",
                "tokenAddress": response["tokenAddress"],
                "token": token_0,
                "amount": amount,
                "collateral_type": collateral_type,
                "status": "Completed" if deposit_result.get("success", False) else "Failed",
                "sender": sender,
                "silo_address": response["to"],
                "tx_hash": deposit_result.get("tx_hash"),
                "receipt": deposit_result.get("receipt"),
                "success": deposit_result.get("success", False),
                "gas_stats": deposit_result.get("gas_stats")
            }
            
            if approval_result:
                result["approval"] = {
                    "tx_hash": approval_result.get("tx_hash"),
                    "success": approval_result.get("success", False),
                    "gas_stats": approval_result.get("gas_stats")
                }
            return result
            
        except Exception as e:
            logger.error(f"Error in Privy Silo deposit: {str(e)}")
            return {"error": f"Deposit failed: {str(e)}"}

class SiloBorrowToolPrivy(SiloBorrowTool):
    description: str = """
    privy_silo_borrow: Borrow tokens from a Silo smart contract using Privy wallet.
    Ex - borrow 1000 USDC from Sonic/USDC pair. Then token_0: USDC, token_1: Sonic, amount: 1000.0, sender: "0xYourWalletAddress"
        borrow 100 Sonic from Sonic/USDC pair. Then token_0: Sonic, token_1: USDC, amount: 100.0, sender: "0xYourWalletAddress"
    Args:
        token_0: Symbol of the token to borrow
        token_1: Symbol of the other token in the Silo pair
        amount: Amount of tokens to borrow
        sender: Address of the sender
        receiver: Address to receive the borrowed assets (optional, defaults to sender)
        id: Optional market ID to specify a specific market
        return_tx_only: If true, only return transaction data without executing (optional)
    """
    
    def _run(self, token_0: str, token_1: str, amount: float,
             sender: str, receiver: str = None, id: int = None, return_tx_only: bool = False):
        try:
            # Call the parent method to get response data
            response = super()._run(token_0, token_1, amount, sender, receiver, id)
            
            if isinstance(response, str) and "Error" in response:
                return {"error": response}
            
            if receiver is None:
                receiver = sender
                
            w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
            
            silo_contract = w3.eth.contract(address=response["to"], abi=SILO_ABI)
            
            # Check if can borrow the requested amount
            max_borrow_amount = silo_contract.functions.maxBorrow(sender).call()
            
            if response["amount"] > max_borrow_amount:
                return {"error": f"Borrow amount exceeds maximum borrowable amount: {max_borrow_amount / (10 ** response.get('decimals', 18))}"}
            
            # Transaction array for multicall
            transactions = [{
                "to": response["to"],
                "data": response["data"],
                "value": 0
            }]
            
            # If return_tx_only is True, just return the transaction data without executing
            if return_tx_only:
                return json.dumps({
                    "type": "borrow_transaction_details",
                    "token": token_0,
                    "token_address": response["tokenAddress"],
                    "amount": amount,
                    "max_borrow_amount": max_borrow_amount / (10 ** response.get('decimals', 18)),
                    "silo_address": response["to"],
                    "transactions": transactions
                })
            
            # Execute the borrow transaction directly
            auth_message = f"I authorize Silo borrow of {amount} {token_0} tokens to {receiver}"
            borrow_result = execute(
                self._agent.connection_manager.connections["privy"],
                sender,
                response["to"],
                response["data"],
                0,
                "borrow",
                auth_message
            )
            
            if "error" in borrow_result:
                return {"error": borrow_result["error"]}
            
            result = {
                "type": "borrow",
                "tokenAddress": response["tokenAddress"],
                "token": token_0,
                "amount": amount,
                "max_borrow_amount": max_borrow_amount / (10 ** response.get('decimals', 18)),
                "status": "Completed" if borrow_result.get("success", False) else "Failed",
                "sender": sender,
                "receiver": receiver,
                "silo_address": response["to"],
                "tx_hash": borrow_result.get("tx_hash"),
                "receipt": borrow_result.get("receipt"),
                "success": borrow_result.get("success", False),
                "gas_stats": borrow_result.get("gas_stats")
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Privy Silo borrow: {str(e)}")
            return {"error": f"Borrow failed: {str(e)}"}

class SiloBorrowSharesToolPrivy(SiloBorrowSharesTool):
    def _run(self, token_0: str, token_1: str, amount: float,
             sender: str, receiver: str = None, id: int = None, return_tx_only: bool = False):
        try:
            # Call the parent method to get response data
            response = super()._run(token_0, token_1, amount, sender, receiver, id)
            
            if isinstance(response, str) and "Error" in response:
                return {"error": response}
            
            if receiver is None:
                receiver = sender
                
            w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
            
            silo_contract = w3.eth.contract(address=response["to"], abi=SILO_ABI)
            
            # Transaction array for multicall
            transactions = [{
                "to": response["to"],
                "data": response["data"],
                "value": 0
            }]
            
            # If return_tx_only is True, just return the transaction data without executing
            if return_tx_only:
                return json.dumps({
                    "type": "borrow_transaction_details",
                    "token": token_0,
                    "token_address": response["tokenAddress"],
                    "amount": amount,
                    "silo_address": response["to"],
                    "transactions": transactions
                })
            
            # Execute the borrow transaction directly
            auth_message = f"I authorize Silo borrow of {amount} {token_0} tokens to {receiver}"
            borrow_result = execute(
                self._agent.connection_manager.connections["privy"],
                sender,
                response["to"],
                response["data"],
                0,
                "borrow",
                auth_message
            )
            
            if "error" in borrow_result:
                return {"error": borrow_result["error"]}
            
            result = {
                "type": "borrow",
                "tokenAddress": response["tokenAddress"],
                "token": token_0,
                "amount": amount,
                "status": "Completed" if borrow_result.get("success", False) else "Failed",
                "sender": sender,
                "receiver": receiver,
                "silo_address": response["to"],
                "tx_hash": borrow_result.get("tx_hash"),
                "receipt": borrow_result.get("receipt"),
                "success": borrow_result.get("success", False),
                "gas_stats": borrow_result.get("gas_stats")
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Privy Silo borrow: {str(e)}")
            return {"error": f"Borrow failed: {str(e)}"}

class SiloLoopingStrategyToolPrivy(SiloLoopingStrategyTool):
    name: str = "privy_silo_looping_execute"
    description: str = """
    privy_silo_looping_execute: Execute a Silo looping strategy in a single transaction using multicall.
    
    This tool takes a looping strategy (deposit-borrow-swap loop) and executes it in a single transaction,
    optimizing for the best yield by borrowing 95% of available credit after each deposit.
    
    Ex - execute looping strategy for Sonic/USDC with 1000 USDC. Then token: "USDC,S", initial_amount: 1000, sender: "0xYourAddress"
        execute S/USDC optimized looping strategy. Then token: "S,USDC", initial_amount: 500, loops: 3, sender: "0xYourAddress"
    
    Args:
        token: Comma-separated tokens for the market (e.g. "USDC,S")
        initial_amount: Initial capital to start the loop with
        sender: Your wallet address
        loops: Number of loops to execute (default: 3)
        borrow_percentage: Percentage of max borrow to use in each loop (default: 95)
    """
    
    def _run(self, token: str, initial_amount: float, sender: str, 
             loops: int = 3, borrow_percentage: float = 95) -> str:
        try:
            # Parse token pair
            tokens = token.split(',')
            if len(tokens) != 2:
                return json.dumps({"error": "Please provide exactly two tokens separated by comma"})
            
            token_0, token_1 = tokens[0].strip(), tokens[1].strip()
            
            logger.info(f"Executing looping strategy for {token_0}/{token_1} with initial amount {initial_amount}")
            
            silo_config_address, is_token0_silo0, _, decimals0 = get_silo_config_address(token_0, token_1)
            _, _, _, decimals1 = get_silo_config_address(token_1, token_0)
            token_idx = 0 if is_token0_silo0 else 1
            silo0_address = self._agent.connection_manager.connections["silo"]._get_silo_address(silo_config_address, token_idx)
            
            silo1_address = self._agent.connection_manager.connections["silo"]._get_silo_address(silo_config_address, 1 - token_idx)
            w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
            silo0_contract = w3.eth.contract(address=silo0_address, abi=SILO_ABI)
            silo1_contract = w3.eth.contract(address=silo1_address, abi=SILO_ABI)
            # Get tokens and market details
            deposit_token = token_0  # We'll deposit the first token
            borrow_token = token_1   # And borrow the second token
            
            # Now build the transaction sequence
            multicall_transactions = []
            current_amount = initial_amount
            value = 0
            deposit_shares = silo0_contract.functions.previewDeposit(int(initial_amount*(10**decimals0)), 1).call()
            # For the initial deposit
            deposit_result = SiloDepositToolPrivy(self._agent, "")._run(
                token_0=deposit_token,
                token_1=borrow_token,
                amount=current_amount,
                collateral_type=1,  # Collateral
                sender=sender,
                return_tx_only=True
            )
            
            deposit_result = json.loads(deposit_result) if isinstance(deposit_result, str) else deposit_result
            
            if "error" in deposit_result:
                return json.dumps({"error": f"Failed to prepare initial deposit: {deposit_result['error']}"})
            
            # Add deposit transactions to multicall
            for tx in deposit_result["transactions"]:
                multicall_transactions.append({
                    "target": tx["to"],
                    "callData": tx["data"]
                })
            
            # Store details for reporting
            looping_details = []
            total_borrowed = 0
            total_deposited = initial_amount
            
            # Execute loops
            for loop in range(1, loops + 1):
                logger.info(f"Preparing loop {loop}")
                
                # Calculate amount to borrow (95% of max borrow or custom percentage)
                borrow_amount = (deposit_shares * 0.95 * (borrow_percentage / 100)) / (10 ** decimals1)  # Adjust for token decimals
                
                if borrow_amount <= 0:
                    logger.info(f"Borrow amount too small after loop {loop-1}")
                    break
                
                logger.info(f"Loop {loop}: Borrowing {borrow_amount} {borrow_token} (95% of max borrow)")
                
                # 1. Borrow step
                borrow_result = SiloBorrowSharesToolPrivy(self._agent, "")._run(
                    token_0=borrow_token,
                    token_1=deposit_token,
                    amount=borrow_amount,
                    sender=sender,
                    return_tx_only=True
                )
                borrow_assets_amount = float(silo1_contract.functions.convertToAssets(int(borrow_amount*(10**decimals1))).call()) / (10**decimals1)
                borrow_result = json.loads(borrow_result) if isinstance(borrow_result, str) else borrow_result
                
                if "error" in borrow_result:
                    return json.dumps({"error": f"Failed to prepare borrow for loop {loop}: {borrow_result['error']}"})
                
                # Add borrow transaction to multicall
                for tx in borrow_result["transactions"]:
                    multicall_transactions.append({
                        "target": tx["to"],
                        "callData": tx["data"]
                    })
                
                total_borrowed += borrow_amount
                
                # 2. Swap step - Convert borrowed token back to deposit token
                swap_result = SonicSwapToolPrivy(self._agent, "")._run(
                    from_token=borrow_token,
                    to_token="wS",
                    amount=borrow_assets_amount,
                    sender=sender,
                    return_tx_only=True
                )
                
                swap_result = json.loads(swap_result) if isinstance(swap_result, str) else swap_result
                
                if "error" in swap_result:
                    return json.dumps({"error": f"Failed to prepare swap for loop {loop}: {swap_result['error']}"})
                
                # Add swap transactions to multicall (approval + swap)
                for tx in swap_result["transactions"]:
                    value += int(tx.get("value", 0))
                    multicall_transactions.append({
                        "target": tx["to"],
                        "callData": tx["data"]
                    })
                
                # 3. Deposit step - Deposit the swapped tokens back as collateral
                deposit_amount = float(swap_result["amount_out"])  # Convert back to deposit token
                
                if deposit_amount <= 0:
                    logger.info(f"Deposit amount too small after swap in loop {loop}")
                    break
                
                logger.info(f"Loop {loop}: Depositing {deposit_amount} {deposit_token} after swap")
                deposit_shares = silo0_contract.functions.previewDeposit(int(deposit_amount*(10**decimals0)), 1).call()
                deposit_result = SiloDepositToolPrivy(self._agent, "")._run(
                    token_0=deposit_token,
                    token_1=borrow_token,
                    amount=deposit_amount,
                    collateral_type=1,  # Collateral
                    sender=sender,
                    return_tx_only=True
                )
                
                deposit_result = json.loads(deposit_result) if isinstance(deposit_result, str) else deposit_result
                
                if "error" in deposit_result:
                    return json.dumps({"error": f"Failed to prepare deposit for loop {loop}: {deposit_result['error']}"})
                
                # Add deposit transactions to multicall
                for tx in deposit_result["transactions"]:
                    multicall_transactions.append({
                        "target": tx["to"],
                        "callData": tx["data"]
                    })
                
                total_deposited += deposit_amount
                
                # Store loop details for reporting
                looping_details.append({
                    "loop": loop,
                    "borrow_amount": borrow_amount,
                    "deposit_amount": deposit_amount,
                })
            
            # If no loops were possible
            if len(looping_details) == 0:
                return json.dumps({"error": "No loops were possible with the current market conditions and parameters"})
            
            # Calculate leverage
            leverage = total_deposited / initial_amount if initial_amount > 0 else 1.0
            
            # Execute all transactions via multicall
            multicall_contract = w3.eth.contract(address=MULTICALL_ADDRESS, abi=MULTICALL_ABI)
            
            multicall_tx = multicall_contract.functions.aggregate(
                multicall_transactions
            ).build_transaction({
                'from': sender,
                'nonce': w3.eth.get_transaction_count(sender),
                'gas': 5000000,  # High gas limit for complex transaction
                'gasPrice': w3.eth.gas_price,
                'value': value
            })
            
            # Execute the multicall transaction with Privy
            auth_message = f"I authorize executing {len(looping_details)} loops of Silo strategy with {initial_amount} {deposit_token}"
            multicall_result = execute(
                self._agent.connection_manager.connections["privy"],
                sender,
                MULTICALL_ADDRESS,
                multicall_tx['data'],
                value,  # Pass the value needed for the transaction
                "silo_looping",
                auth_message
            )
            
            if "error" in multicall_result:
                return json.dumps({"error": multicall_result["error"]})
            
            # Build the success result
            result = {
                "status": "success",
                "type": "silo_loop_strategy",
                "token_pair": f"{deposit_token}/{borrow_token}",
                "initial_amount": initial_amount,
                "completed_loops": len(looping_details),
                "total_deposited": total_deposited,
                "total_borrowed": total_borrowed,
                "leverage": f"{leverage:.2f}x",
                "tx_hash": multicall_result.get("tx_hash"),
                "receipt": multicall_result.get("receipt"),
                "loop_details": looping_details,
                "transactions_count": len(multicall_transactions)
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error executing looping strategy: {str(e)}")
            return json.dumps({"error": f"Execution failed: {str(e)}"})

# Keep the other Privy Silo tools as is
class SiloRepayToolPrivy(SiloRepayTool):
    def _run(self, token_0: str, token_1: str, amount: float,
             sender: str = None, id: int = None):
        try:
            response = super()._run(token_0, token_1, amount, sender, id)
            
            if isinstance(response, str) and "Error" in response:
                return {"error": response}
                
            w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
            
            token_contract = w3.eth.contract(
                address=response["tokenAddress"], abi=ERC20_ABI)
            current_allowance = token_contract.functions.allowance(
                sender, response["to"]).call()
            
            approval_result = None
            if current_allowance < response["amount"]:
                logger.info(
                    f"Approving {token_0} for Silo repay. Current allowance: {current_allowance}, Required: {response['amount']}")
                
                amount_to_approve = response["amount"]
                
                approval_result = approve_token(
                    self._agent.connection_manager.connections["privy"],
                    response["tokenAddress"], response["to"], amount_to_approve, sender)
                
                if "error" in approval_result:
                    return {"error": approval_result["error"]}
                
                if not approval_result.get("success", False):
                    return {"error": "Token approval transaction failed", "tx_hash": approval_result.get("tx_hash")}
                
                logger.info(
                    f"Token approval successful: {approval_result.get('tx_hash')}")
            
            auth_message = f"I authorize Silo repayment of {amount} {token_0} tokens"
            repay_result = execute(
                self._agent.connection_manager.connections["privy"],
                sender,
                response["to"],
                response["data"],
                0,
                "repay",
                auth_message
            )
            
            if "error" in repay_result:
                return {"error": repay_result["error"]}
            
            result = {
                "type": "repay",
                "tokenAddress": response["tokenAddress"],
                "token": token_0,
                "amount": amount,
                "status": "Completed" if repay_result.get("success", False) else "Failed",
                "sender": sender,
                "silo_address": response["to"],
                "tx_hash": repay_result.get("tx_hash"),
                "receipt": repay_result.get("receipt"),
                "success": repay_result.get("success", False),
                "gas_stats": repay_result.get("gas_stats")
            }
            
            if approval_result:
                result["approval"] = {
                    "tx_hash": approval_result.get("tx_hash"),
                    "success": approval_result.get("success", False),
                    "gas_stats": approval_result.get("gas_stats")
                }
            return result
            
        except Exception as e:
            logger.error(f"Error in Privy Silo repay: {str(e)}")
            return {"error": f"Repay failed: {str(e)}"}

class SiloWithdrawToolPrivy(SiloWithdrawTool):
    def _run(self, token_0: str, token_1: str, amount: float, receiver: str = None,
             collateral_type: int = 0, sender: str = None, id: int = None):
        try:
            response = super()._run(token_0, token_1, amount, receiver, collateral_type, sender, id)
            
            if isinstance(response, str) and "Error" in response:
                return {"error": response}
            
            if receiver is None:
                receiver = sender
            
            auth_message = f"I authorize Silo withdrawal of {amount} {token_0} tokens (type: {collateral_type}) to {receiver}"
            withdraw_result = execute(
                self._agent.connection_manager.connections["privy"],
                sender,
                response["to"],
                response["data"],
                0,
                "withdraw",
                auth_message
            )
            
            if "error" in withdraw_result:
                return {"error": withdraw_result["error"]}
            
            result = {
                "type": "withdraw",
                "tokenAddress": response["tokenAddress"],
                "token": token_0,
                "amount": amount,
                "collateral_type": collateral_type,
                "status": "Completed" if withdraw_result.get("success", False) else "Failed",
                "sender": sender,
                "receiver": receiver,
                "silo_address": response["to"],
                "tx_hash": withdraw_result.get("tx_hash"),
                "receipt": withdraw_result.get("receipt"),
                "success": withdraw_result.get("success", False),
                "gas_stats": withdraw_result.get("gas_stats")
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Privy Silo withdraw: {str(e)}")
            return {"error": f"Withdraw failed: {str(e)}"}

class SiloClaimRewardsToolPrivy(SiloClaimRewardsTool):
    def _run(self, sender: str = None):
        try:
            response = super()._run(sender)
            
            if isinstance(response, str) and "Error" in response:
                return {"error": response}
            
            auth_message = f"I authorize claiming Silo rewards"
            claim_result = execute(
                self._agent.connection_manager.connections["privy"],
                sender,
                response["to"],
                response["data"],
                0,
                "claim_rewards",
                auth_message
            )
            
            if "error" in claim_result:
                return {"error": claim_result["error"]}
            
            result = {
                "type": "claim_rewards",
                "status": "Completed" if claim_result.get("success", False) else "Failed",
                "sender": sender,
                "tx_hash": claim_result.get("tx_hash"),
                "receipt": claim_result.get("receipt"),
                "success": claim_result.get("success", False),
                "gas_stats": claim_result.get("gas_stats")
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Privy Silo claim rewards: {str(e)}")
            return {"error": f"Claim rewards failed: {str(e)}"}

def get_privy_silo_tools(agent, llm) -> list:
    """Return a list of all Privy Silo-related tools."""
    return [
        SiloDepositToolPrivy(agent, llm),
        SiloBorrowToolPrivy(agent, llm),
        SiloRepayToolPrivy(agent, llm),
        SiloWithdrawToolPrivy(agent, llm),
        SiloClaimRewardsToolPrivy(agent, llm),
        SiloLoopingStrategyToolPrivy(agent, llm)
    ]
