from src.tools.silo_tools import SiloDepositTool, SiloBorrowTool, SiloRepayTool, SiloWithdrawTool, SiloClaimRewardsTool, SiloLoopingStrategyTool, get_silo_config_address, SiloBorrowSharesTool
from .sonic_tools import SonicSwapToolPrivy, SonicWrapToolPrivy
import json
import logging
from web3 import Web3
from src.constants.abi import ERC20_ABI, SILO_ABI
from .helpers import approve_token, execute
from decimal import Decimal, ROUND_FLOOR, getcontext

def ceil_to_8_decimal(number):
    getcontext().prec = 10 
    d = Decimal(str(number))
    result = d.quantize(Decimal('0.00000001'), rounding=ROUND_FLOOR)
    return float(result)

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
    
    def _run(self, token: str, initial_amount: float, sender: str, 
             loops: int = 3, borrow_percentage: float = 95, id: int = None, deposit_token: str = None) -> str:
        try:
            # Parse token pair
            tokens = token.split(',')
            if len(tokens) != 2:
                return json.dumps({"error": "Please provide exactly two tokens separated by comma"})
            
            token_0, token_1 = tokens[0].strip(), tokens[1].strip()
            
            # If deposit_token is not specified, use token_0
            if deposit_token is None:
                deposit_token = token_0
            
            logger.info(f"Executing looping strategy for {token_0}/{token_1} with initial amount {initial_amount} using deposit token {deposit_token}")
            
            # Get Silo config and token information
            try:
                silo_config_result = get_silo_config_address(token_0, token_1, id)
                if isinstance(silo_config_result, dict) and "error" in silo_config_result:
                    return json.dumps({"error": f"Failed to get market configuration: {silo_config_result['error']}"})
                
                silo_config_address, is_token0_silo0, deposit_token_address, decimals0 = silo_config_result
                _, _, _, decimals1 = get_silo_config_address(token_1, token_0, id)
                
                # Determine which token is collateral and which is borrowed
                silo_deposit_token = token_0
                borrow_token = token_1
                deposit_token_decimals = decimals0
                borrow_token_decimals = decimals1
                
                logger.info(f"Deposit token (silo): {silo_deposit_token}, Decimals: {deposit_token_decimals}")
                logger.info(f"Borrow token: {borrow_token}, Decimals: {borrow_token_decimals}")
            except Exception as e:
                logger.error(f"Error getting token information: {str(e)}")
                return json.dumps({"error": f"Failed to get token information: {str(e)}"})
            
            # Store details for reporting
            looping_details = []
            total_borrowed = 0
            total_deposited = initial_amount
            amount_to_deposit = initial_amount
            
            # Initialize web3 connection
            w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
            
            # Check if we need to swap the deposit token to silo_deposit_token
            swapped = False
            
            if deposit_token.upper() != silo_deposit_token.upper():
                # Need to swap deposit_token to silo_deposit_token first
                logger.info(f"Swapping {initial_amount} {deposit_token} to {silo_deposit_token}")
                
                # Special case: if silo_deposit_token is S/wS, we'll get wS after swap
                swap_to_token = silo_deposit_token
                if silo_deposit_token.upper() == "S":
                    swap_to_token = "wS"
                
                swap_result = SonicSwapToolPrivy(self._agent, "")._run(
                    from_token=deposit_token if deposit_token.upper() != "S" else "wS",
                    to_token=swap_to_token,
                    amount=initial_amount,
                    sender=sender,
                    slippage=0.5
                )
                
                if "error" in swap_result:
                    return json.dumps({"error": f"Failed to swap deposit token: {swap_result['error']}"})
                
                logger.info(f"Swap result: {swap_result}")
                swapped = True
                amount_to_deposit = swap_result["amount_out"]
                logger.info(f"Will deposit {amount_to_deposit} {silo_deposit_token} after swap")
            
            # Check if we need to wrap S to wS for Silo
            elif silo_deposit_token.upper() == "S":
                logger.info(f"Wrapping {amount_to_deposit} S to wS for Silo deposit")
                wrap_result = SonicWrapToolPrivy(self._agent)._run(
                    amount=amount_to_deposit,
                    sender=sender
                )
                
                if "error" in wrap_result:
                    return json.dumps({"error": f"Failed to wrap S to wS: {wrap_result['error']}"})
                
                logger.info(f"Wrap result: {wrap_result}")
                wrapped = True
            
            # Step 1: Initial deposit
            try:
                logger.info(f"Step 1: Initial deposit of {amount_to_deposit} {silo_deposit_token}")
                deposit_result = SiloDepositToolPrivy(self._agent, "")._run(
                    token_0=silo_deposit_token,
                    token_1=borrow_token,
                    amount=amount_to_deposit,
                    collateral_type=1,  # Collateral
                    sender=sender,
                    id=id
                )
                
                if isinstance(deposit_result, str):
                    deposit_result = json.loads(deposit_result)
                
                if "error" in deposit_result:
                    return json.dumps({"error": f"Initial deposit failed: {deposit_result['error']}"})
                
                if not deposit_result.get("success", False):
                    return json.dumps({"error": "Initial deposit transaction failed", "deposit_result": deposit_result})
                
                logger.info(f"Initial deposit success: {deposit_result.get('tx_hash')}")
                
                # Get actual deposited shares for better borrowing calculation
                w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
                
                # Get Silo addresses based on token positions
                token_idx = 0 if is_token0_silo0 else 1
                silo0_address = self._agent.connection_manager.connections["silo"]._get_silo_address(
                    silo_config_address, token_idx)
                silo1_address = self._agent.connection_manager.connections["silo"]._get_silo_address(
                    silo_config_address, 1 - token_idx)
                
                silo0_contract = w3.eth.contract(address=silo0_address, abi=SILO_ABI)
                silo1_contract = w3.eth.contract(address=silo1_address, abi=SILO_ABI)
                
            except Exception as e:
                logger.error(f"Error in initial deposit: {str(e)}")
                return json.dumps({"error": f"Initial deposit failed with exception: {str(e)}"})
            
            # Execute loops
            for loop in range(1, loops + 1):
                try:
                    logger.info(f"Starting loop {loop}")
                    
                    # Step 2: Check max borrow and calculate amount
                    max_borrow = silo1_contract.functions.maxBorrow(sender).call() / (10 ** borrow_token_decimals)
                    borrow_amount = max_borrow * (borrow_percentage / 100)
                    
                    if borrow_amount <= 0.00001:  # Minimum threshold to avoid dust amounts
                        logger.info(f"Borrow amount too small: {borrow_amount}, stopping loops")
                        break
                    
                    logger.info(f"Loop {loop}: Borrowing {borrow_amount} {borrow_token} ({borrow_percentage}% of max borrow {max_borrow})")
                    
                    # Step 3: Borrow
                    borrow_result = SiloBorrowToolPrivy(self._agent, "")._run(
                        token_0=borrow_token,
                        token_1=silo_deposit_token,
                        amount=borrow_amount,
                        sender=sender,
                        id=id
                    )
                    
                    if isinstance(borrow_result, str):
                        borrow_result = json.loads(borrow_result)
                    
                    if "error" in borrow_result:
                        logger.error(f"Borrow failed in loop {loop}: {borrow_result['error']}")
                        break
                    
                    if not borrow_result.get("success", False):
                        logger.error(f"Borrow transaction failed in loop {loop}")
                        break
                    
                    logger.info(f"Borrow success: {borrow_result.get('tx_hash')}")
                    total_borrowed += borrow_amount
                    
                    # Step 4: Swap borrowed token back to deposit token
                    logger.info(f"Loop {loop}: Swapping {borrow_amount} {borrow_token} to {silo_deposit_token}")
                    deposit_token_contract = w3.eth.contract(address=deposit_token_address, abi=ERC20_ABI)
                    # Check deposit token balance before swap
                    deposit_token_balance_before_wei = deposit_token_contract.functions.balanceOf(sender).call()
                    deposit_token_balance_before = deposit_token_balance_before_wei / (10 ** deposit_token_decimals)
                    logger.info(f"Balance before swap: {deposit_token_balance_before} {silo_deposit_token}")
                    
                    # Fix swap retry mechanism with proper structure
                    swap_success = False
                    
                    for i in range(10):  # Try up to 10 times with increasing slippage
                        try:
                            current_slippage = 0.5 + (i * 0.5)  # Start at 0.5% and increase by 0.5% each retry
                            logger.info(f"Swap attempt {i+1} with slippage {current_slippage}%")
                            
                            swap_result = SonicSwapToolPrivy(self._agent, "")._run(
                                from_token=borrow_token if borrow_token.upper() != "S" else "wS",
                                to_token=silo_deposit_token if silo_deposit_token.upper() != "S" else "wS",
                                amount=borrow_amount,
                                sender=sender,
                                slippage=current_slippage
                            )
                            
                            # Check if swap was successful
                            if swap_result and "error" not in swap_result:
                                # Check deposit token balance after swap to calculate actual received amount
                                deposit_token_balance_after_wei = deposit_token_contract.functions.balanceOf(sender).call()
                                deposit_token_balance_after = deposit_token_balance_after_wei / (10 ** deposit_token_decimals)
                                
                                # Calculate the actual amount received based on balance difference
                                deposit_amount = deposit_token_balance_after - deposit_token_balance_before
                                
                                # Round to avoid precision issues
                                deposit_amount = float(f"{ceil_to_8_decimal(deposit_amount)}")
                                
                                logger.info(f"Balance after swap: {deposit_token_balance_after} {silo_deposit_token}")
                                logger.info(f"Actual swap output: {deposit_amount} {silo_deposit_token}")
                                
                                if deposit_amount > 0.00001:  # Minimum threshold
                                    swap_success = True
                                    logger.info(f"Swap successful on attempt {i+1}: Got {deposit_amount} {silo_deposit_token}")
                                    break
                                else:
                                    logger.warning(f"Swap output too small: {deposit_amount}")
                            else:
                                error_msg = swap_result.get("error", "Unknown error") if swap_result else "Empty swap result"
                                logger.warning(f"Swap attempt {i+1} failed: {error_msg}")
                        except Exception as swap_error:
                            logger.warning(f"Exception during swap attempt {i+1}: {str(swap_error)}")
                    
                    # Check if all swap attempts failed
                    if not swap_success:
                        logger.error(f"All swap attempts failed in loop {loop}")
                        break
                    
                    # Step 5: Deposit the swapped tokens back as collateral
                    logger.info(f"Loop {loop}: Depositing {deposit_amount} {silo_deposit_token}")
                    
                    deposit_result = SiloDepositToolPrivy(self._agent, "")._run(
                        token_0=silo_deposit_token,
                        token_1=borrow_token,
                        amount=deposit_amount,
                        collateral_type=1,  # Collateral
                        sender=sender,
                        id=id
                    )
                    
                    if isinstance(deposit_result, str):
                        deposit_result = json.loads(deposit_result)
                    
                    if "error" in deposit_result:
                        logger.error(f"Deposit failed in loop {loop}: {deposit_result['error']}")
                        break
                    
                    if not deposit_result.get("success", False):
                        logger.error(f"Deposit transaction failed in loop {loop}")
                        break
                    
                    logger.info(f"Deposit success: {deposit_result.get('tx_hash')}")
                    total_deposited += deposit_amount
                    
                    # Store loop details with fixed precision
                    looping_details.append({
                        "loop": loop,
                        "max_borrow": float(f"{ceil_to_8_decimal(max_borrow)}"),
                        "borrow_amount": float(f"{ceil_to_8_decimal(borrow_amount)}"),
                        "swap_amount_out": float(f"{ceil_to_8_decimal(deposit_amount)}"),
                        "deposit_amount": float(f"{ceil_to_8_decimal(deposit_amount)}"),
                        "borrow_tx": borrow_result.get("tx_hash"),
                        "deposit_tx": deposit_result.get("tx_hash")
                    })
                    
                except Exception as e:
                    logger.error(f"Error in loop {loop}: {str(e)}")
                    break
            
            # If no loops were completed
            if len(looping_details) == 0:
                result = {
                    "status": "partial_success",
                    "message": "Initial deposit succeeded but no loops were completed",
                    "initial_deposit": {
                        "deposit_token": deposit_token,  # Original deposit token specified by user
                        "silo_deposit_token": silo_deposit_token if silo_deposit_token.upper() != "S" else "wS",  # Actual token deposited in Silo (might be wS)
                        "amount": initial_amount,
                        "tx_hash": deposit_result.get("tx_hash")
                    }
                }
                
                if swapped:
                    result["initial_deposit"]["swapped"] = True
                    result["initial_deposit"]["swap_amount"] = amount_to_deposit
                
                if wrapped:
                    result["initial_deposit"]["wrapped"] = True
                
                return json.dumps(result)
            
            # Calculate leverage
            leverage = total_deposited / initial_amount if initial_amount > 0 else 1.0
            
            # Build the success result with fixed precision
            result = {
                "status": "success",
                "type": "silo_loop_strategy",
                "deposit_token": {
                    "original": deposit_token,  # Original deposit token specified by user
                    "silo": silo_deposit_token if silo_deposit_token.upper() != "S" else "wS"  # Actual token deposited in Silo (might be wS)
                },
                "token_pair": f"{silo_deposit_token}/{borrow_token}",
                "initial_amount": float(f"{ceil_to_8_decimal(initial_amount)}"),
                "completed_loops": len(looping_details),
                "total_deposited": float(f"{ceil_to_8_decimal(total_deposited)}"),
                "total_borrowed": float(f"{ceil_to_8_decimal(total_borrowed)}"),
                "leverage": f"{leverage:.2f}x",
                "initial_deposit_tx": deposit_result.get("tx_hash"),
                "loop_details": looping_details
            }
            
            # Add swap/wrap information
            if swapped:
                result["token_conversion"] = {
                    "type": "swap",
                    "from_token": deposit_token,
                    "to_token": silo_deposit_token if silo_deposit_token.upper() != "S" else "wS",
                    "original_amount": initial_amount,
                    "converted_amount": amount_to_deposit
                }
            elif wrapped:
                result["token_conversion"] = {
                    "type": "wrap",
                    "from_token": "S",
                    "to_token": "wS",
                    "amount": initial_amount
                }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error executing looping strategy: {str(e)}")
            return json.dumps({"error": f"Execution failed: {str(e)}"})

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
                "success": claim_result.get("success", False),
                "gas_stats": claim_result.get("gas_stats")
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Privy Silo claim rewards: {str(e)}")
            return {"error": f"Claim rewards failed: {str(e)}"}

class SiloExitStrategyToolPrivy:
    def __init__(self, agent, llm=None):
        self._agent = agent
        self._llm = llm
    
    def _run(self, strategy_result: str, sender: str, swap_slippage: float = 0.5, id: int = None) -> str:
        try:
            # Parse the strategy result
            if isinstance(strategy_result, str):
                strategy_data = json.loads(strategy_result)
            else:
                strategy_data = strategy_result
                
            if "status" not in strategy_data or strategy_data["status"] != "success" or "type" not in strategy_data or strategy_data["type"] != "silo_loop_strategy":
                return json.dumps({"error": "Invalid strategy result. Expected output from SiloLoopingStrategyToolPrivy."})
            
            # Extract key information
            token_pair = strategy_data["token_pair"]
            tokens = token_pair.split('/')
            if len(tokens) != 2:
                return json.dumps({"error": f"Invalid token pair format: {token_pair}. Expected format: 'TokenA/TokenB'"})
            
            # Get the original deposit token specified by the user
            original_deposit_token = strategy_data.get("deposit_token", {}).get("original", tokens[0].strip())
            
            # Get the tokens used in Silo
            silo_deposit_token = tokens[0].strip()
            borrow_token = tokens[1].strip()
            total_borrowed = float(strategy_data["total_borrowed"])
            total_deposited = float(strategy_data["total_deposited"])
            
            logger.info(f"Unwinding Silo strategy for {token_pair}")
            logger.info(f"Original deposit token: {original_deposit_token}")
            logger.info(f"Silo deposit token: {silo_deposit_token}")
            logger.info(f"Total borrowed: {total_borrowed} {borrow_token}")
            logger.info(f"Total deposited: {total_deposited} {silo_deposit_token}")
            
            # Get token information and silo addresses
            try:
                # Initialize web3
                w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
                
                # Silo Lens address for querying borrows
                SILO_LENS_ADDRESS = "0xB6AdBb29f2D8ae731C7C72036A7FD5A7E970B198"
                SILO_LENS_ABI = [
                    {
                        "inputs": [
                            {"internalType": "address", "name": "silo", "type": "address"},
                            {"internalType": "address", "name": "user", "type": "address"}
                        ],
                        "name": "debtBalanceOfUnderlying",
                        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                        "stateMutability": "view",
                        "type": "function"
                    }
                ]
                
                silo_lens_contract = w3.eth.contract(address=SILO_LENS_ADDRESS, abi=SILO_LENS_ABI)
                
                # Get Silo configuration and token info
                silo_config_result = get_silo_config_address(silo_deposit_token, borrow_token, id)
                if isinstance(silo_config_result, dict) and "error" in silo_config_result:
                    return json.dumps({"error": f"Failed to get market configuration: {silo_config_result['error']}"})
                
                silo_config_address, is_deposit_token_silo0, deposit_token_address, deposit_token_decimals = silo_config_result
                borrow_config_result = get_silo_config_address(borrow_token, silo_deposit_token, id)
                _, _, borrow_token_address, borrow_token_decimals = borrow_config_result

                # Get the silo addresses based on token positions
                token_idx = 0 if is_deposit_token_silo0 else 1
                deposit_silo_address = self._agent.connection_manager.connections["silo"]._get_silo_address(
                    silo_config_address, token_idx)
                borrow_silo_address = self._agent.connection_manager.connections["silo"]._get_silo_address(
                    silo_config_address, 1 - token_idx)
                
                # Initialize token contracts and silo contracts
                deposit_token_contract = w3.eth.contract(address=deposit_token_address, abi=ERC20_ABI)
                borrow_token_contract = w3.eth.contract(address=borrow_token_address, abi=ERC20_ABI)
                deposit_silo_contract = w3.eth.contract(address=deposit_silo_address, abi=SILO_ABI)
                borrow_silo_contract = w3.eth.contract(address=borrow_silo_address, abi=SILO_ABI)
                
                logger.info(f"Deposit silo: {deposit_silo_address}")
                logger.info(f"Borrow silo: {borrow_silo_address}")
                
                # Check initial wallet balances
                initial_deposit_token_balance_wei = deposit_token_contract.functions.balanceOf(sender).call()
                initial_deposit_token_balance = initial_deposit_token_balance_wei / (10 ** deposit_token_decimals)
                initial_deposit_token_balance = float(f"{ceil_to_8_decimal(initial_deposit_token_balance)}")
                
                initial_borrow_token_balance_wei = borrow_token_contract.functions.balanceOf(sender).call()
                initial_borrow_token_balance = initial_borrow_token_balance_wei / (10 ** borrow_token_decimals)
                initial_borrow_token_balance = float(f"{ceil_to_8_decimal(initial_borrow_token_balance)}")
                
                logger.info(f"Initial wallet balances - {silo_deposit_token}: {initial_deposit_token_balance}, {borrow_token}: {initial_borrow_token_balance}")
                
            except Exception as e:
                logger.error(f"Error getting silo information: {str(e)}")
                return json.dumps({"error": f"Failed to get silo information: {str(e)}"})
            
            # Check initial positions
            try:
                # Check initial borrow balance
                initial_borrow_wei = silo_lens_contract.functions.debtBalanceOfUnderlying(
                    borrow_silo_address, sender
                ).call()
                initial_borrow = initial_borrow_wei / (10 ** borrow_token_decimals)
                # Round to avoid precision issues
                initial_borrow = float(f"{ceil_to_8_decimal(initial_borrow)}")
                
                # Check initial collateral balance
                initial_collateral_wei = deposit_silo_contract.functions.maxWithdraw(sender).call()
                initial_collateral = initial_collateral_wei / (10 ** deposit_token_decimals)
                # Round to avoid precision issues
                initial_collateral = float(f"{ceil_to_8_decimal(initial_collateral)}")
                
                logger.info(f"Initial collateral: {initial_collateral} {silo_deposit_token}")
                logger.info(f"Initial borrow: {initial_borrow} {borrow_token}")
                
                # If no positions, exit early
                if initial_collateral <= 0.001 and initial_borrow <= 0.001:
                    return json.dumps({
                        "status": "success",
                        "message": "No positions to unwind. Strategy already exited.",
                        "deposit_token": silo_deposit_token,
                        "borrow_token": borrow_token
                    })
                
            except Exception as e:
                logger.error(f"Error checking initial positions: {str(e)}")
                return json.dumps({"error": f"Failed to check initial positions: {str(e)}"})
            
            # NEW EXIT STRATEGY:
            # 1. Withdraw maximum available collateral
            # 2. Swap to borrow token
            # 3. Repay borrow
            # 4. Repeat until positions are empty
            
            unwinding_steps = []
            iteration = 1
            
            while True:
                step_result = {
                    "iteration": iteration,
                    "status": "pending",
                    "actions": []
                }
                
                try:
                    # Step 1: Check how much we can withdraw
                    max_withdraw_wei = deposit_silo_contract.functions.maxWithdraw(sender).call()
                    max_withdraw = max_withdraw_wei / (10 ** deposit_token_decimals)
                    # Round to avoid precision issues
                    max_withdraw = float(f"{ceil_to_8_decimal(max_withdraw)}")
                    
                    # Check how much is still borrowed
                    current_borrow_wei = silo_lens_contract.functions.debtBalanceOfUnderlying(
                        borrow_silo_address, sender
                    ).call()
                    current_borrow = current_borrow_wei / (10 ** borrow_token_decimals)
                    # Round to avoid precision issues
                    current_borrow = float(f"{ceil_to_8_decimal(current_borrow)}")
                    
                    logger.info(f"Iteration {iteration}: Max withdrawable: {max_withdraw} {silo_deposit_token}")
                    logger.info(f"Iteration {iteration}: Current borrow: {current_borrow} {borrow_token}")
                    
                    # Check if we're done
                    if max_withdraw <= 0.001 and current_borrow <= 0.001:
                        logger.info("Positions fully unwound!")
                        break
                    
                    # If nothing to withdraw but still have borrow, we need to repay from wallet
                    if max_withdraw <= 0.001 and current_borrow > 0.001:
                        # Check if we have enough in wallet to repay
                        borrow_token_balance_wei = borrow_token_contract.functions.balanceOf(sender).call()
                        borrow_token_balance = borrow_token_balance_wei / (10 ** borrow_token_decimals)
                        
                        if borrow_token_balance >= current_borrow:
                            logger.info(f"No collateral left but can repay {current_borrow} {borrow_token} from wallet")
                            
                            # Repay from wallet
                            repay_result = SiloRepayToolPrivy(self._agent, self._llm)._run(
                                token_0=borrow_token,
                                token_1=silo_deposit_token,
                                amount=current_borrow,
                                sender=sender,
                                id=id
                            )
                            
                            if isinstance(repay_result, str):
                                try:
                                    repay_result = json.loads(repay_result)
                                except:
                                    pass
                            
                            if "error" in repay_result:
                                logger.warning(f"Failed to repay: {repay_result['error']}")
                                step_result["status"] = "failed"
                                step_result["error"] = f"Repay failed: {repay_result['error']}"
                            else:
                                logger.info(f"Repaid {current_borrow} {borrow_token}")
                                step_result["actions"].append({
                                    "type": "repay_from_wallet",
                                    "amount": current_borrow,
                                    "tx_hash": repay_result.get("tx_hash")
                                })
                                step_result["status"] = "success"
                        else:
                            logger.warning(f"No collateral left and insufficient {borrow_token} in wallet to repay remaining debt")
                            step_result["status"] = "failed"
                            step_result["error"] = f"No collateral left and insufficient {borrow_token} in wallet"
                        
                        unwinding_steps.append(step_result)
                        break
                    
                    # Step 2: Withdraw available collateral
                    withdraw_amount = max_withdraw
                    # Round to avoid precision issues
                    withdraw_amount = float(f"{ceil_to_8_decimal(withdraw_amount)}")
                    
                    if withdraw_amount <= 0.001:
                        logger.warning(f"Withdraw amount too small: {withdraw_amount}")
                        break
                    
                    logger.info(f"Withdrawing {withdraw_amount} {silo_deposit_token}")
                    withdraw_result = SiloWithdrawToolPrivy(self._agent, self._llm)._run(
                        token_0=silo_deposit_token,
                        token_1=borrow_token,
                        amount=withdraw_amount,
                        collateral_type=1,  # Collateral
                        sender=sender,
                        id=id
                    )
                    
                    if isinstance(withdraw_result, str):
                        try:
                            withdraw_result = json.loads(withdraw_result)
                        except:
                            pass
                    
                    if "error" in withdraw_result:
                        logger.warning(f"Failed to withdraw: {withdraw_result['error']}")
                        step_result["status"] = "failed"
                        step_result["error"] = f"Withdrawal failed: {withdraw_result['error']}"
                        unwinding_steps.append(step_result)
                        break
                    
                    logger.info(f"Withdrawal successful: {withdraw_result.get('tx_hash')}")
                    step_result["actions"].append({
                        "type": "withdrawal",
                        "amount": withdraw_amount,
                        "tx_hash": withdraw_result.get("tx_hash")
                    })
                    
                    # Step 3: Swap withdrawn collateral to borrow token
                    deposit_token_balance_wei = deposit_token_contract.functions.balanceOf(sender).call()
                    deposit_token_balance = deposit_token_balance_wei / (10 ** deposit_token_decimals)
                    # Round to avoid precision issues
                    deposit_token_balance = float(f"{ceil_to_8_decimal(deposit_token_balance)}")
                    
                    swap_amount = min(withdraw_amount, deposit_token_balance)  # Use 98% of balance
                    # Round to avoid precision issues
                    swap_amount = float(f"{ceil_to_8_decimal(swap_amount)}")
                    
                    if swap_amount <= 0.001:
                        logger.warning(f"Swap amount too small: {swap_amount}")
                        step_result["status"] = "partial"
                        step_result["reason"] = "Withdrawal successful but swap amount too small"
                        unwinding_steps.append(step_result)
                        break
                    
                    logger.info(f"Swapping {swap_amount} {silo_deposit_token} for {borrow_token}")
                    
                    # Check borrow token balance before swap
                    borrow_token_balance_before_wei = borrow_token_contract.functions.balanceOf(sender).call()
                    borrow_token_balance_before = borrow_token_balance_before_wei / (10 ** borrow_token_decimals)
                    # Round to avoid precision issues
                    borrow_token_balance_before = float(f"{ceil_to_8_decimal(borrow_token_balance_before)}")
                    logger.info(f"Balance before swap: {borrow_token_balance_before} {borrow_token}")
                    
                    # Use retry mechanism for swap
                    swap_success = False
                    for i in range(10):  # Try up to 3 times with increasing slippage
                        try:
                            current_slippage = swap_slippage + (i * 0.5)  # Increase slippage each retry
                            logger.info(f"Swap attempt {i+1} with slippage {current_slippage}%")
                            
                            swap_result = SonicSwapToolPrivy(self._agent, self._llm)._run(
                                from_token=silo_deposit_token if silo_deposit_token.upper() != "S" else "wS",
                                to_token=borrow_token if borrow_token.upper() != "S" else "wS",
                                amount=swap_amount,
                                sender=sender,
                                slippage=current_slippage
                            )
                            
                            if swap_result and "error" not in swap_result:
                                # Check balance after swap to calculate actual received amount
                                borrow_token_balance_after_wei = borrow_token_contract.functions.balanceOf(sender).call()
                                borrow_token_balance_after = borrow_token_balance_after_wei / (10 ** borrow_token_decimals)
                                
                                # Calculate the actual amount received based on balance difference
                                received_amount = borrow_token_balance_after - borrow_token_balance_before
                                # Round to avoid precision issues
                                received_amount = float(f"{ceil_to_8_decimal(received_amount)}")
                                
                                logger.info(f"Balance after swap: {borrow_token_balance_after} {borrow_token}")
                                logger.info(f"Actual swap output: {received_amount} {borrow_token}")
                                
                                if received_amount > 0.00001:  # Ensure we received a meaningful amount
                                    swap_success = True
                                    logger.info(f"Swap successful on attempt {i+1}: Got {received_amount} {borrow_token}")
                                    
                                    step_result["actions"].append({
                                        "type": "swap",
                                        "amount": swap_amount,
                                        "received": received_amount,
                                        "attempt": i+1,
                                        "slippage": current_slippage,
                                        "tx_hash": swap_result.get("tx")
                                    })
                                    break
                                else:
                                    logger.warning(f"Received amount too small: {received_amount}")
                            else:
                                error_msg = swap_result.get("error", "Unknown error") if swap_result else "Empty swap result"
                                logger.warning(f"Swap attempt {i+1} failed: {error_msg}")
                        except Exception as swap_error:
                            logger.warning(f"Exception during swap attempt {i+1}: {str(swap_error)}")
                    
                    if not swap_success:
                        logger.warning("All swap attempts failed")
                        step_result["status"] = "partial"
                        step_result["error"] = "All swap attempts failed"
                        step_result["reason"] = "Withdrawal successful but swap failed"
                        unwinding_steps.append(step_result)
                        break
                    
                    # Step 4: Repay borrow with swapped tokens
                    # Get current borrow token balance (after swap)
                    borrow_token_balance_wei = borrow_token_contract.functions.balanceOf(sender).call()
                    borrow_token_balance = borrow_token_balance_wei / (10 ** borrow_token_decimals)
                    # Round to avoid precision issues
                    borrow_token_balance = float(f"{ceil_to_8_decimal(borrow_token_balance)}")
                    
                    repay_amount = min(borrow_token_balance - borrow_token_balance_before, current_borrow)  # Repay up to 98% of balance or all debt
                    # Round to avoid precision issues
                    repay_amount = float(f"{ceil_to_8_decimal(repay_amount)}")
                    
                    if repay_amount <= 0.001:
                        logger.warning(f"Repay amount too small: {repay_amount}")
                        step_result["status"] = "partial"
                        step_result["reason"] = "Withdrawal and swap successful but repay amount too small"
                        unwinding_steps.append(step_result)
                        continue  # Continue to next iteration even if we couldn't repay this time
                    
                    logger.info(f"Repaying {repay_amount} {borrow_token}")
                    repay_result = SiloRepayToolPrivy(self._agent, self._llm)._run(
                        token_0=borrow_token,
                        token_1=silo_deposit_token,
                        amount=repay_amount,
                        sender=sender,
                        id=id
                    )
                    
                    if isinstance(repay_result, str):
                        try:
                            repay_result = json.loads(repay_result)
                        except:
                            pass
                    
                    if "error" in repay_result:
                        logger.warning(f"Failed to repay: {repay_result['error']}")
                        step_result["status"] = "partial"
                        step_result["error"] = f"Repay failed: {repay_result['error']}"
                        step_result["reason"] = "Withdrawal and swap successful but repay failed"
                    else:
                        logger.info(f"Repay successful: {repay_result.get('tx_hash')}")
                        step_result["actions"].append({
                            "type": "repay",
                            "amount": repay_amount,
                            "tx_hash": repay_result.get("tx_hash")
                        })
                        step_result["status"] = "success"
                    
                    unwinding_steps.append(step_result)
                    
                    # Increment iteration counter
                    iteration += 1
                    
                    # Check if we should continue
                    if iteration > 10:  # Safety limit
                        logger.warning("Reached maximum number of iterations (10)")
                        break
                    
                except Exception as e:
                    logger.error(f"Error in unwinding iteration {iteration}: {str(e)}")
                    step_result["status"] = "failed"
                    step_result["error"] = str(e)
                    unwinding_steps.append(step_result)
                    break
            
            # Check final state
            try:
                # Get final positions
                final_collateral_wei = deposit_silo_contract.functions.maxWithdraw(sender).call()
                final_collateral = final_collateral_wei / (10 ** deposit_token_decimals)
                # Round to avoid precision issues
                final_collateral = float(f"{ceil_to_8_decimal(final_collateral)}")
                
                final_borrow_wei = silo_lens_contract.functions.debtBalanceOfUnderlying(
                    borrow_silo_address, sender
                ).call()
                final_borrow = final_borrow_wei / (10 ** borrow_token_decimals)
                # Round to avoid precision issues
                final_borrow = float(f"{ceil_to_8_decimal(final_borrow)}")
                
                # Get wallet balances
                deposit_token_balance_wei = deposit_token_contract.functions.balanceOf(sender).call()
                deposit_token_balance = deposit_token_balance_wei / (10 ** deposit_token_decimals)
                # Round to avoid precision issues
                deposit_token_balance = float(f"{ceil_to_8_decimal(deposit_token_balance)}")
                
                borrow_token_balance_wei = borrow_token_contract.functions.balanceOf(sender).call()
                borrow_token_balance = borrow_token_balance_wei / (10 ** borrow_token_decimals)
                # Round to avoid precision issues
                borrow_token_balance = float(f"{ceil_to_8_decimal(borrow_token_balance)}")
                
                logger.info(f"Final positions - Collateral: {final_collateral} {silo_deposit_token}, Borrow: {final_borrow} {borrow_token}")
                logger.info(f"Final wallet balances - {silo_deposit_token}: {deposit_token_balance}, {borrow_token}: {borrow_token_balance}")
                
                # Calculate net change in wallet balances
                net_deposit_token_balance = deposit_token_balance - initial_deposit_token_balance
                net_borrow_token_balance = borrow_token_balance - initial_borrow_token_balance
                
                logger.info(f"Net changes in wallet balances - {silo_deposit_token}: {net_deposit_token_balance}, {borrow_token}: {net_borrow_token_balance}")
                
                final_swaps = []
                # Perform swaps to original deposit token if it's different from silo_deposit_token or borrow_token
                if original_deposit_token.upper() not in [silo_deposit_token.upper(), borrow_token.upper()]:
                    # Swap deposit token gains to original token if we have any
                    if net_deposit_token_balance > 0.001:
                        logger.info(f"Swapping {net_deposit_token_balance} {silo_deposit_token} to {original_deposit_token}")
                        swap_result = SonicSwapToolPrivy(self._agent, self._llm)._run(
                            from_token=silo_deposit_token if silo_deposit_token.upper() != "S" else "wS",
                            to_token=original_deposit_token if original_deposit_token.upper() != "S" else "wS",
                            amount=net_deposit_token_balance,
                            sender=sender,
                            slippage=swap_slippage
                        )
                        
                        if "error" not in swap_result:
                            final_swaps.append({
                                "from_token": silo_deposit_token,
                                "to_token": original_deposit_token,
                                "amount": net_deposit_token_balance,
                                "received": swap_result.get("amount_out", 0),
                                "tx_hash": swap_result.get("tx")
                            })
                    
                    # Swap borrow token gains to original token if we have any
                    if net_borrow_token_balance > 0.001:
                        logger.info(f"Swapping {net_borrow_token_balance} {borrow_token} to {original_deposit_token}")
                        swap_result = SonicSwapToolPrivy(self._agent, self._llm)._run(
                            from_token=borrow_token if borrow_token.upper() != "S" else "wS",
                            to_token=original_deposit_token if original_deposit_token.upper() != "S" else "wS",
                            amount=net_borrow_token_balance,
                            sender=sender,
                            slippage=swap_slippage
                        )
                        
                        if "error" not in swap_result:
                            final_swaps.append({
                                "from_token": borrow_token,
                                "to_token": original_deposit_token,
                                "amount": net_borrow_token_balance,
                                "received": swap_result.get("amount_out", 0),
                                "tx_hash": swap_result.get("tx")
                            })
                
                # If the original deposit token is same as silo_deposit_token
                elif original_deposit_token.upper() == silo_deposit_token.upper():
                    # Just swap borrow token balance if we have any
                    if net_borrow_token_balance > 0.001:
                        logger.info(f"Swapping {net_borrow_token_balance} {borrow_token} to {original_deposit_token}")
                        swap_result = SonicSwapToolPrivy(self._agent, self._llm)._run(
                            from_token=borrow_token if borrow_token.upper() != "S" else "wS",
                            to_token=original_deposit_token if original_deposit_token.upper() != "S" else "wS",
                            amount=net_borrow_token_balance,
                            sender=sender,
                            slippage=swap_slippage
                        )
                        
                        if "error" not in swap_result:
                            final_swaps.append({
                                "from_token": borrow_token,
                                "to_token": original_deposit_token,
                                "amount": net_borrow_token_balance,
                                "received": swap_result.get("amount_out", 0),
                                "tx_hash": swap_result.get("tx")
                            })
                
                # If the original deposit token is same as borrow_token
                elif original_deposit_token.upper() == borrow_token.upper():
                    # Just swap deposit token balance if we have any
                    if net_deposit_token_balance > 0.001:
                        logger.info(f"Swapping {net_deposit_token_balance} {silo_deposit_token} to {original_deposit_token}")
                        swap_result = SonicSwapToolPrivy(self._agent, self._llm)._run(
                            from_token=silo_deposit_token if silo_deposit_token.upper() != "S" else "wS",
                            to_token=original_deposit_token if original_deposit_token.upper() != "S" else "wS",
                            amount=net_deposit_token_balance,
                            sender=sender,
                            slippage=swap_slippage
                        )
                        
                        if "error" not in swap_result:
                            final_swaps.append({
                                "from_token": silo_deposit_token,
                                "to_token": original_deposit_token,
                                "amount": net_deposit_token_balance,
                                "received": swap_result.get("amount_out", 0),
                                "tx_hash": swap_result.get("tx")
                            })

                # claim rewards
                claim_rewards_result = SiloClaimRewardsToolPrivy(self._agent, self._llm)._run(sender)
                
                # Build final result with fixed precision
                result = {
                    "status": "success",
                    "type": "silo_exit_strategy",
                    "token_pair": token_pair,
                    "original_strategy": {
                        "total_borrowed": float(f"{ceil_to_8_decimal(total_borrowed)}"),
                        "total_deposited": float(f"{ceil_to_8_decimal(total_deposited)}")
                    },
                    "unwinding_steps": unwinding_steps,
                    "final_swaps": final_swaps,
                    "iterations_executed": iteration - 1,
                    "initial_state": {
                        "collateral_balance": float(f"{ceil_to_8_decimal(initial_collateral)}"),
                        "borrow_balance": float(f"{ceil_to_8_decimal(initial_borrow)}"),
                        "wallet_balances": {
                            silo_deposit_token: initial_deposit_token_balance,
                            borrow_token: initial_borrow_token_balance
                        }
                    },
                    "final_state": {
                        "collateral_balance": float(f"{ceil_to_8_decimal(final_collateral)}"),
                        "borrow_balance": float(f"{ceil_to_8_decimal(final_borrow)}"),
                        "wallet_balances": {
                            silo_deposit_token: float(f"{ceil_to_8_decimal(deposit_token_balance)}"),
                            borrow_token: float(f"{ceil_to_8_decimal(borrow_token_balance)}")
                        }
                    },
                    "fully_unwound": (final_collateral <= 0.001 and final_borrow <= 0.001),
                    "claim_rewards_tx": claim_rewards_result.get("tx_hash")
                }
                

                # Add original deposit token information
                need_conversion = False
                if original_deposit_token.upper() != silo_deposit_token.upper():
                    # If the original deposit token is S but we have wS
                    if original_deposit_token.upper() == "S" and silo_deposit_token.upper() == "S":
                        result["final_state"]["note"] = "Your original deposit was in S but you received wS. To convert back to S, you would need to use the unwrap function which is not available through this tool."
                        need_conversion = True
                    # If the original deposit token is different from the silo deposit token
                    elif len(final_swaps) == 0:  # Only add this note if we couldn't swap back automatically
                        result["final_state"]["note"] = f"Your original deposit was in {original_deposit_token} but you received {silo_deposit_token}. To convert back, you would need to swap."
                        need_conversion = True
                
                # Add token conversion details
                result["token_info"] = {
                    "original_deposit_token": original_deposit_token,
                    "silo_deposit_token": silo_deposit_token,
                    "borrow_token": borrow_token,
                    "needs_conversion": need_conversion,
                    "auto_converted": len(final_swaps) > 0
                }
                
                return json.dumps(result, indent=2)
                
            except Exception as e:
                logger.error(f"Error checking final state: {str(e)}")
                
                # Return partial result
                return json.dumps({
                    "status": "partial_success",
                    "type": "silo_exit_strategy",
                    "token_pair": token_pair,
                    "original_strategy": {
                        "total_borrowed": total_borrowed,
                        "total_deposited": total_deposited
                    },
                    "unwinding_steps": unwinding_steps,
                    "iterations_executed": iteration - 1,
                    "error": f"Failed to check final state: {str(e)}"
                }, indent=2)
            
        except Exception as e:
            logger.error(f"Error exiting strategy: {str(e)}")
            return json.dumps({"error": f"Failed to exit strategy: {str(e)}"})

# Update the get_privy_silo_tools function to include the new tool
def get_privy_silo_tools(agent, llm) -> list:
    """Return a list of all Privy Silo-related tools."""
    return [
        SiloDepositToolPrivy(agent, llm),
        SiloBorrowToolPrivy(agent, llm),
        SiloRepayToolPrivy(agent, llm),
        SiloWithdrawToolPrivy(agent, llm),
        SiloClaimRewardsToolPrivy(agent, llm),
        SiloLoopingStrategyToolPrivy(agent, llm),
        SiloExitStrategyToolPrivy(agent, llm)
    ]
