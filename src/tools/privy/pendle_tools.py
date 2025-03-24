import json
import logging
from web3 import Web3
from src.tools.pendle_tools import PendleSwapTool
from src.constants.abi import ERC20_ABI
from .helpers import approve_token, execute

logger = logging.getLogger("tools.privy.pendle_tools")

class PendleSwapToolPrivy(PendleSwapTool):
    
    def _run(self, market_symbol: str, token_in_symbol: str, token_out_symbol: str, 
             amount_in: str, user_address: str, token_in_type: str = None, 
             token_out_type: str = None, slippage: float = 0.01) -> str:
        try:
            # Call the parent method to get transaction data
            response = super()._run(
                market_symbol=market_symbol,
                token_in_symbol=token_in_symbol,
                token_out_symbol=token_out_symbol,
                amount_in=amount_in,
                user_address=user_address,
                token_in_type=token_in_type,
                token_out_type=token_out_type,
                slippage=slippage
            )
            # Parse response from the parent method
            if isinstance(response, str):
                try:
                    response = json.loads(response)
                except:
                    return {"error": f"Invalid response from parent method: {response}"}
            
            if "error" in response:
                return {"error": response["error"]}
            
            # Initialize web3 connection
            w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
            
            # Extract transaction data
            tx_data = response.get("transaction", {})
            if not tx_data or "to" not in tx_data or "data" not in tx_data:
                return {"error": "Missing transaction data in parent response"}
            
            to_address = tx_data.get("to")
            data = tx_data.get("data")
            value = tx_data.get("value", 0)
            
            # Handle token approval if needed
            approval_result = None
            
            # If this is not a native token swap (not using S as input), we may need approval
            if token_in_symbol.upper() != "S" and "token_address" in response and response["token_address"]:
                token_address = Web3.to_checksum_address(response["token_address"])
                amount_wei = int(float(response.get("amount_in", 0)))
                token_contract = w3.eth.contract(address=token_address, abi=ERC20_ABI)
                
                # Check current allowance
                current_allowance = token_contract.functions.allowance(
                    user_address, to_address
                ).call()
                
                if current_allowance < amount_wei:
                    logger.info(f"Approving {token_in_symbol} for Pendle swap. Current allowance: {current_allowance}, Required: {amount_wei}")
                    
                    # Execute approval transaction
                    approval_result = approve_token(
                        self._agent.connection_manager.connections["privy"],
                        token_address, to_address, amount_wei, user_address
                    )
                    
                    if "error" in approval_result:
                        return {"error": approval_result["error"]}
                    
                    if not approval_result.get("success", False):
                        return {"error": "Token approval transaction failed", "tx_hash": approval_result.get("tx_hash")}
                    
                    logger.info(f"Token approval successful: {approval_result.get('tx_hash')}")
            
            # Execute the swap transaction
            auth_message = f"I authorize swapping {amount_in} {token_in_symbol} for {token_out_symbol} on Pendle"
            swap_result = execute(
                self._agent.connection_manager.connections["privy"],
                user_address,
                to_address,
                data,
                hex(int(value or 0  )),
                "pendle_swap",
                auth_message
            )
            
            if "error" in swap_result:
                return {"error": swap_result["error"]}
            
            # Format the result
            result = {
                "status": "success",
                "type": "pendle_swap",
                "market": market_symbol,
                "from_token": token_in_symbol,
                "to_token": token_out_symbol,
                "amount_in": amount_in,
                "amount_out": response.get("amount_out", 0),
                "slippage": slippage,
                "price_impact": response.get("price_impact", "unknown"),
                "tx_hash": swap_result.get("tx_hash"),
                "receipt": swap_result.get("receipt"),
                "success": swap_result.get("success", False),
                "gas_stats": swap_result.get("gas_stats")
            }
            
            if approval_result:
                result["approval"] = {
                    "tx_hash": approval_result.get("tx_hash"),
                    "success": approval_result.get("success", False),
                    "gas_stats": approval_result.get("gas_stats")
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Privy Pendle swap: {str(e)}")
            return {"error": f"Swap failed: {str(e)}"}

def get_privy_pendle_tools(agent, llm) -> list:
    """Return a list of all Privy Pendle-related tools."""
    return [
        PendleSwapToolPrivy(agent, llm),
        # Add other Privy Pendle tools here as they are implemented
    ]
