import json
import logging
from web3 import Web3
from src.tools.sonic_tools import SonicTokenLookupTool, SonicSwapTool, SonicWrapTool
from src.constants.abi import ERC20_ABI
from src.action_handler import execute_action
from .helpers import approve_token, execute

logger = logging.getLogger("tools.privy.sonic_tools")

class SonicSwapToolPrivy(SonicSwapTool):
    name: str = "privy_sonic_swap_kyber"
    description: str = """
    privy_sonic_swap_kyber: Swap tokens Using KyberSwap via Privy wallet
    Example: For "Swap 100 Sonic to BTC", use: {"from_token": "S", "to_token": "BTC", "amount": 100, "sender": "0xYourWalletAddress"}
    "Swap 1 S to Anon with 5% Slippage", use: {"from_token": "S", "to_token": "Anon", "amount": 1, "slippage": 5.0, "sender": "0xYourWalletAddress"}
    Swap between any tokens on Sonic using your Privy wallet. Input should be a JSON string with:
    - from_token: token to swap from
    - to_token: token to swap to
    - amount: amount to swap
    - sender: Your wallet address
    - slippage: slippage tolerance (optional) Default - 0.5
    """
    
    def _run(self, from_token: str, to_token: str, amount: float, sender: str, slippage: float = 0.5):
        try:
            swap_params = {"amount": float(amount), "sender": sender, "slippage": float(slippage)}

            # Handle from_token
            if from_token.upper() == "S":
                swap_params["token_in"] = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"
            else:
                from_token_lookup = json.loads(SonicTokenLookupTool(self._agent)._run(from_token))
                if "error" in from_token_lookup:
                    return ({"error": f"Invalid from_token {from_token}"})
                swap_params["token_in"] = from_token_lookup["address"]

            # Handle to_token
            if to_token.upper() == "S":
                swap_params["token_out"] = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"
            else:
                to_token_lookup = json.loads(SonicTokenLookupTool(self._agent)._run(to_token))
                if "error" in to_token_lookup:
                    return ({"error": f"Invalid to_token {to_token}"})
                swap_params["token_out"] = to_token_lookup["address"]
            
            logger.info(f"Getting swap summary for {amount} {from_token} to {to_token}")
            
            # First get the route summary for information
            route_summary = execute_action(
                agent=self._agent,
                action_name="get-swap-summary",
                **swap_params
            )
            
            if "amountIn" not in route_summary or "amountOut" not in route_summary:
                return ({"error": "Could not get valid swap quote"})
            
            router_address = route_summary.get("routerAddress")
            if not router_address:
                return ({"error": "Missing router address in swap summary"})
                
            # Initialize web3 connection
            w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
            
            # Format amounts for approval
            amount_in_wei = int(route_summary["amountIn"])
            
            # Handle approval if not native token (Sonic)
            approval_result = None
            if from_token.upper() != "S":
                # Check if token needs approval
                token_contract = w3.eth.contract(address=swap_params["token_in"], abi=ERC20_ABI)
                current_allowance = token_contract.functions.allowance(sender, router_address).call()
                
                if current_allowance < amount_in_wei:
                    logger.info(f"Approving {from_token} for swap. Current allowance: {current_allowance}, Required: {amount_in_wei}")
                    
                    # Execute token approval
                    approval_result = approve_token(
                        self._agent.connection_manager.connections["privy"],
                        swap_params["token_in"], router_address, amount_in_wei, sender
                    )
                    
                    if "error" in approval_result:
                        return ({"error": approval_result["error"]})
                    
                    if not approval_result.get("success", False):
                        return ({"error": "Token approval transaction failed", "tx_hash": approval_result.get("tx_hash")})
                    
                    logger.info(f"Token approval successful: {approval_result.get('tx_hash')}")
            
            # Now get the actual swap transaction data
            try:
                swap_tx = execute_action(
                    agent=self._agent,
                    action_name="swap-sonic",
                    **swap_params
                )
                
                # The swap-sonic action returns the transaction data, not the actual transaction
                if not swap_tx:
                    return ({"error": "Failed to generate swap transaction data"})
                
                # Check if there's an error message
                if isinstance(swap_tx, str):
                    return ({"error": swap_tx})
                
            except Exception as e:
                logger.error(f"Failed to generate swap transaction: {str(e)}")
                return ({"error": f"Failed to generate swap transaction: {str(e)}"})
            
            # Execute the swap transaction with Privy
            auth_message = f"I authorize swapping {amount} {from_token} tokens for {to_token}"
            swap_result = execute(
                self._agent.connection_manager.connections["privy"],
                sender,
                swap_tx["to"],
                swap_tx["data"],
                0 if from_token.upper() != "S" else amount_in_wei,  # Send value only if from_token is S
                "swap",
                auth_message
            )
            
            if "error" in swap_result:
                return ({"error": swap_result["error"]})
            
            # Format route summary for response
            route_summary["amountIn"] = str(int(route_summary["amountIn"]))
            route_summary["amountOut"] = str(int(route_summary["amountOut"]))
            token_contract = w3.eth.contract(address=swap_params["token_out"], abi=ERC20_ABI)
            
            decimals = token_contract.functions.decimals().call()
            # Build result in the same format as the original SonicSwapTool
            result = {
                "status": "success",
                "type": "swap",
                "from_token": from_token,
                "to_token": to_token,
                "amount_in": amount,
                "amount_out": float(route_summary["amountOut"]) / (10 ** int(decimals)),  # Approximate conversion
                "slippage": slippage,
                "tx": swap_result.get("tx_hash"),
                "details": swap_params,
                "approve": route_summary,
                "swap": {
                    "tx_hash": swap_result.get("tx_hash"),
                    "receipt": swap_result.get("receipt"),
                    "success": swap_result.get("success", False),
                    "gas_stats": swap_result.get("gas_stats")
                }
            }
            
            if approval_result:
                result["approval"] = {
                    "tx_hash": approval_result.get("tx_hash"),
                    "success": approval_result.get("success", False),
                    "gas_stats": approval_result.get("gas_stats")
                }
            
            return (result)
            
        except Exception as e:
            logger.error(f"Error in Privy Sonic swap: {str(e)}")
            return ({"error": f"Swap failed: {str(e)}"})

class SonicWrapToolPrivy(SonicWrapTool):
    name: str = "privy_sonic_wrap"
    description: str = """
    privy_sonic_wrap: Wrap tokens via Privy wallet
    Example: For "Wrap 100 S(sonic)", use: {"amount": 100, "sender": "0xYourWalletAddress"}
    - sender: Your wallet address
    - amount: amount to wrap
    """

    def _run(self, amount: float, sender: str) -> str:
        try:
            logger.info(f"Wrapping {amount} S tokens via Privy wallet")
            
            wrap_params = {
                "sender": sender,
                "amount": float(amount)
            }
            
            # Get wrap transaction data
            try:
                wrap_tx = execute_action(
                    agent=self._agent,
                    action_name="wrap-sonic",
                    **wrap_params
                )
                logger.info(wrap_tx)
                if not wrap_tx:
                    return ({"error": "Failed to generate wrap transaction data"})
                
                # Check if there's an error message
                if isinstance(wrap_tx, str):
                    return ({"error": wrap_tx})
                
            except Exception as e:
                logger.error(f"Failed to generate wrap transaction: {str(e)}")
                return ({"error": f"Failed to generate wrap transaction: {str(e)}"})
            
            # Execute the wrap transaction with Privy
            amount_in_wei = int(float(amount) * 10**18)  # Convert to wei
            auth_message = f"I authorize wrapping {amount} S tokens"
            
            wrap_result = execute(
                self._agent.connection_manager.connections["privy"],
                sender,
                wrap_tx["to"],
                "0x",
                amount_in_wei,  # Send value since we're wrapping S
                "wrap",
                auth_message
            )
            
            if "error" in wrap_result:
                return ({"error": wrap_result["error"]})
            
            # Build success result
            result = {
                "status": "success",
                "type": "wrap",
                "amount": amount,
                "tx": wrap_result.get("tx_hash"),
                "details": wrap_params,
                "wrap": {
                    "tx_hash": wrap_result.get("tx_hash"),
                    "receipt": wrap_result.get("receipt"),
                    "success": wrap_result.get("success", False),
                    "gas_stats": wrap_result.get("gas_stats")
                }
            }
            
            return (result)
            
        except Exception as e:
            logger.error(f"Error in Privy Sonic wrap: {str(e)}")
            return ({"error": f"Wrap failed: {str(e)}"})

def get_privy_sonic_tools(agent, llm) -> list:
    """Return a list of all Privy Sonic-related tools."""
    return [
        SonicSwapToolPrivy(agent, llm),
        SonicWrapToolPrivy(agent, llm),
    ]
