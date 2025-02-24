from langchain.tools import BaseTool
import json
import logging
from src.action_handler import execute_action
from .price_tools import GetTokenPriceTool

logger = logging.getLogger("tools.sonic_tools")

class SonicTokenLookupTool(BaseTool):
    name: str = "sonic_token_lookup"
    description: str = """
    sonic_token_lookup: Get token address
    Input should be a token symbol (e.g. "BTC", "ETH")
    Returns the token's contract address
    """

    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def _run(self, token: str) -> str:
        try:
            logger.info(f"Looking up token address for: {token}")
            
            if token.upper() == "S":
                return json.dumps({"address": None})

            response = execute_action(
                agent=self._agent,
                action_name="get-token-by-ticker",
                ticker=token
            )

            if not response:
                return json.dumps({"error": f"Token {token} not found"})

            return json.dumps({"address": response})

        except Exception as e:
            logger.error(f"Token lookup failed: {str(e)}")
            return json.dumps({"error": str(e)})

class SonicBalanceCheckTool(BaseTool):
    name: str = "sonic_balance_check"
    description: str = """
    sonic_balance_check: Check token balance
    Example: For "Check S balance", use: {"token": "S"}
    Also use before transfer/send/swaps to check balance
    Check balance of any token on Sonic. Input should be a JSON string with:
    - address: sender address
    - token: token symbol (e.g. "S", "BTC", "ETH")
    """

    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def _run(self, address: str, token: str = "S") -> str:
        # TODO ADD SEPARATE CHECK FOR USDC
        try:
            logger.info(f"Checking balance for token: {token}")
            
            balance_params = {"address": address}
            
            action_name = "get-sonic-balance"
            if token.upper() == "S":
                balance_params["token_address"] = None
            else:
                token_lookup_result = json.loads(SonicTokenLookupTool(self._agent)._run(token))
                if "error" in token_lookup_result:
                    return json.dumps({"error": f"Invalid token {token}"})
                balance_params["token_address"] = token_lookup_result["address"]

            response = execute_action(
                agent=self._agent,
                action_name=action_name,
                **balance_params
            )

            if not response:
                if response != 0:
                    return json.dumps({"error": f"Could not fetch balance for {token}"})

            return json.dumps({
                "status": "success",
                "balance": str(response),
                "token": token,
                "details": balance_params
            })

        except Exception as e:
            logger.error(f"Balance check failed: {str(e)}")
            return json.dumps({
                "error": str(e),
                "parameters": {
                    "token": token
                }
            })

class SonicTokenTransferTool(BaseTool):
    name: str = "sonic_token_transfer"
    description: str = """
    sonic_token_transfer: Transfer tokens
    Example: For "Send/Transfer/Execute/Transact 100 S to 0x456", use: {"to_address": "0x456", "amount": 100, "token": "S"}
    Transfer any token on Sonic. Input should be a JSON string with:
    - from_address: sender address
    - to_address: recipient address
    - amount: amount to send
    - token: token symbol (e.g. "S", "BTC", "ETH")
    Transfer only if "from_address" has enough balance
    """
    
    def __init__(self, agent, llm):
        super().__init__()
        self._agent = agent
        self._llm = llm
    
    def _run(self, from_address: str, to_address: str, amount: float, token: str = "S") -> str:
        if not all([from_address, to_address, amount, token]):
            return json.dumps({
                "error": "Missing required parameters"
            })

        transfer_params = {
            "from_address": from_address,
            "to_address": to_address,
            "amount": float(amount)
        }

        if token.upper() == "S":
            transfer_params["token_address"] = None
            action_name = "send-sonic"
        else:
            token_lookup_result = json.loads(SonicTokenLookupTool(self._agent)._run(token))
            if "error" in token_lookup_result:
                return json.dumps({"error": f"Invalid token {token}"})
            transfer_params["token_address"] = token_lookup_result["address"]
            action_name = "send-sonic-token"

        logger.info(f"Transferring {amount} {token} to {to_address}")
        
        response = execute_action(
            agent=self._agent,
            action_name=action_name,
            **transfer_params
        )
        response["type"] = "transfer"
        tx_url = self._agent.connection_manager.connections[self._llm].interrupt_chat(
            query=json.dumps(response)
        )

        return json.dumps({
            "status": "success",
            "type": "transfer",
            "tx": tx_url,
            "details": transfer_params,
        })

class SonicSwapTool(BaseTool):
    name: str = "sonic_swap"
    description: str = """
    sonic_swap: Swap tokens
    Example: For "Swap 100 Sonic to BTC", use: {"from_token": "S", "to_token": "BTC", "amount": 100}
    "Swap 1 S to Anon with 5% Slippage", use: {"from_token": "S", "to_token": "Anon", "amount": 1, "slippage": 5.0}
    Swap between any tokens on Sonic. Input should be a JSON string with:
    - from_address: sender address
    - from_token: token to swap from
    - to_token: token to swap to
    - amount: amount to swap
    - slippage: slippage tolerance (optional) Default - 0.5
    """

    def __init__(self, agent, llm):
        super().__init__()
        self._agent = agent
        self._llm = llm

    def _run(self, from_address: str, from_token: str, to_token: str, amount: float, slippage: float = 0.5) -> str:
        if not all([from_address, from_token, to_token, amount]):
            return json.dumps({
                "error": "Missing required parameters"
            })
        swap_params = {"amount": float(amount), "sender": from_address, "slippage": 0.5 if not slippage else float(slippage)}

        # Handle from_token
        if from_token.upper() == "S":
            swap_params["token_in"] = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"
        else:
            from_token_lookup = json.loads(SonicTokenLookupTool(self._agent)._run(from_token))
            if "error" in from_token_lookup:
                return json.dumps({"error": f"Invalid from_token {from_token}"})
            swap_params["token_in"] = from_token_lookup["address"]

        # Handle to_token
        if to_token.upper() == "S":
            swap_params["token_out"] = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"
        else:
            to_token_lookup = json.loads(SonicTokenLookupTool(self._agent)._run(to_token))
            if "error" in to_token_lookup:
                return json.dumps({"error": f"Invalid to_token {to_token}"})
            swap_params["token_out"] = to_token_lookup["address"]

        logger.info(f"Swapping {amount} {from_token} to {to_token}")
        
        route_summary = execute_action(
            agent=self._agent,
            action_name="get-swap-summary",
            **swap_params
        )
        try :
            response = execute_action(
                agent=self._agent,
                action_name="swap-sonic",
                **swap_params
            )
            if response == None:
                raise Exception("Swap failed. Return amount is too low, please try again with higher slippage")
            route_summary["amountIn"] = str(int(route_summary["amountIn"]) / (10 ** 18))
            route_summary["amountOut"] = str(int(route_summary["amountOut"]) / (10 ** 18))
            output = {
                "approve": route_summary,
                "swap": response
            }
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "details": swap_params
            })

            
        tx_url = self._agent.connection_manager.connections[self._llm].interrupt_chat(
            query=json.dumps(output)
        )
        
        return json.dumps({
            "status": "success",
            "type": "swap",
            "tx": tx_url,
            "details": swap_params
        })
def get_sonic_tools(agent, llm) -> list:
    """Return a list of all Sonic-related tools."""
    return [
        SonicTokenLookupTool(agent),
        SonicBalanceCheckTool(agent),
        SonicTokenTransferTool(agent, llm),
        SonicSwapTool(agent, llm),
        GetTokenPriceTool()
    ]
