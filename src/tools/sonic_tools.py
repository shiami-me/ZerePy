from langchain.tools import BaseTool
import json
import logging
from src.action_handler import execute_action

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
    Check balance of any token on Sonic. Input should be a JSON string with:
    - token: token symbol (e.g. "S", "BTC", "ETH")
    """

    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def _run(self, token: str = "S") -> str:
        try:
            logger.info(f"Checking balance for token: {token}")
            
            balance_params = {}
            
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
    IMPORTANT: Wait for the transfer to complete before requesting transaction data.
    """
    
    def __init__(self, agent):
        super().__init__()
        self._agent = agent
    
    def _run(self, from_address: str, to_address: str, amount: float, token: str = "S") -> str:
        try:
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

            return json.dumps({
                "status": "interrupt",
                "tx": response,
                "details": transfer_params,
                "next_action": "sonic_request_transaction_data"  # Added to guide sequence
            })

        except Exception as e:
            logger.error(f"Transfer failed: {str(e)}")
            return json.dumps({
                "error": str(e),
                "parameters": {
                    "to_address": to_address,
                    "amount": amount,
                    "token": token
                }
            })

class SonicSwapTool(BaseTool):
    name: str = "sonic_swap"
    description: str = """
    sonic_swap: Swap tokens
    Example: For "Swap 100 S to BTC", use: {"from_token": "S", "to_token": "BTC", "amount": 100}
    Swap between any tokens on Sonic. Input should be a JSON string with:
    - from_token: token to swap from
    - to_token: token to swap to
    - amount: amount to swap
    """

    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def _run(self, from_token: str, to_token: str, amount: float) -> str:
        try:
            if not all([from_token, to_token, amount]):
                return json.dumps({
                    "error": "Missing required parameters"
                })

            swap_params = {"amount": float(amount)}

            # Handle from_token
            if from_token.upper() == "S":
                swap_params["from_token_address"] = None
            else:
                from_token_lookup = json.loads(SonicTokenLookupTool(self._agent)._run(from_token))
                if "error" in from_token_lookup:
                    return json.dumps({"error": f"Invalid from_token {from_token}"})
                swap_params["from_token_address"] = from_token_lookup["address"]

            # Handle to_token
            if to_token.upper() == "S":
                swap_params["to_token_address"] = None
            else:
                to_token_lookup = json.loads(SonicTokenLookupTool(self._agent)._run(to_token))
                if "error" in to_token_lookup:
                    return json.dumps({"error": f"Invalid to_token {to_token}"})
                swap_params["to_token_address"] = to_token_lookup["address"]

            logger.info(f"Swapping {amount} {from_token} to {to_token}")
            
            # response = execute_action(
            #     agent=self._agent,
            #     action_name="swap-sonic",
            #     **swap_params
            # )

            return json.dumps({
                "status": "success",
                "transaction_url": "uri",
                "details": swap_params
            })

        except Exception as e:
            logger.error(f"Swap failed: {str(e)}")
            return json.dumps({
                "error": str(e),
                "parameters": {
                    "from_token": from_token,
                    "to_token": to_token,
                    "amount": amount
                }
            })
            
class SonicRequestTransactionDataTool(BaseTool):
    name: str = "sonic_request_transaction_data"
    description: str = """
sonic_request_transaction_data - Executed after sonic transactions are completed
Returns the confirmed transaction data.
Example - 1. Send 100 S to 0x123 2. Swap 100 S to BTC
"""

    def __init__(self, agent, llm):
        super().__init__()
        self._agent = agent
        self._llm = llm

    def _run(self) -> str:
        logger.info("Requesting transaction data")
        tx_data = self._agent.connection_manager.connections[self._llm].interrupt_chat(
            query="Requestion transaction data from the user, return the confirmed transaction data."
        )
        return json.dumps({
            "status": "success",
            "transaction_data": tx_data
        })


def get_sonic_tools(agent, llm) -> list:
    """Return a list of all Sonic-related tools."""
    return [
        SonicTokenLookupTool(agent),
        SonicBalanceCheckTool(agent),
        SonicTokenTransferTool(agent),
        SonicSwapTool(agent),
        SonicRequestTransactionDataTool(agent, llm)
    ]
