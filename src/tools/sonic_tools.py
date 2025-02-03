from langchain.tools import BaseTool
import json
import logging
from src.action_handler import execute_action

logger = logging.getLogger("tools.sonic_tools")

class SonicTokenLookupTool(BaseTool):
    name: str = "sonic_token_lookup"
    description: str = """
    sonic_token_lookup: Look up token addresses by ticker symbol
    Example: For "What's the address for BTC?", use: "BTC"
    Use this tool to look up token addresses by their ticker symbol.
    Input should be a ticker symbol (e.g., "BTC", "ETH").
    Returns the token address if found, null if not found.
    """
    
    def __init__(self):
        super().__init__()
    
    def _run(self, ticker: str) -> str:
        try:
            # result = execute_action(
            #     action_name="get-token-by-ticker",
            #     kwargs={"ticker": ticker}
            # )
            logger.info("SonicTokenLookupTool: Looking up token...")
            return json.dumps({"ticker": ticker, "address": "address"})
        except Exception as e:
            logger.error(f"Failed to lookup token: {str(e)}")
            return json.dumps({"error": str(e)})

class SonicBalanceCheckTool(BaseTool):
    name: str = "sonic_balance_check"
    description: str = """
    sonic_balance_check: Check token balances
    Example: For "Check my S balance", use: {"address": null, "token_address": null}
    Example: For "Check my BTC balance at 0x123", use: {"address": "0x123", "token_address": "0xBTC_ADDRESS"}
    Check balance of $S or other tokens on Sonic.
    Input should be a JSON string with optional 'address' and 'token_address' fields.
    If no address is provided, checks the default wallet balance.
    Use sonic_token_lookup for token addresses lookup and pass the ticker like, eg- "Check my S balance" - S is the ticker
    """
    
    def __init__(self):
        super().__init__()
    
    def _run(self, **kwargs) -> str:
        try:
            # result = execute_action(
            #     action_name="get-sonic-balance",
            #     kwargs=kwargs
            # )
            logger.info("SonicBalanceCheckTool: Checking balance...")
            return json.dumps({
                "balance": "balance",
                "address": kwargs.get("address", "default"),
                "token_address": kwargs.get("token_address", "S")
            })
        except Exception as e:
            logger.error(f"Failed to check balance: {str(e)}")
            return json.dumps({"error": str(e)})

class SonicTokenTransferTool(BaseTool):
    name: str = "sonic_token_transfer"
    description: str = """
    sonic_token_transfer: Transfer tokens
    Example: For "Send 100 S to 0x456", use: {"to_address": "0x456", "amount": 100}
    Example: For "Send 50 BTC to 0x789", use: {"to_address": "0x789", "amount": 50, "token_address": "0xBTC_ADDRESS"}
    Transfer $S or other tokens on Sonic.
    Input should be a JSON string with:
    - to_address: recipient address
    - amount: amount to send
    - token_address: (optional) token address (if not $S)
    
    Check the balance before transfer.
    Use sonic_token_lookup for token addresses lookup and pass the ticker like, eg- "Check my S balance" - S is the ticker
    """
    
    def __init__(self):
        super().__init__()
    
    def _run(self, **kwargs) -> str:
        try:
            action_name = "send-sonic" if "token_address" not in kwargs else "send-sonic-token"
            
            # result = execute_action(
            #     action_name=action_name,
            #     kwargs=kwargs
            # )
            logger.info("SonicTokenTransferTool: Transferring tokens...")
            return json.dumps({
                "status": "success",
                "transaction_url": "uri",
                "details": kwargs
            })
        except Exception as e:
            logger.error(f"Failed to transfer tokens: {str(e)}")
            return json.dumps({"error": str(e)})

class SonicSwapTool(BaseTool):
    name: str = "sonic_swap"
    description: str = """
    sonic_swap: Swap tokens
    Example: For "Swap 100 S to BTC", use: {
        "token_in": "0xS_ADDRESS",
        "token_out": "0xBTC_ADDRESS",
        "amount": 100,
        "slippage": 0.5
    }
    Swap tokens on Sonic DEX.
    Input should be a JSON string with:
    - token_in: input token address
    - token_out: output token address
    - amount: amount to swap
    - slippage: (optional) slippage tolerance (default 0.5%)
    
    Check the balance before swap.
    Use sonic_token_lookup for token addresses lookup and pass the ticker like, eg- "Check my S balance" - S is the ticker.
    """
    
    def __init__(self):
        super().__init__()
    
    def _run(self, **kwargs) -> str:
        try:
            # result = execute_action(
            #     action_name="swap-sonic",
            #     kwargs=kwargs
            # )
            logger.info("SonicSwapTool: Swapping tokens...")
            return json.dumps({
                "status": "success",
                "transaction_url": "uri",
                "details": kwargs
            })
        except Exception as e:
            logger.error(f"Failed to swap tokens: {str(e)}")
            return json.dumps({"error": str(e)})

# Helper function to initialize all Sonic tools
def get_sonic_tools() -> list:
    """
    Initialize and return all Sonic tools
    
    Args:
        agent: The agent instance to use for tool execution
        
    Returns:
        list: List of initialized Sonic tools
    """
    return [
        SonicTokenLookupTool(),
        SonicBalanceCheckTool(),
        SonicTokenTransferTool(),
        SonicSwapTool()
    ]
