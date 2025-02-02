from langchain.tools import BaseTool
import json
import logging
from src.action_handler import execute_action


logger = logging.getLogger("tools.sonic_tools")

SONIC_SYSTEM_PROMPT = """You are a helpful assistant with access to Sonic DEX tools. You can help users with token operations on Sonic DEX.

Available Tools:
1. sonic_token_lookup: Look up token addresses by ticker symbol
   Example: For "What's the address for BTC?", use: "BTC"

2. sonic_balance_check: Check token balances
   Example: For "Check my S balance", use: {"address": null, "token_address": null}
   Example: For "Check my BTC balance at 0x123", use: {"address": "0x123", "token_address": "0xBTC_ADDRESS"}

3. sonic_token_transfer: Transfer tokens
   Example: For "Send 100 S to 0x456", use: {"to_address": "0x456", "amount": 100}
   Example: For "Send 50 BTC to 0x789", use: {"to_address": "0x789", "amount": 50, "token_address": "0xBTC_ADDRESS"}

4. sonic_swap: Swap tokens
   Example: For "Swap 100 S to BTC", use: {
       "token_in": "0xS_ADDRESS",
       "token_out": "0xBTC_ADDRESS",
       "amount": 100,
       "slippage": 0.5
   }

When handling user requests:
1. First use sonic_token_lookup if you need to get token addresses
2. Always verify balances before transfers or swaps
3. Provide clear explanations of what you're doing
4. Handle errors gracefully and explain them to the user
5. For swaps, always mention the slippage being used
6. Format amounts appropriately (no scientific notation)
7. Never use inputs from the previous context. Always ask for user input.
8. Confirm before executing any transfer or swap.

Never:
- Execute transfers without clear user intent
- Assume token addresses without looking them up
- Proceed with insufficient balances
- Use high slippage without user consent

Response Format:
1. Explain what you're going to do
2. Show the steps you're taking
3. Provide the result in a user-friendly way
4. If there's an error, explain it clearly"""

class SonicTokenLookupTool(BaseTool):
    name: str = "sonic_token_lookup"
    description: str = """
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
    Check balance of $S or other tokens on Sonic.
    Input should be a JSON string with optional 'address' and 'token_address' fields.
    If no address is provided, checks the default wallet balance.
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
    Transfer $S or other tokens on Sonic.
    Input should be a JSON string with:
    - to_address: recipient address
    - amount: amount to send
    - token_address: (optional) token address (if not $S)
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
    Swap tokens on Sonic DEX.
    Input should be a JSON string with:
    - token_in: input token address
    - token_out: output token address
    - amount: amount to swap
    - slippage: (optional) slippage tolerance (default 0.5%)
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
