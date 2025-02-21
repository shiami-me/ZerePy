from langchain.tools import BaseTool
import json
import logging
import requests

from src.utils.stores.chainid_store import ChainIDStore
from .sonic_tools import SonicTokenLookupTool
from typing import Optional

logger = logging.getLogger("tools.debridge_tools")

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

def get_token_by_ticker(ticker: str, chain: str) -> Optional[str]:
    """Get token address by ticker symbol"""
    try:
        logger.info(chain)
        response = requests.get(
            f"https://api.dexscreener.com/latest/dex/search?q={ticker}"
        )
        response.raise_for_status()

        data = response.json()
        if not data.get('pairs'):
            return None
        pairs = [
            pair for pair in data["pairs"] if chain.lower() in pair.get("chainId")
        ]
        pairs.sort(key=lambda x: x.get("fdv", 0), reverse=True)

        pairs = [
            pair
            for pair in pairs
            if pair.get("baseToken", {}).get("symbol", "").lower() == ticker.lower()
        ]

        if pairs:
            return pairs[0].get("baseToken", {}).get("address")
        return None
    except Exception as e:
        logger.error(f"Error getting token by ticker: {e}")
        return None

class DebridgeBridgeTool(BaseTool):
    name: str = "debridge_tx"
    description: str = """
    debridge_tx: Bridge assets from Sonic to other chains using Debridge
    Example: Bridge 1 Sonic to ETH. Use: {"srcChainTokenIn": "S", "srcChainTokenInAmount": 1, "dstChain": "ETH", "dstChainTokenOut": "ETH", "dstChainTokenOutRecipient": "sender address"}
    Bridge 1 S to USDC on ETH. Use: {"srcChainTokenIn": "S", "srcChainTokenInAmount": 1, "dstChain": "ETH", "dstChainTokenOut": "USDC", "dstChainTokenOutRecipient": "sender address"}
    
    Input should be a JSON string with:
    - srcChainTokenIn: Source chain token symbol (e.g. "S", "BTC", "ETH")
    - srcChainTokenInAmount: Amount of source chain token
    - dstChain: Destination chain (e.g. "ETH", "BSC")
    - dstChainTokenOut: Destination chain token symbol (e.g. "ETH", "USDC")
    - dstChainTokenOutRecipient: Sender Address or specified address
    """

    def __init__(self, agent):
        super().__init__()
        self._agent = agent
        self._chain_id_store = ChainIDStore()

    def _run(self, srcChainTokenIn: str, srcChainTokenInAmount: int, dstChain: str, dstChainTokenOut: str, dstChainTokenOutRecipient: str) -> str:
        try:
            # Get chain IDs
            dstChainId = self._chain_id_store.get_chain_id(dstChain)

            # Check if source token is native
            if self._chain_id_store.is_native_token(dstChain, srcChainTokenIn):
                srcToken = ZERO_ADDRESS
            else:
                srcToken = json.loads(SonicTokenLookupTool(self._agent)._run(srcChainTokenIn))["address"]
                if not srcToken: 
                    srcToken = ZERO_ADDRESS

            # Check if destination token is native
            if self._chain_id_store.is_native_token(dstChain, dstChainTokenOut):
                dstToken = ZERO_ADDRESS
            else:
                dstToken = get_token_by_ticker(dstChainTokenOut, dstChain)

            logger.info(f"Bridging {srcChainTokenInAmount} {srcChainTokenIn} to {dstChainTokenOut} on {dstChain}")
            logger.info(f"Source Token: {srcToken}, Destination Token: {dstToken}, Destination Chain ID: {dstChainId}")

            # Call bridge function
            response = self._agent.connection_manager.connections["debridge"].bridge(
                srcChainTokenIn=srcToken,
                srcChainTokenInAmount=int(srcChainTokenInAmount*(10**18)),
                dstChainId=dstChainId,
                dstChainTokenOut=dstToken,
                dstChainTokenOutAmount="auto",
                dstChainTokenOutRecipient=dstChainTokenOutRecipient
            )
            logger.info(f"Bridge response: {response}")

            if not response:
                return json.dumps({"error": "Failed to bridge assets"})

            # Format response to match ApproveTransaction interface
            estimation = response.get("estimation", {})
            srcChainTokenIn = estimation.get("srcChainTokenIn", {})
            dstChainTokenOut = estimation.get("dstChainTokenOut", {})
            tx = response.get("tx", {})

            approve_transaction = {
                "approve": {
                    "amountIn": srcChainTokenIn.get("amount", "0"),
                    "amountOut": dstChainTokenOut.get("amount", "0"),
                    "amountInUsd": srcChainTokenIn.get("approximateUsdValue", 0),
                    "amountOutUsd": dstChainTokenOut.get("approximateUsdValue", 0),
                    "gas": int(tx.get("gas", "0")),
                    "gasPrice": tx.get("gasPrice", "0"),
                    "gasUsd": 0,
                    "tokenIn": srcChainTokenIn.get("address", ZERO_ADDRESS),
                    "routerAddress": tx.get("to", ZERO_ADDRESS),
                    "chainId": 146,
                    "route": []
                },
                "swap": {
                    "type": "bridge",
                    "to": tx.get("to", ZERO_ADDRESS),
                    "value": tx.get("value", "0"),
                    "chainId": 146,
                    "data": tx.get("data", "0x"),
                    "gas": tx.get("gas", "0"),
                    "maxFeePerGas": tx.get("maxFeePerGas", "0"),
                    "maxPriorityFeePerGas": tx.get("maxPriorityFeePerGas", "0"),
                    "tokenAddress": srcChainTokenIn.get("address", ZERO_ADDRESS),
                    "amount": srcChainTokenIn.get("amount", "0")
                }
            }

            return json.dumps(approve_transaction)

        except Exception as e:
            logger.error(f"Bridge operation failed: {str(e)}")
            return json.dumps({"error": str(e)})

def get_debridge_tools(agent) -> list:
    """Return a list of all Debridge-related tools."""
    return [DebridgeBridgeTool(agent)]
