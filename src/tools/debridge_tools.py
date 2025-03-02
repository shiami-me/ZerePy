from langchain.tools import BaseTool
import json
import logging
import requests

from src.utils.stores.chainid_store import ChainIDStore
from .sonic_tools import SonicTokenLookupTool
from typing import Optional

logger = logging.getLogger("tools.debridge_tools")

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

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

            srcToken = json.loads(SonicTokenLookupTool(self._agent)._run(srcChainTokenIn))["address"]
            if not srcToken: 
                srcToken = ZERO_ADDRESS

            dstToken = self._agent.connection_manager.connections["debridge"].get_token_address(dstChainId, dstChainTokenOut)

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
            if tx["to"]:
                tx = self._agent.connection_manager.connections["sonic"].estimate_gas(tx)
            elif tx["allowanceTarget"]:
                tx["to"] = tx["allowanceTarget"]
            approve_transaction = {
                "approve": {
                    "amountIn": srcChainTokenIn.get("amount", "0") / (10**(srcChainTokenIn.get("decimals", 18))),
                    "amountOut": dstChainTokenOut.get("amount", "0") / (10**(dstChainTokenOut.get("decimals", 18))),
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
            approve_transaction["status"] = "Initiated. Continue in the frontend."

            return json.dumps(approve_transaction)

        except Exception as e:
            logger.error(f"Bridge operation failed: {str(e)}")
            return json.dumps({"error": str(e)})

def get_debridge_tools(agent) -> list:
    """Return a list of all Debridge-related tools."""
    return [DebridgeBridgeTool(agent)]
