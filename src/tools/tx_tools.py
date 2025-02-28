from langchain.tools import BaseTool
import json
import logging
import traceback
from typing import Dict, List, Optional, Any
from requests.exceptions import RequestException, Timeout, ConnectionError

logger = logging.getLogger("tools.tx_tools")

class GetTransactionsTool(BaseTool):
    name: str = "get_transactions"
    description: str = """
    get_transactions: Get list of transactions for a user address
    
    Natural language examples:
    - "Show me transactions for address 0x1234..."
    - "What are the recent transactions for my wallet?"
    - "Get the last 20 transactions for this address"
    
    Input should be a JSON string with:
    - userAddress: User wallet address - Connected Wallet (required)
    - offset: Number of transactions to return (optional, default: 10)
    
    Example: {"userAddress": "0x1234...", "offset": 20}
    """

    def __init__(self, agent):
        super().__init__()
        self._agent = agent
        
    def _run(self, userAddress: str, offset: int = 10) -> str:
        try:
            logger.info(f"Getting transactions for address {userAddress}")
            logger.info(offset)
            
            if not userAddress:
                return json.dumps({"error": "User address cannot be empty", "status": "error"})
                
            if "tx" not in self._agent.connection_manager.connections:
                return json.dumps({"error": "Transaction service connection not configured", "status": "error"})
            
            # Use fixed values for most parameters as per requirements
            response = self._agent.connection_manager.connections["tx"].get_tx_list(
                address=userAddress,
                startblock=0,
                endblock=99999999,
                page=1,
                offset=int(offset),
                sort="asc"
            )
            
            # Process and format the transaction data for better readability
            if response.get("status") == "1":
                transactions = response.get("result", [])
                formatted_txs = []
                
                for tx in transactions:
                    formatted_tx = {
                        "hash": tx.get("hash"),
                        "blockNumber": tx.get("blockNumber"),
                        "timeStamp": tx.get("timeStamp"),
                        "from": tx.get("from"),
                        "to": tx.get("to"),
                        "value": self._format_value(tx.get("value")),
                        "gasPrice": self._format_value(tx.get("gasPrice"), 9),
                        "isError": tx.get("isError") == "1",
                        "txreceipt_status": tx.get("txreceipt_status"),
                        "functionName": tx.get("functionName", "").split("(")[0],
                    }
                    formatted_txs.append(formatted_tx)
                    
                return json.dumps({
                    "status": "success",
                    "data": formatted_txs,
                    "count": len(formatted_txs),
                    "message": f"Found {len(formatted_txs)} transactions"
                })
            else:
                return json.dumps({
                    "status": "success",
                    "data": [],
                    "count": 0,
                    "message": response.get("message", "No transactions found")
                })
                
        except RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return json.dumps({"error": f"API request failed: {str(e)}", "status": "error"})
        except Timeout:
            logger.error("Request timed out")
            return json.dumps({"error": "Request timed out, please try again later", "status": "error"})
        except ConnectionError:
            logger.error("Connection error")
            return json.dumps({"error": "Unable to connect to transaction service", "status": "error"})
        except Exception as e:
            logger.error(f"Failed to get transactions: {str(e)}")
            logger.debug(traceback.format_exc())
            return json.dumps({"error": str(e), "status": "error"})
    
    def _format_value(self, value_str: str, decimals: int = 18) -> str:
        """Format blockchain values to human-readable format"""
        if not value_str or value_str == "0":
            return "0"
        try:
            value = int(value_str) / (10 ** decimals)
            if value < 0.000001 and value > 0:
                return f"{value:.8f}"
            return f"{value:.6f}"
        except (ValueError, TypeError):
            return value_str

def get_tx_tools(agent) -> List[BaseTool]:
    """Return a list of all transaction-related tools."""
    return [
        GetTransactionsTool(agent)
    ]
