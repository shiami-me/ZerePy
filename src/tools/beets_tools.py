from langchain.tools import BaseTool
import json
import logging
import traceback
from requests.exceptions import RequestException, Timeout, ConnectionError
from typing import Optional, Dict, Any, List, Union, Type
from pydantic import BaseModel, Field

logger = logging.getLogger("tools.beets_tools")

class BeetsSwapTool(BaseTool):
    name: str = "beets_swap"
    description: str = """
    beets_swap: Swap tokens using Beets (Balancer)
    
    Natural language examples:
    - "I want to swap 10 ETH for USDC on Beets with 0.5 percent slippage"
    - "Trade my 5 WBTC for DAI using Beets"
    - "Exchange 100 USDC for ETH with Beets Balancer"
    
    Input should be a JSON string with:
    - tokenIn: Token symbol, name or address to swap from (required)
    - tokenOut: Token symbol, name or address to swap to (required)
    - slippage: Slippage tolerance (e.g. 1 for 1%) (required)
    - userAddress: User address (required)
    
    Example: Swap 1 ETH for USDC. Use: {"tokenIn": "ETH", "tokenOut": "USDC", "slippage": 1, "userAddress": "0xUser.."}
    """

    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def _run(self, tokenIn: str, tokenOut: str, slippage: float = 0.005, userAddress: str = "") -> str:
        try:
            logger.info(f"Swapping {tokenIn} for {tokenOut}")
            
            # Validate inputs
            if not tokenIn or not tokenOut:
                return json.dumps({"error": "Token information cannot be empty", "status": "error"})
    
            if slippage < 0 or slippage > 1:
                return json.dumps({"error": "Slippage must be between 0 and 1", "status": "error"})
            
            # Check if beets connection exists
            if "beets" not in self._agent.connection_manager.connections:
                return json.dumps({"error": "Beets connection not configured", "status": "error"})
            
            # Execute the swap
            params = {
                "tokenIn": tokenIn,
                "tokenOut": tokenOut,
                "slippage": slippage,
                "userAddress": userAddress
            }
                
            response = self._agent.connection_manager.connections["beets"].swap(**params)
            
            logger.info(f"Swap response: {response}")
            if not response:
                return json.dumps({"error": "Failed to swap tokens", "status": "error"})

            return json.dumps({
                "status": "success",
                "data": response,
                "message": f"Successfully swapped {tokenIn} for {tokenOut}"
            })

        except RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return json.dumps({"error": f"API request failed: {str(e)}", "status": "error"})
        except Timeout:
            logger.error("Request timed out")
            return json.dumps({"error": "Request timed out, please try again later", "status": "error"})
        except ConnectionError:
            logger.error("Connection error")
            return json.dumps({"error": "Unable to connect to Beets service", "status": "error"})
        except KeyError as e:
            logger.error(f"Missing key in response: {str(e)}")
            return json.dumps({"error": f"Invalid response format: {str(e)}", "status": "error"})
        except Exception as e:
            logger.error(f"Swap failed: {str(e)}")
            logger.debug(traceback.format_exc())
            return json.dumps({"error": str(e), "status": "error"})

class AddLiquidityInput(BaseModel):
    type: str = Field(..., description="Liquidity type: 'proportional', 'unbalanced', 'single-token', 'boosted-proportional', or 'boosted-unbalanced'")
    pool: str = Field(..., description="Name or address of the Beets pool")
    userAddress: str = Field(..., description="User wallet address(connected wallet)")
    tokensIn: List[str] = Field(..., description="List of input token symbols")
    amountsIn: List[float] = Field([], description="List of input token amounts")
    bptOutAmount: float = Field(0, description="Amount of BPT tokens to receive")
    slippage: float = Field(0.5, description="Slippage tolerance as a percentage (e.g., 0.5 for 0.5%)")


class BeetsAddLiquidityTool(BaseTool):
    name: str = "beets_add_liquidity"
    description: str = """
    beets_add_liquidity: Add liquidity to a Beets pool
    
    Natural language examples:
    - "I want to add 100 USDC of proportional liquidity to Beets pool 'Conspiracy Concerto' with 0.5% slippage". Input = {"type": "proportional", "pool": "Conspiracy Concerto", "slippage": 0.5, "userAddress": "0xUser..", "tokensIn": ["USDC"], "amountsIn": [100]}
    - "Add unbalanced liquidity with 200 USDC and 0.5 ETH to Beets pool 'Conspiracy Concerto' with 1% slippage". Input = {"type": "unbalanced", "pool": "Conspiracy Concerto", "slippage": 1, "userAddress": "0xUser..", "tokensIn": ["USDC", "ETH"], "amountsIn": [200, 0.5]}
    - "Add unbalanced liquidity with 200 USDC to Beets Pool 'Conspiracy Concerto' with 1% slippage". Input = {"type": "unbalanced", "pool": "Conspiracy Concerto", "slippage": 1, "userAddress": "0xUser..", "tokensIn": ["USDC"], "amountsIn": [200]}
    - "Provide DAI single token liquidity to Beets pool 'Conspiracy Concerto' to get 10 BPT". Input = {"type": "single-token", "pool": "Conspiracy Concerto", "slippage": 0.5, "userAddress": "0xUser..", "tokensIn": ["DAI"], "bptOutAmount": 10}
    - "Add boosted proportional liquidity with 500 USDC amount to pool 'Conspiracy Concerto' using USDC and wETH tokens". Input = {"type": "boosted-proportional", "pool": "Conspiracy Concerto", "slippage": 0.5, "userAddress": "0xUser..", "tokensIn": ["USDC"], "amountsIn": [500]}
    
    Input should be a JSON string with:
    - type: "proportional", "unbalanced", "single-token", "boosted-proportional", or "boosted-unbalanced" (required). Default = "proportional"
    - pool: Pool Name (required)
    - userAddress: User wallet address - Connected Wallet (required)
    - tokensIn: List of input token symbols (required)
    - amountsIn: List of input token amounts (for every type except single-token. Not needed for single-token), Default = []
    - bptOutAmount: Amount of BPT tokens to receive (only for 'single-token' type). Default = 0
    - slippage: Slippage tolerance (required). Default = 0.5

    """
    
    args_schema: Type[BaseModel] = AddLiquidityInput

    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def _run(self, type: str, pool: str, userAddress: str, tokensIn: List[str], amountsIn: List[float] = [], bptOutAmount: float = 0, slippage: float = 1.0) -> str:
        try:
            logger.info(f"Adding {type} liquidity to pool {pool}")
            logger.info(f"Tokens in: {tokensIn}, amounts in: {amountsIn}, BPT out: {bptOutAmount}")
            logger.info(f"Slippage: {slippage}, User address: {userAddress}")
            # Validate common inputs
            if not pool:
                return json.dumps({"error": "Pool cannot be empty", "status": "error"})
                
            if not userAddress:
                return json.dumps({"error": "User address cannot be empty. Please connect wallet", "status": "error"})

            if not tokensIn:
                return json.dumps({"error": "Input tokens cannot be empty", "status": "error"})

            valid_types = ["proportional", "unbalanced", "single-token", "boosted-proportional", "boosted-unbalanced"]
            if type not in valid_types:
                return json.dumps({
                    "error": f"Invalid liquidity type. Must be one of: {', '.join(valid_types)}", 
                    "status": "error"
                })
            pools = json.loads(BeetsPoolsQueryTool._run(self, textSearch=pool))
            pool_data = next((pool_info for pool_info in pools["data"] if pool_info["name"] == pool), None)
            if not pool_data:
                return json.dumps({
                    "error": f"Pool '{pool}' not found",
                    "status": "error"
                })
            logger.info(pool_data)
            poolId=pool_data["id"]
            version=pool_data["protocolVersion"]
            action_name = f"add_{type.replace('-', '_')}_liquidity_v{version}"
            
            # Prepare parameters based on the type of liquidity addition
            params = {
                "poolId": poolId,
                "slippage": slippage,
                "userAddress": userAddress
            }
            token_symbols = tokensIn  

            tokensIn = [token for token in pool_data.get("poolTokens", []) if token["symbol"] in token_symbols]
            # Validate and add type-specific parameters
            if type == "proportional":
                if not amountsIn or not tokensIn:
                    return json.dumps({
                        "error": "Reference token and amount are required for proportional liquidity. Please check if input tokens are valid",
                        "status": "error"
                    })
                    
                # Format the reference amount for the API
                params["referenceAmount"] = {
                    "address": tokensIn[0]["address"],
                    "amount": amountsIn[0],
                    "decimals": tokensIn[0]["decimals"]
                }
                
            elif type == "boosted-proportional":
                if not amountsIn or not tokensIn:
                    return json.dumps({
                        "error": "Reference token and amount are required for proportional liquidity. Please check if input tokens are valid",
                        "status": "error"
                    })
                
                params["tokensIn"] = [
                    token["underlyingToken"]["address"] 
                    for token in pool_data.get("poolTokens", []) 
                    if token.get("underlyingToken")
                ]
                
                if not params["tokensIn"]:
                    return json.dumps({
                        "error": "No underlying tokens found for the pool. It may not be a boosted pool.",
                        "status": "error"
                    })
                    
                # Format the reference amount for the API
                params["referenceAmount"] = {
                    "address": tokensIn[0]["address"],
                    "amount": amountsIn[0],
                    "decimals": tokensIn[0]["decimals"]
                }
                
            elif type == "unbalanced" or type == "boosted-unbalanced":
                if not amountsIn or not tokensIn:
                    return json.dumps({
                        "error": "Reference token and amount are required for proportional liquidity. Please check if input tokens are valid",
                        "status": "error"
                    })
                    
                # Format each amount in the list
                formatted_amounts = []
                i = 0
                for amount in amountsIn:
                    formatted_amounts.append({
                        "address": tokensIn[i]["address"],
                        "amount": amount,
                        "decimals": tokensIn[i]["decimals"]
                    })
                    i += 1
                params["amountsIn"] = formatted_amounts
                
            elif type == "single-token":
                if bptOutAmount is None:
                    return json.dumps({
                        "error": "BPT out amount is required for single token liquidity",
                        "status": "error"
                    })
                if not tokensIn:
                    return json.dumps({
                        "error": "Token address or symbol is required for single token liquidity",
                        "status": "error"
                    })
                    
                # Format BPT output for the API
                params["bptOut"] = {
                    "address": pool_data["address"],
                    "amount": bptOutAmount,
                    "decimals": 18
                }
                params["tokenIn"] = tokensIn[0]["address"]
            
            logger.info(f"Using method: {action_name} with params: {params}")
            
            # Check if the method exists
            if not hasattr(self._agent.connection_manager.connections["beets"], action_name):
                return json.dumps({
                    "error": f"Method {action_name} not found in Beets connection", 
                    "status": "error"
                })
                
            # Call the appropriate method
            response = getattr(self._agent.connection_manager.connections["beets"], action_name)(**params)
            
            logger.info(f"Add liquidity response: {response}")
            if not response:
                return json.dumps({"error": "Failed to add liquidity", "status": "error"})
            
            response["status"] = "Initiated. Continue in the frontend."
            return json.dumps(response)

        except RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return json.dumps({"error": f"API request failed: {str(e)}", "status": "error"})
        except Timeout:
            logger.error("Request timed out")
            return json.dumps({"error": "Request timed out, please try again later", "status": "error"})
        except ConnectionError:
            logger.error("Connection error")
            return json.dumps({"error": "Unable to connect to Beets service", "status": "error"})
        except AttributeError as e:
            logger.error(f"Method error: {str(e)}")
            return json.dumps({"error": f"Operation not supported: {str(e)}", "status": "error"})
        except Exception as e:
            logger.error(f"Add liquidity failed: {str(e)}")
            logger.debug(traceback.format_exc())
            return json.dumps({"error": str(e), "status": "error"})


class RemoveLiquidityInput(BaseModel):
    type: str = Field(..., description="Liquidity removal type: 'proportional', 'single-token-exact-in', 'single-token-exact-out', 'unbalanced', or 'boosted-proportional'")
    pool: str = Field(..., description="Name or address of the Beets pool")
    slippage: float = Field(0.5, description="Slippage tolerance as a percentage (e.g., 0.5 for 0.5%)")
    userAddress: str = Field(..., description="User wallet address(connected wallet)")
    bptAmount: float = Field(0, description="Amount of BPT tokens to withdraw (for proportional, boosted-proportional, and single-token-exact-in)")
    tokensOut: List[str] = Field([], description="List of output tokens(for single-token types & unbalanced)")
    amountsOut: List[float] = Field([], description="List of token amounts to withdraw (for unbalanced & single-token-exact-out)")


class BeetsRemoveLiquidityTool(BaseTool):
    name: str = "beets_remove_liquidity"
    description: str = """
    beets_remove_liquidity: Remove liquidity from a Beets pool
    
    Natural language examples:
    - "I want to withdraw 10 BPT proportionally from Beets pool 'Conspiracy Concerto' with 0.5% slippage". Input = {"type": "proportional", "pool": "Conspiracy Concerto", "bptAmount": 10, "slippage": 0.5, "userAddress": "0xUser.."}
    - "Remove 15 BPT from Beets pool 'Conspiracy Concerto' and get USDC in return". Input = {"type": "single-token-exact-in", "pool": "Conspiracy Concerto", "bptAmount": 15, "tokensOut": ["USDC"], "slippage": 0.5, "userAddress": "0xUser.."}
    - "Take out exactly 500 USDC from my liquidity in Beets pool 'Conspiracy Concerto'". Input = {"type": "single-token-exact-out", "pool": "Conspiracy Concerto", "tokensOut": ["USDC"], "amountsOut": [500], "slippage": 0.5, "userAddress": "0xUser.."}
    - "Remove unbalanced liquidity from 'Conspiracy Concerto' to get specific amounts of tokens". Input = {"type": "unbalanced", "pool": "Conspiracy Concerto", "tokensOut": ["USDC", "ETH"], "amountsOut": [100, 0.5], "slippage": 0.5, "userAddress": "0xUser.."}
    - "Withdraw 20 BPT with boosted proportional strategy from 'Conspiracy Concerto' pool". Input = {"type": "boosted-proportional", "pool": "Conspiracy Concerto", "bptAmount": 20, "slippage": 0.5, "userAddress": "0xUser.."}
    
    Input should be a JSON string with:
    - type: "proportional", "single-token-exact-in", "single-token-exact-out", "unbalanced", or "boosted-proportional" (required)
    - pool: Pool Name (required)
    - slippage: Slippage tolerance (required). Default = 0.5
    - userAddress: User wallet address - Connected Wallet (required)
    - bptAmount: Amount of BPT tokens to withdraw (for proportional, boosted-proportional, and single-token-exact-in)
    - tokensOut: List of output tokens(for single-token types & unbalanced)
    - amountsOut: List of token amounts to withdraw (for unbalanced & single-token-exact-out)
    """
    
    args_schema: Type[BaseModel] = RemoveLiquidityInput

    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def _run(self, type: str, pool: str, userAddress: str, 
             bptAmount: float = 0, tokensOut: List[str] = [], amountsOut: List[float] = [], slippage: float = 0.5) -> str:
        try:
            logger.info(f"Removing {type} liquidity from pool {pool}")
            logger.info(f"User address: {userAddress}")
            logger.info(f"BPT amount: {bptAmount}, Tokens out: {tokensOut}, Amounts out: {amountsOut}")
            logger.info(f"Slippage: {slippage}")
            # Validate common inputs
            if not pool:
                return json.dumps({"error": "Pool cannot be empty", "status": "error"})
                
            if slippage <= 0 or slippage > 100:
                return json.dumps({"error": "Slippage must be between 0 and 100", "status": "error"})
                
            if not userAddress:
                return json.dumps({"error": "User address cannot be empty. Please connect wallet", "status": "error"})
                
            valid_types = ["proportional", "single-token-exact-in", "single-token-exact-out", "unbalanced", "boosted-proportional"]
            if type not in valid_types:
                return json.dumps({
                    "error": f"Invalid liquidity removal type. Must be one of: {', '.join(valid_types)}", 
                    "status": "error"
                })
            
            # Get pool data from BeetsPoolsQueryTool
            pools = json.loads(BeetsPoolsQueryTool(self._agent)._run(textSearch=pool))
            if pools.get("status") != "success" or not pools.get("data"):
                return json.dumps({
                    "error": f"Failed to get pool information for '{pool}'",
                    "status": "error"
                })
                
            pool_data = next((pool_info for pool_info in pools["data"] if pool_info["name"] == pool), None)
            if not pool_data:
                return json.dumps({
                    "error": f"Pool '{pool}' not found",
                    "status": "error"
                })
                
            poolId = pool_data["id"]
            version = pool_data["protocolVersion"]
            
            # Determine the appropriate method name based on type and version
            action_name = f"remove_{type.replace('-', '_')}_liquidity_v{version}"
            
            # Check if beets connection exists
            if "beets" not in self._agent.connection_manager.connections:
                return json.dumps({"error": "Beets connection not configured", "status": "error"})
            
            # Set up base parameters
            params = {
                "poolId": poolId,
                "slippage": slippage,
                "userAddress": userAddress
            }
            
            # Handle type-specific parameters
            if type in ["proportional", "boosted-proportional"]:
                if bptAmount <= 0:
                    return json.dumps({
                        "error": f"BPT amount is required for {type} withdrawal",
                        "status": "error"
                    })
                    
                # Format BPT amount for API
                params["bptIn"] = {
                    "address": pool_data["address"],
                    "amount": bptAmount,
                    "decimals": 18
                }
                
            elif type == "single-token-exact-in":
                if bptAmount <= 0:
                    return json.dumps({
                        "error": "BPT amount is required for single token exact in withdrawal",
                        "status": "error"
                    })
                if not tokensOut or len(tokensOut) == 0:
                    return json.dumps({
                        "error": "Token out symbol is required for single token withdrawal",
                        "status": "error"
                    })
                    
                # Find the token in pool tokens
                token_data = next((token for token in pool_data.get("poolTokens", []) 
                                  if token["symbol"] == tokensOut[0]), None)
                if not token_data:
                    return json.dumps({
                        "error": f"Token '{tokensOut[0]}' not found in pool",
                        "status": "error"
                    })
                    
                # Format BPT amount for API
                params["bptIn"] = {
                    "address": pool_data["address"],
                    "amount": bptAmount,
                    "decimals": 18
                }
                params["tokenOut"] = token_data["address"]
                
            elif type == "single-token-exact-out":
                if len(amountsOut) == 0:
                    return json.dumps({
                        "error": "Amount out is required for single token exact out withdrawal",
                        "status": "error"
                    })
                if not tokensOut or len(tokensOut) == 0:
                    return json.dumps({
                        "error": "Token out symbol is required for single token withdrawal",
                        "status": "error"
                    })
                    
                # Find the token in pool tokens
                token_data = next((token for token in pool_data.get("poolTokens", []) 
                                  if token["symbol"] == tokensOut[0]), None)
                if not token_data:
                    return json.dumps({
                        "error": f"Token '{tokensOut[0]}' not found in pool",
                        "status": "error"
                    })
                    
                # Format amount out for API
                params["amountOut"] = {
                    "address": token_data["address"],
                    "amount": amountsOut[0],
                    "decimals": token_data["decimals"]
                }
                
            elif type == "unbalanced":
                if len(amountsOut) == 0 or len(tokensOut) == 0:
                    return json.dumps({
                        "error": "Tokens list and amounts list are required for unbalanced withdrawal",
                        "status": "error"
                    })
                    
                if len(tokensOut) != len(amountsOut):
                    return json.dumps({
                        "error": "The number of tokens must match the number of amounts for unbalanced withdrawal",
                        "status": "error"
                    })
                    
                # Format each amount in the list
                formatted_amounts = []
                for i in range(len(tokensOut)):
                    token_symbol = tokensOut[i]
                    amount = amountsOut[i]
                        
                    # Find the token in pool tokens
                    token_data = next((token for token in pool_data.get("poolTokens", []) 
                                      if token["symbol"] == token_symbol), None)
                    if not token_data:
                        return json.dumps({
                            "error": f"Token '{token_symbol}' not found in pool",
                            "status": "error"
                        })
                    formatted_amounts.append({
                        "address": token_data["address"],
                        "amount": amount,
                        "decimals": token_data["decimals"]
                    })
                
                params["amountsOut"] = formatted_amounts
            
            logger.info(f"Using method: {action_name} with params: {params}")
            
            # Check if the method exists
            if not hasattr(self._agent.connection_manager.connections["beets"], action_name):
                return json.dumps({
                    "error": f"Method {action_name} not found in Beets connection. This operation might not be supported for {version} pools.", 
                    "status": "error"
                })
            
            # Call the appropriate method
            response = getattr(self._agent.connection_manager.connections["beets"], action_name)(**params)
            
            logger.info(f"Remove liquidity response: {response}")
            if not response:
                return json.dumps({"error": "Failed to remove liquidity", "status": "error"})
            response["status"] = "Initiated. Continue in the frontend."
            return json.dumps(response)

        except RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return json.dumps({"error": f"API request failed: {str(e)}", "status": "error"})
        except Timeout:
            logger.error("Request timed out")
            return json.dumps({"error": "Request timed out, please try again later", "status": "error"})
        except ConnectionError:
            logger.error("Connection error")
            return json.dumps({"error": "Unable to connect to Beets service", "status": "error"})
        except AttributeError as e:
            logger.error(f"Method error: {str(e)}")
            return json.dumps({"error": f"Operation not supported: {str(e)}", "status": "error"})
        except Exception as e:
            logger.error(f"Remove liquidity failed: {str(e)}")
            logger.debug(traceback.format_exc())
            return json.dumps({"error": str(e), "status": "error"})

class BeetsTokenQueryTool(BaseTool):
    name: str = "beets_token_query"
    description: str = """
    beets_token_query: Get token information from Beets
    
    Natural language examples:
    - "Find information about the USDC token on Beets"
    - "What's the contract address of ETH token on Beets?"
    - "Get details on token with address 0x1234...abcd on Beets"
    
    Input should be a JSON string with one of the following:
    - symbol: Token symbol (or)
    - address: Token contract address
    
    Examples:
    {"symbol": "ETH"} 
    {"address": "0x1234...abcd"}
    """

    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def _run(self, symbol: str = None, address: str = None) -> str:
        try:
            logger.info(f"Token query with symbol: {symbol}, address: {address}")
            
            # Check if beets connection exists
            if "beets" not in self._agent.connection_manager.connections:
                return json.dumps({"error": "Beets connection not configured", "status": "error"})
            
            # Handle case where input might be a string (handle parsing if needed)
            if symbol is not None:
                logger.info(f"Getting token information for symbol: {symbol}")
                response = self._agent.connection_manager.connections["beets"].get_token_by_symbol(symbol)
                return json.dumps({
                    "status": "success", 
                    "data": response,
                    "message": f"Found token information for {symbol}"
                })
            elif address is not None:
                logger.info(f"Getting token information for address: {address}")
                response = self._agent.connection_manager.connections["beets"].get_token_by_address(address)
                return json.dumps({
                    "status": "success", 
                    "data": response,
                    "message": f"Found token information for address {address}"
                })
            else:
                return json.dumps({"error": "Either token symbol or address must be provided", "status": "error"})

        except Exception as e:
            logger.error(f"Token query failed: {str(e)}")
            logger.debug(traceback.format_exc())
            return json.dumps({"error": str(e), "status": "error"})

class BeetsPoolsQueryTool(BaseTool):
    name: str = "beets_pools_query"
    description: str = """
    beets_pools_query: Get and filter pools from Beets
    
    Natural language examples:
    - "Show me all Beets pools". Input = {}
    - "List the top 5 Beets pools sorted by APR". Input = {"first": 5, "orderBy": "apr", "orderDirection": "desc"}
    - "Find the highest liquidity pools on Beets". Input = {"orderBy": "totalLiquidity", "orderDirection": "desc"}
    - "Search for 'ETH' pools on Beets". Input = {"textSearch": "ETH"}
    - "Get pools for user 0x123... on Beets". Input = {"userAddress": "0x123..."}
    - Tell me some USDC pools on Beets. Input = {"textSearch": "USDC"}
    
    Input should be a JSON string with optional parameters:
    - userAddress: User's wallet address to get user-specific data
    - first: Maximum number of pools to return
    - orderBy: Field to sort by - options: "apr", "fees24h", "totalLiquidity", "volume24h", "totalShares", "userBalanceUsd"
    - orderDirection: Sort direction - options: "asc", "desc"
    - skip: Number of pools to skip (for pagination)
    - textSearch: Text to search for in pool name/symbol
    
    Outputs: 
        id
        address
        chain
        name
        symbol
        protocolVersion
        type
        userBalance {
            totalBalanceUsd
        }
        dynamicData {
            totalLiquidity
            volume24h
            yieldCapture24h
        }
        poolTokens {
            id
            address
            symbol
            decimals
            name
            logoURI
            underlyingToken {
                address
                symbol
                name
                decimals
            }
        }
    
    Examples:
    {"first": 10, "orderBy": "apr", "orderDirection": "desc"}
    {"textSearch": "ETH", "first": 5}
    {"userAddress": "0xUser...", "orderBy": "userBalanceUsd", "orderDirection": "desc"}
    """

    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def _run(self, userAddress: str = None, first: int = None, orderBy: str = None, 
             orderDirection: str = None, skip: int = None, textSearch: str = None) -> str:
        try:
            # Build log message based on provided parameters
            log_parts = ["Querying Beets pools"]
            if userAddress:
                log_parts.append(f"for user {userAddress}")
            if textSearch:
                log_parts.append(f"with search term '{textSearch}'")
            if orderBy:
                direction = orderDirection or "desc"
                log_parts.append(f"ordered by {orderBy} {direction}")
            if first:
                log_parts.append(f"limit {first}")
            if skip:
                log_parts.append(f"skipping {skip}")
                
            logger.info(" ".join(log_parts))
            
            # Check if beets connection exists
            if "beets" not in self._agent.connection_manager.connections:
                return json.dumps({"error": "Beets connection not configured", "status": "error"})
            
            # Call the get_pools method with provided parameters
            response = self._agent.connection_manager.connections["beets"].get_pools(
                userAddress=userAddress,
                first=first,
                orderBy=orderBy,
                orderDirection=orderDirection,
                skip=skip,
                textSearch=textSearch
            )
            
            return json.dumps({
                "status": "success", 
                "data": response,
            })

        except RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return json.dumps({"error": f"API request failed: {str(e)}", "status": "error"})
        except Timeout:
            logger.error("Request timed out")
            return json.dumps({"error": "Request timed out, please try again later", "status": "error"})
        except ConnectionError:
            logger.error("Connection error")
            return json.dumps({"error": "Unable to connect to Beets service", "status": "error"})
        except Exception as e:
            logger.error(f"Pools query failed: {str(e)}")
            logger.debug(traceback.format_exc())
            return json.dumps({"error": str(e), "status": "error"})

class BeetsStakeDepositTool(BaseTool):
    name: str = "beets_stake_deposit"
    description: str = """
    beets_stake_deposit: Stake SONIC tokens on Beets
    
    Natural language examples:
    - "Stake 100 SONIC tokens"
    - "Deposit 50 SONIC to staking"
    - "I want to stake SONIC"
    
    Input should be a JSON string with:
    - amount: Amount of SONIC to stake (required)
    - userAddress: User wallet address - Connected Wallet (required)
    
    Example: {"amount": "10", "userAddress": "0xUser.."}
    """

    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def _run(self, amount: str, userAddress: str) -> str:
        try:
            logger.info(f"Staking {amount} SONIC tokens")
            
            # Validate inputs
            if not amount:
                return json.dumps({"error": "Amount cannot be empty", "status": "error"})
    
            if not userAddress:
                return json.dumps({"error": "User address cannot be empty", "status": "error"})
            
            # Check if beets connection exists
            if "beets" not in self._agent.connection_manager.connections:
                return json.dumps({"error": "Beets connection not configured", "status": "error"})
            amount = str(int(float(amount)*(10**18)))
            # Execute the stake deposit
            response = self._agent.connection_manager.connections["beets"].stake_deposit(amount, userAddress)
            
            logger.info(f"Stake deposit response: {response}")
            if not response:
                return json.dumps({"error": "Failed to stake SONIC tokens", "status": "error"})
            response["transaction"]["type"] = "stake"
            response["transaction"]["chainId"] = 146
            return json.dumps(response["transaction"])

        except RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return json.dumps({"error": f"API request failed: {str(e)}", "status": "error"})
        except Timeout:
            logger.error("Request timed out")
            return json.dumps({"error": "Request timed out, please try again later", "status": "error"})
        except ConnectionError:
            logger.error("Connection error")
            return json.dumps({"error": "Unable to connect to Beets service", "status": "error"})
        except KeyError as e:
            logger.error(f"Missing key in response: {str(e)}")
            return json.dumps({"error": f"Invalid response format: {str(e)}", "status": "error"})
        except Exception as e:
            logger.error(f"Stake deposit failed: {str(e)}")
            logger.debug(traceback.format_exc())
            return json.dumps({"error": str(e), "status": "error"})


class BeetsStakeUndelegateTool(BaseTool):
    name: str = "beets_stake_undelegate"
    description: str = """
    beets_stake_undelegate: Undelegate staked SONIC(stS) shares from the pool
    
    Natural language examples:
    - "Undelegate 50 SONIC shares"
    - "Start the withdrawal process for my staked SONIC"
    - "I want to undelegate my SONIC stake"
    
    Input should be a JSON string with:
    - amountShares: Amount of shares to undelegate (required)
    - userAddress: User wallet address (required)
    
    Example: {"amountShares": "5", "userAddress": "0xUser.."}
    """

    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def _run(self, amountShares: str, userAddress: str) -> str:
        try:
            logger.info(f"Undelegating {amountShares} SONIC shares")
            
            # Validate inputs
            if not amountShares:
                return json.dumps({"error": "Amount shares cannot be empty", "status": "error"})
    
            if not userAddress:
                return json.dumps({"error": "User address cannot be empty", "status": "error"})
            
            # Check if beets connection exists
            if "beets" not in self._agent.connection_manager.connections:
                return json.dumps({"error": "Beets connection not configured", "status": "error"})
            amountShares = str(int(float(amountShares)*(10**18)))
            
            # Execute the undelegate
            response = self._agent.connection_manager.connections["beets"].stake_undelegate(amountShares, userAddress)
            
            logger.info(f"Stake undelegate response: {response}")
            if not response:
                return json.dumps({"error": "Failed to undelegate SONIC shares", "status": "error"})
            response["transaction"]["type"] = "undelegate"
            response["transaction"]["chainId"] = 146
            return json.dumps(response["transaction"])

        except RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return json.dumps({"error": f"API request failed: {str(e)}", "status": "error"})
        except Timeout:
            logger.error("Request timed out")
            return json.dumps({"error": "Request timed out, please try again later", "status": "error"})
        except ConnectionError:
            logger.error("Connection error")
            return json.dumps({"error": "Unable to connect to Beets service", "status": "error"})
        except KeyError as e:
            logger.error(f"Missing key in response: {str(e)}")
            return json.dumps({"error": f"Invalid response format: {str(e)}", "status": "error"})
        except Exception as e:
            logger.error(f"Stake undelegate failed: {str(e)}")
            logger.debug(traceback.format_exc())
            return json.dumps({"error": str(e), "status": "error"})


class BeetsStakeWithdrawTool(BaseTool):
    name: str = "beets_stake_withdraw"
    description: str = """
    beets_stake_withdraw: Withdraw undelegated SONIC tokens
    
    Natural language examples:
    - "Withdraw my undelegated SONIC tokens"
    - "Complete the withdrawal for my SONIC tokens"
    - "I want to finalize my SONIC withdrawal with ID 123"
    
    Input should be a JSON string with:
    - withdrawId: ID of the withdrawal request (required)
    
    Example: {"withdrawId": "123"}
    """

    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def _run(self, withdrawId: str) -> str:
        try:
            logger.info(f"Withdrawing SONIC tokens with withdrawal ID {withdrawId}")
            
            # Validate inputs
            if not withdrawId:
                return json.dumps({"error": "Withdrawal ID cannot be empty", "status": "error"})
            
            # Check if beets connection exists
            if "beets" not in self._agent.connection_manager.connections:
                return json.dumps({"error": "Beets connection not configured", "status": "error"})
            
            # Execute the withdrawal
            response = self._agent.connection_manager.connections["beets"].stake_withdraw(withdrawId)
            
            logger.info(f"Stake withdraw response: {response}")
            if not response:
                return json.dumps({"error": "Failed to withdraw SONIC tokens", "status": "error"})
            response["transaction"]["type"] = "withdraw_stake"
            response["transaction"]["chainId"] = 146
            return json.dumps(response["transaction"])

        except RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return json.dumps({"error": f"API request failed: {str(e)}", "status": "error"})
        except Timeout:
            logger.error("Request timed out")
            return json.dumps({"error": "Request timed out, please try again later", "status": "error"})
        except ConnectionError:
            logger.error("Connection error")
            return json.dumps({"error": "Unable to connect to Beets service", "status": "error"})
        except KeyError as e:
            logger.error(f"Missing key in response: {str(e)}")
            return json.dumps({"error": f"Invalid response format: {str(e)}", "status": "error"})
        except Exception as e:
            logger.error(f"Stake withdraw failed: {str(e)}")
            logger.debug(traceback.format_exc())
            return json.dumps({"error": str(e), "status": "error"})


class BeetsStakingInfoTool(BaseTool):
    name: str = "beets_staking_info"
    description: str = """
    beets_staking_info: Get staking information for a user
    
    Natural language examples:
    - "Show my SONIC staking information"
    - "Check my staked SONIC balance"
    - "Get details about my SONIC staking position"
    
    Input should be a JSON string with:
    - userAddress: User wallet address (required)
    
    Example: {"userAddress": "0xUser.."}
    """

    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def _run(self, userAddress: str) -> str:
        try:
            logger.info(f"Getting staking information for user {userAddress}")
            
            # Validate inputs
            if not userAddress:
                return json.dumps({"error": "User address cannot be empty", "status": "error"})
            
            # Check if beets connection exists
            if "beets" not in self._agent.connection_manager.connections:
                return json.dumps({"error": "Beets connection not configured", "status": "error"})
            
            # Get staking information
            response = self._agent.connection_manager.connections["beets"].get_staking_info(userAddress)
            
            logger.info(f"Staking info response: {response}")
            if not response:
                return json.dumps({"error": "Failed to get staking information", "status": "error"})

            return json.dumps({
                "status": "success",
                "data": response,
                "message": f"Retrieved staking information for {userAddress}"
            })

        except RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return json.dumps({"error": f"API request failed: {str(e)}", "status": "error"})
        except Timeout:
            logger.error("Request timed out")
            return json.dumps({"error": "Request timed out, please try again later", "status": "error"})
        except ConnectionError:
            logger.error("Connection error")
            return json.dumps({"error": "Unable to connect to Beets service", "status": "error"})
        except KeyError as e:
            logger.error(f"Missing key in response: {str(e)}")
            return json.dumps({"error": f"Invalid response format: {str(e)}", "status": "error"})
        except Exception as e:
            logger.error(f"Getting staking info failed: {str(e)}")
            logger.debug(traceback.format_exc())
            return json.dumps({"error": str(e), "status": "error"})

def get_beets_tools(agent) -> list:
    """Return a list of all Beets-related tools."""
    return [
        BeetsSwapTool(agent),
        BeetsAddLiquidityTool(agent),
        BeetsRemoveLiquidityTool(agent),
        BeetsTokenQueryTool(agent),
        BeetsPoolsQueryTool(agent),
        BeetsStakeDepositTool(agent),
        BeetsStakeUndelegateTool(agent),
        BeetsStakeWithdrawTool(agent),
        BeetsStakingInfoTool(agent)
    ]
