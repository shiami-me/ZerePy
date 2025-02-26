from langchain.tools import BaseTool
import json
import logging
import traceback
from requests.exceptions import RequestException, Timeout, ConnectionError
from typing import Optional, Dict, Any, List, Union

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
    - slippage: Slippage tolerance (e.g. 0.01 for 1%) (required)
    - userAddress: User address (required)
    - poolId: Pool ID to use for the swap (optional)
    
    Example: Swap 1 ETH for USDC. Use: {"tokenIn": "ETH", "tokenOut": "USDC", "slippage": 0.01, "userAddress": "0xUser.."}
    """

    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def _run(self, tokenIn: str, tokenOut: str, slippage: float = 0.005, userAddress: str = "", poolId: str = None) -> str:
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
            
            if poolId:
                params["poolId"] = poolId
                
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


class BeetsAddLiquidityTool(BaseTool):
    name: str = "beets_add_liquidity"
    description: str = """
    beets_add_liquidity: Add liquidity to a Beets pool
    
    Natural language examples:
    - "I want to add 100 USDC of proportional liquidity to Beets v3 pool 0x1234...abcd with 0.5% slippage"
    - "Add unbalanced liquidity with 200 USDC and 0.5 ETH to Beets v2 pool 0xabcd"
    - "Provide 50 DAI single token liquidity to Beets pool 0x5678...efgh to get 10 BPT"
    - "Add boosted proportional liquidity with 500 USDC reference amount to pool 0x1234 using USDC and wETH tokens"
    
    Input should be a JSON string with:
    - type: "proportional", "unbalanced", "single-token", "boosted-proportional", or "boosted-unbalanced" (required)
    - version: "v2" or "v3" (required)
    - poolId: Pool ID (required)
    - slippage: Slippage tolerance (required)
    - userAddress: User wallet address (required)
    
    Additional parameters based on type:
    - For proportional: 
        referenceToken: Token symbol/address
        referenceAmount: Amount to provide
    - For unbalanced: 
        amountsIn: Array of {token: symbol/address, amount: value} objects
    - For single-token: 
        bptOut: Amount of BPT tokens to receive
        tokenIn: Token symbol/address
    - For boosted-proportional: 
        referenceToken: Token symbol/address
        referenceAmount: Amount to provide
        tokensIn: Array of token symbols/addresses to use
    - For boosted-unbalanced: 
        amountsIn: Array of {token: symbol/address, amount: value} objects
    
    Examples:
    - Proportional: {"type": "proportional", "version": "v3", "referenceToken": "USDC", "referenceAmount": 100, "poolId": "0xabc", "slippage": 0.01, "userAddress": "0xUser.."}
    - Unbalanced: {"type": "unbalanced", "version": "v3", "amountsIn": [{"token": "USDC", "amount": 100}, {"token": "ETH", "amount": 0.5}], "poolId": "0xabc", "slippage": 0.01, "userAddress": "0xUser.."}
    """

    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def _run(self, type: str, version: str, poolId: str, slippage: float, userAddress: str, 
             referenceToken: str = None, referenceAmount: float = None, 
             amountsIn: List = None, bptOut: float = None, tokenIn: str = None, 
             tokensIn: List = None, tokenDecimals: int = 18) -> str:
        try:
            logger.info(f"Adding {type} liquidity to {version} pool {poolId}")
            
            # Validate common inputs
            if not poolId:
                return json.dumps({"error": "Pool ID cannot be empty", "status": "error"})
                
            if slippage <= 0 or slippage > 1:
                return json.dumps({"error": "Slippage must be between 0 and 1", "status": "error"})
                
            if not userAddress:
                return json.dumps({"error": "User address cannot be empty", "status": "error"})
                
            if version not in ["v2", "v3"]:
                return json.dumps({"error": "Version must be either 'v2' or 'v3'", "status": "error"})
                
            valid_types = ["proportional", "unbalanced", "single-token", "boosted-proportional", "boosted-unbalanced"]
            if type not in valid_types:
                return json.dumps({
                    "error": f"Invalid liquidity type. Must be one of: {', '.join(valid_types)}", 
                    "status": "error"
                })
            
            # Check if beets connection exists
            if "beets" not in self._agent.connection_manager.connections:
                return json.dumps({"error": "Beets connection not configured", "status": "error"})
            
            action_name = f"add_{type.replace('-', '_')}_liquidity_{version}"
            
            # Prepare parameters based on the type of liquidity addition
            params = {
                "poolId": poolId,
                "slippage": slippage,
                "userAddress": userAddress
            }
            
            # Validate and add type-specific parameters
            if type == "proportional":
                if not referenceToken or not referenceAmount:
                    return json.dumps({
                        "error": "Reference token and amount are required for proportional liquidity",
                        "status": "error"
                    })
                    
                # Format the reference amount for the API
                params["referenceAmount"] = {
                    "address": referenceToken,
                    "amount": referenceAmount,
                    "decimals": tokenDecimals
                }
                
            elif type == "boosted-proportional":
                if not referenceToken or not referenceAmount:
                    return json.dumps({
                        "error": "Reference token and amount are required for boosted proportional liquidity",
                        "status": "error"
                    })
                if not tokensIn:
                    return json.dumps({
                        "error": "Tokens list is required for boosted proportional liquidity",
                        "status": "error"
                    })
                    
                # Format the reference amount for the API
                params["referenceAmount"] = {
                    "address": referenceToken,
                    "amount": referenceAmount,
                    "decimals": tokenDecimals
                }
                params["tokensIn"] = tokensIn
                
            elif type == "unbalanced" or type == "boosted-unbalanced":
                if not amountsIn or not isinstance(amountsIn, list) or len(amountsIn) == 0:
                    return json.dumps({
                        "error": "Amounts list with at least one token is required for unbalanced liquidity",
                        "status": "error"
                    })
                    
                # Format each amount in the list
                formatted_amounts = []
                for amount in amountsIn:
                    if isinstance(amount, dict) and "token" in amount and "amount" in amount:
                        formatted_amounts.append({
                            "address": amount["token"],
                            "amount": amount["amount"],
                            "decimals": amount.get("decimals", tokenDecimals)
                        })
                    else:
                        return json.dumps({
                            "error": "Each amount must contain 'token' and 'amount' fields",
                            "status": "error"
                        })
                
                params["amountsIn"] = formatted_amounts
                
            elif type == "single-token":
                if bptOut is None:
                    return json.dumps({
                        "error": "BPT out amount is required for single token liquidity",
                        "status": "error"
                    })
                if not tokenIn:
                    return json.dumps({
                        "error": "Token address or symbol is required for single token liquidity",
                        "status": "error"
                    })
                    
                # Format BPT output for the API
                params["bptOut"] = {
                    "address": poolId,  # BPT token address is typically the pool address
                    "amount": bptOut,
                    "decimals": 18  # BPT tokens typically have 18 decimals
                }
                params["tokenIn"] = tokenIn
            
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

            return json.dumps({
                "status": "success", 
                "transaction": response,
                "message": f"Successfully added {type} liquidity to pool {poolId}"
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
        except AttributeError as e:
            logger.error(f"Method error: {str(e)}")
            return json.dumps({"error": f"Operation not supported: {str(e)}", "status": "error"})
        except Exception as e:
            logger.error(f"Add liquidity failed: {str(e)}")
            logger.debug(traceback.format_exc())
            return json.dumps({"error": str(e), "status": "error"})


class BeetsRemoveLiquidityTool(BaseTool):
    name: str = "beets_remove_liquidity"
    description: str = """
    beets_remove_liquidity: Remove liquidity from a Beets pool
    
    Natural language examples:
    - "I want to withdraw 10 BPT proportionally from Beets v3 pool 0x1234...abcd with 0.5% slippage"
    - "Remove 15 BPT from Beets pool 0xabcd and get USDC in return"
    - "Take out exactly 500 USDC from my liquidity in Beets v2 pool 0x5678"
    - "Withdraw my liquidity from Beets pool 0x1234 to get DAI tokens"
    - "Remove unbalanced liquidity from v2 pool to get 100 USDC and 0.5 ETH"
    - "Withdraw 20 BPT with boosted proportional strategy from v3 pool"
    
    Input should be a JSON string with:
    - type: "proportional", "single-token-exact-in", "single-token-exact-out", "unbalanced", or "boosted-proportional" (required)
    - version: "v2" or "v3" (required)
    - poolId: Pool ID (required)
    - slippage: Slippage tolerance (required)
    - userAddress: User wallet address (required)
    
    Additional parameters based on type:
    - For proportional or boosted-proportional: 
        bptAmount: Amount of BPT tokens to withdraw
    - For single-token-exact-in: 
        bptAmount: Amount of BPT tokens to remove
        tokenOut: Token symbol/address to receive
    - For single-token-exact-out: 
        tokenOut: Token symbol/address to receive
        amountOut: Amount of tokens to receive
    - For unbalanced:
        amountsOut: Array of {token: symbol/address, amount: value} objects 
    
    Examples:
    - Proportional: {"type": "proportional", "version": "v3", "bptAmount": 10, "poolId": "0xabc", "slippage": 0.01, "userAddress": "0xUser.."}
    - Single token (exact in): {"type": "single-token-exact-in", "version": "v3", "bptAmount": 10, "tokenOut": "USDC", "poolId": "0xabc", "slippage": 0.01, "userAddress": "0xUser.."}
    - Single token (exact out): {"type": "single-token-exact-out", "version": "v2", "tokenOut": "USDC", "amountOut": 100, "poolId": "0xabc", "slippage": 0.01, "userAddress": "0xUser.."}
    - Unbalanced: {"type": "unbalanced", "version": "v2", "amountsOut": [{"token": "USDC", "amount": 100}, {"token": "ETH", "amount": 0.5}], "poolId": "0xabc", "slippage": 0.01, "userAddress": "0xUser.."}
    - Boosted proportional: {"type": "boosted-proportional", "version": "v3", "bptAmount": 10, "poolId": "0xabc", "slippage": 0.01, "userAddress": "0xUser.."}
    """

    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def _run(self, type: str, version: str, poolId: str, slippage: float, userAddress: str, 
             bptAmount: float = None, tokenOut: str = None, amountOut: float = None, 
             amountsOut: List = None, tokenDecimals: int = 18) -> str:
        try:
            logger.info(f"Removing {type} liquidity from {version} pool {poolId}")
            
            # Validate common inputs
            if not poolId:
                return json.dumps({"error": "Pool ID cannot be empty", "status": "error"})
                
            if slippage <= 0 or slippage > 1:
                return json.dumps({"error": "Slippage must be between 0 and 1", "status": "error"})
                
            if not userAddress:
                return json.dumps({"error": "User address cannot be empty", "status": "error"})
                
            if version not in ["v2", "v3"]:
                return json.dumps({"error": "Version must be either 'v2' or 'v3'", "status": "error"})
                
            valid_types = ["proportional", "single-token-exact-in", "single-token-exact-out", "unbalanced", "boosted-proportional"]
            if type not in valid_types:
                return json.dumps({
                    "error": f"Invalid liquidity removal type. Must be one of: {', '.join(valid_types)}", 
                    "status": "error"
                })
            
            # Check if beets connection exists
            if "beets" not in self._agent.connection_manager.connections:
                return json.dumps({"error": "Beets connection not configured", "status": "error"})
            
            # Check type and version compatibility
            if type == "unbalanced" and version != "v2":
                return json.dumps({
                    "error": "Unbalanced liquidity removal is only supported for v2 pools", 
                    "status": "error"
                })
                
            if type == "boosted-proportional" and version != "v3":
                return json.dumps({
                    "error": "Boosted proportional liquidity removal is only supported for v3 pools", 
                    "status": "error"
                })
            
            # Set action name based on type and version
            action_name = f"remove_{type.replace('-', '_')}_{version}"
            
            # Set up base parameters
            params = {
                "poolId": poolId,
                "slippage": slippage,
                "userAddress": userAddress
            }
            
            # Validate and add type-specific parameters
            if type == "proportional" or type == "boosted-proportional":
                if bptAmount is None:
                    return json.dumps({
                        "error": f"BPT amount is required for {type} withdrawal",
                        "status": "error"
                    })
                    
                # Format BPT amount for API
                params["bptIn"] = {
                    "address": poolId,  # BPT token address is typically the pool address
                    "amount": bptAmount,
                    "decimals": 18  # BPT tokens typically have 18 decimals
                }
                
            elif type == "single-token-exact-in":
                if bptAmount is None:
                    return json.dumps({
                        "error": "BPT amount is required for single token exact in withdrawal",
                        "status": "error"
                    })
                if not tokenOut:
                    return json.dumps({
                        "error": "Token out symbol/address is required for single token withdrawal",
                        "status": "error"
                    })
                    
                # Format BPT amount for API
                params["bptIn"] = {
                    "address": poolId,  # BPT token address is typically the pool address
                    "amount": bptAmount,
                    "decimals": 18  # BPT tokens typically have 18 decimals
                }
                params["tokenOut"] = tokenOut
                
            elif type == "single-token-exact-out":
                if amountOut is None:
                    return json.dumps({
                        "error": "Amount out is required for single token exact out withdrawal",
                        "status": "error"
                    })
                if not tokenOut:
                    return json.dumps({
                        "error": "Token out symbol/address is required for single token withdrawal",
                        "status": "error"
                    })
                    
                # Format amount out for API
                params["amountOut"] = {
                    "address": tokenOut,
                    "amount": amountOut,
                    "decimals": tokenDecimals
                }
                
            elif type == "unbalanced":
                if not amountsOut or not isinstance(amountsOut, list) or len(amountsOut) == 0:
                    return json.dumps({
                        "error": "Amounts list with at least one token is required for unbalanced withdrawal",
                        "status": "error"
                    })
                    
                # Format each amount in the list
                formatted_amounts = []
                for amount in amountsOut:
                    if isinstance(amount, dict) and "token" in amount and "amount" in amount:
                        formatted_amounts.append({
                            "address": amount["token"],
                            "amount": amount["amount"],
                            "decimals": amount.get("decimals", tokenDecimals)
                        })
                    else:
                        return json.dumps({
                            "error": "Each amount must contain 'token' and 'amount' fields",
                            "status": "error"
                        })
                
                params["amountsOut"] = formatted_amounts
            
            logger.info(f"Using method: {action_name} with params: {params}")
            
            # Check if the method exists
            if not hasattr(self._agent.connection_manager.connections["beets"], action_name):
                return json.dumps({
                    "error": f"Method {action_name} not found in Beets connection", 
                    "status": "error"
                })
            
            # Call the appropriate method
            response = getattr(self._agent.connection_manager.connections["beets"], action_name)(**params)
            
            logger.info(f"Remove liquidity response: {response}")
            if not response:
                return json.dumps({"error": "Failed to remove liquidity", "status": "error"})

            return json.dumps({
                "status": "success", 
                "transaction": response,
                "message": f"Successfully removed {type} liquidity from pool {poolId}"
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

class BeetsPoolQueryTool(BaseTool):
    name: str = "beets_pool_query"
    description: str = """
    beets_pool_query: Get pool information from Beets
    
    Natural language examples:
    - "Find information about pool 0x1234...abcd on Beets"
    - "What are the recent events for my address 0xUser... on Beets pools?"
    - "Show me the first 5 pool events for user 0xUser... on Beets"
    
    Input should be a JSON string with one of the following:
    - poolId: Pool ID for specific pool information (or)
    - userAddress: User address for pool events, with optional 'first' and 'skip' parameters
    
    Examples:
    {"poolId": "0x1234...abcd"} 
    {"userAddress": "0xUser...", "first": 10, "skip": 0}
    """

    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def _run(self, poolId: str = None, userAddress: str = None, first: int = None, skip: int = None) -> str:
        try:
            logger.info(f"Pool query with poolId: {poolId}, userAddress: {userAddress}")
            
            # Check if beets connection exists
            if "beets" not in self._agent.connection_manager.connections:
                return json.dumps({"error": "Beets connection not configured", "status": "error"})
            
            if poolId is not None:
                logger.info(f"Getting pool information for ID: {poolId}")
                response = self._agent.connection_manager.connections["beets"].get_pool_by_id(poolId)
                return json.dumps({
                    "status": "success", 
                    "data": response,
                    "message": f"Found pool information for ID {poolId}"
                })
            elif userAddress is not None:
                logger.info(f"Getting pool events for user: {userAddress} (first: {first}, skip: {skip})")
                response = self._agent.connection_manager.connections["beets"].get_pool_events(
                    userAddress=userAddress,
                    first=first,
                    skip=skip
                )
                return json.dumps({
                    "status": "success", 
                    "data": response,
                    "message": f"Found pool events for user {userAddress}"
                })
            else:
                return json.dumps({"error": "Either poolId or userAddress must be provided", "status": "error"})

        except Exception as e:
            logger.error(f"Pool query failed: {str(e)}")
            logger.debug(traceback.format_exc())
            return json.dumps({"error": str(e), "status": "error"})

def get_beets_tools(agent) -> list:
    """Return a list of all Beets-related tools."""
    return [
        BeetsSwapTool(agent),
        BeetsAddLiquidityTool(agent),
        BeetsRemoveLiquidityTool(agent),
        BeetsTokenQueryTool(agent),
        BeetsPoolQueryTool(agent)
    ]
