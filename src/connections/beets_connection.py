import logging
import requests
from typing import Dict, Any, List, Optional, Union
import os
from requests.exceptions import RequestException, Timeout
from .base_connection import BaseConnection, Action, ActionParameter

logger = logging.getLogger("connections.beets_connection")

class BeetsConnectionError(Exception):
    """Base exception for Beets connection errors"""
    pass

class BeetsConnection(BaseConnection):
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base_url = config.get("api_base_url", "http://localhost:3000")
        self._initialize()

    def _initialize(self):
        """Initialize Beets connection"""
        logger.info(f"Initialized BEETS connection with API URL: {self.api_base_url}")

    def is_llm_provider(self) -> bool:
        return False

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the configuration parameters"""
        # Add API URL if not provided
        if "api_base_url" not in config:
            config["api_base_url"] = "http://localhost:3000"
        return config

    def register_actions(self) -> None:
        # Add Liquidity Actions (V3)
        self.actions['add_boosted_unbalanced_liquidity_v3'] = Action(
            name='add_boosted_unbalanced_liquidity_v3',
            description='Add boosted unbalanced liquidity to a v3 pool',
            parameters=[
                ActionParameter(name='amountsIn', type=list, required=True, description='Array of token amounts to add'),
                ActionParameter(name='poolId', type=str, required=True, description='Pool ID'),
                ActionParameter(name='slippage', type=float, required=True, description='Slippage tolerance'),
                ActionParameter(name='userAddress', type=str, required=True, description='User wallet address')
            ]
        )
        
        self.actions['add-boosted-proportional-liquidity-v3'] = Action(
            name='add-boosted-proportional-liquidity-v3',
            description='Add boosted proportional liquidity to a v3 pool',
            parameters=[
                ActionParameter(name='referenceAmount', type=Dict, required=True, description='Reference amount for proportional join'),
                ActionParameter(name='tokensIn', type=list, required=True, description='Array of token addresses'),
                ActionParameter(name='poolId', type=str, required=True, description='Pool ID'),
                ActionParameter(name='slippage', type=float, required=True, description='Slippage tolerance'),
                ActionParameter(name='userAddress', type=str, required=True, description='User wallet address')
            ]
        )
        
        self.actions['add-unbalanced-liquidity-v3'] = Action(
            name='add-unbalanced-liquidity-v3',
            description='Add unbalanced liquidity to a v3 pool',
            parameters=[
                ActionParameter(name='amountsIn', type=list, required=True, description='Array of token amounts to add'),
                ActionParameter(name='poolId', type=str, required=True, description='Pool ID'),
                ActionParameter(name='slippage', type=float, required=True, description='Slippage tolerance'),
                ActionParameter(name='userAddress', type=str, required=True, description='User wallet address')
            ]
        )
        
        self.actions['add-proportional-liquidity-v3'] = Action(
            name='add-proportional-liquidity-v3',
            description='Add proportional liquidity to a v3 pool',
            parameters=[
                ActionParameter(name='referenceAmount', type=Dict, required=True, description='Reference amount for proportional join'),
                ActionParameter(name='poolId', type=str, required=True, description='Pool ID'),
                ActionParameter(name='slippage', type=float, required=True, description='Slippage tolerance'),
                ActionParameter(name='userAddress', type=str, required=True, description='User wallet address')
            ]
        )
        
        self.actions['add-single-token-liquidity-v3'] = Action(
            name='add-single-token-liquidity-v3',
            description='Add single token liquidity to a v3 pool',
            parameters=[
                ActionParameter(name='bptOut', type=float, required=True, description='BPT tokens to receive'),
                ActionParameter(name='tokenIn', type=str, required=True, description='Token address to add'),
                ActionParameter(name='poolId', type=str, required=True, description='Pool ID'),
                ActionParameter(name='slippage', type=float, required=True, description='Slippage tolerance'),
                ActionParameter(name='userAddress', type=str, required=True, description='User wallet address')
            ]
        )
        
        # Add Liquidity Actions (V2)
        self.actions['add-unbalanced-liquidity-v2'] = Action(
            name='add-unbalanced-liquidity-v2',
            description='Add unbalanced liquidity to a v2 pool',
            parameters=[
                ActionParameter(name='amountsIn', type=list, required=True, description='Array of token amounts to add'),
                ActionParameter(name='poolId', type=str, required=True, description='Pool ID'),
                ActionParameter(name='slippage', type=float, required=True, description='Slippage tolerance'),
                ActionParameter(name='userAddress', type=str, required=True, description='User wallet address')
            ]
        )
        
        self.actions['add-proportional-liquidity-v2'] = Action(
            name='add-proportional-liquidity-v2',
            description='Add proportional liquidity to a v2 pool',
            parameters=[
                ActionParameter(name='referenceAmount', type=Dict, required=True, description='Reference amount for proportional join'),
                ActionParameter(name='poolId', type=str, required=True, description='Pool ID'),
                ActionParameter(name='slippage', type=float, required=True, description='Slippage tolerance'),
                ActionParameter(name='userAddress', type=str, required=True, description='User wallet address')
            ]
        )
        
        self.actions['add-single-token-liquidity-v2'] = Action(
            name='add-single-token-liquidity-v2',
            description='Add single token liquidity to a v2 pool',
            parameters=[
                ActionParameter(name='bptOut', type=float, required=True, description='BPT tokens to receive'),
                ActionParameter(name='tokenIn', type=str, required=True, description='Token address to add'),
                ActionParameter(name='poolId', type=str, required=True, description='Pool ID'),
                ActionParameter(name='slippage', type=float, required=True, description='Slippage tolerance'),
                ActionParameter(name='userAddress', type=str, required=True, description='User wallet address')
            ]
        )
        
        # Remove Liquidity Actions (V3)
        self.actions['remove-single-token-exact-out-v3'] = Action(
            name='remove-single-token-exact-out-v3',
            description='Remove liquidity from a v3 pool for exact token amount out',
            parameters=[
                ActionParameter(name='amountOut', type=float, required=True, description='Token amount to receive'),
                ActionParameter(name='poolId', type=str, required=True, description='Pool ID'),
                ActionParameter(name='slippage', type=float, required=True, description='Slippage tolerance'),
                ActionParameter(name='userAddress', type=str, required=True, description='User wallet address')
            ]
        )
        
        self.actions['remove-single-token-exact-in-v3'] = Action(
            name='remove-single-token-exact-in-v3',
            description='Remove exact BPT amount from a v3 pool',
            parameters=[
                ActionParameter(name='bptIn', type=float, required=True, description='BPT tokens to remove'),
                ActionParameter(name='tokenOut', type=str, required=True, description='Token address to receive'),
                ActionParameter(name='poolId', type=str, required=True, description='Pool ID'),
                ActionParameter(name='slippage', type=float, required=True, description='Slippage tolerance'),
                ActionParameter(name='userAddress', type=str, required=True, description='User wallet address')
            ]
        )
        
        self.actions['remove-proportional-liquidity-v3'] = Action(
            name='remove-proportional-liquidity-v3',
            description='Remove proportional liquidity from a v3 pool',
            parameters=[
                ActionParameter(name='bptIn', type=float, required=True, description='BPT tokens to remove'),
                ActionParameter(name='poolId', type=str, required=True, description='Pool ID'),
                ActionParameter(name='slippage', type=float, required=True, description='Slippage tolerance'),
                ActionParameter(name='userAddress', type=str, required=True, description='User wallet address')
            ]
        )
        
        # Add new action for Remove Boosted Proportional Liquidity (V3)
        self.actions['remove_boosted_proportional_liquidity_v3'] = Action(
            name='remove_boosted_proportional_liquidity_v3',
            description='Remove boosted proportional liquidity from a v3 pool',
            parameters=[
                ActionParameter(name='bptIn', type=Dict, required=True, description='BPT tokens to remove'),
                ActionParameter(name='poolId', type=str, required=True, description='Pool ID'),
                ActionParameter(name='slippage', type=float, required=True, description='Slippage tolerance'),
                ActionParameter(name='userAddress', type=str, required=True, description='User wallet address')
            ]
        )
        
        # Add new action for Remove Unbalanced Liquidity (V2)
        self.actions['remove_unbalanced_liquidity_v2'] = Action(
            name='remove_unbalanced_liquidity_v2',
            description='Remove unbalanced liquidity from a v2 pool',
            parameters=[
                ActionParameter(name='amountsOut', type=list, required=True, description='Token amounts to receive'),
                ActionParameter(name='poolId', type=str, required=True, description='Pool ID'),
                ActionParameter(name='slippage', type=float, required=True, description='Slippage tolerance'),
                ActionParameter(name='userAddress', type=str, required=True, description='User wallet address')
            ]
        )
        
        # Remove Liquidity Actions (V2)
        self.actions['remove-single-token-exact-out-v2'] = Action(
            name='remove-single-token-exact-out-v2',
            description='Remove liquidity from a v2 pool for exact token amount out',
            parameters=[
                ActionParameter(name='amountOut', type=float, required=True, description='Token amount to receive'),
                ActionParameter(name='poolId', type=str, required=True, description='Pool ID'),
                ActionParameter(name='slippage', type=float, required=True, description='Slippage tolerance'),
                ActionParameter(name='userAddress', type=str, required=True, description='User wallet address')
            ]
        )
        
        self.actions['remove-single-token-exact-in-v2'] = Action(
            name='remove-single-token-exact-in-v2',
            description='Remove exact BPT amount from a v2 pool',
            parameters=[
                ActionParameter(name='bptIn', type=float, required=True, description='BPT tokens to remove'),
                ActionParameter(name='tokenOut', type=str, required=True, description='Token address to receive'),
                ActionParameter(name='poolId', type=str, required=True, description='Pool ID'),
                ActionParameter(name='slippage', type=float, required=True, description='Slippage tolerance'),
                ActionParameter(name='userAddress', type=str, required=True, description='User wallet address')
            ]
        )
        
        self.actions['remove-proportional-liquidity-v2'] = Action(
            name='remove-proportional-liquidity-v2',
            description='Remove proportional liquidity from a v2 pool',
            parameters=[
                ActionParameter(name='bptIn', type=float, required=True, description='BPT tokens to remove'),
                ActionParameter(name='poolId', type=str, required=True, description='Pool ID'),
                ActionParameter(name='slippage', type=float, required=True, description='Slippage tolerance'),
                ActionParameter(name='userAddress', type=str, required=True, description='User wallet address')
            ]
        )
        
        # Swap Action
        self.actions['swap'] = Action(
            name='swap',
            description='Swap tokens using Beets',
            parameters=[
                ActionParameter(name='tokenIn', type=str, required=True, description='Token address to swap from'),
                ActionParameter(name='tokenOut', type=str, required=True, description='Token address to swap to'),
                ActionParameter(name='slippage', type=float, required=True, description='Slippage tolerance'),
                ActionParameter(name='userAddress', type=str, required=True, description='User wallet address')
            ]
        )

        # Query actions for tokens and pools
        self.actions['get_token_by_symbol'] = Action(
            name='get_token_by_symbol',
            description='Get token information by symbol',
            parameters=[
                ActionParameter(name='symbol', type=str, required=True, description='Token symbol or name')
            ]
        )
        
        self.actions['get_token_by_address'] = Action(
            name='get_token_by_address',
            description='Get token information by address',
            parameters=[
                ActionParameter(name='address', type=str, required=True, description='Token contract address')
            ]
        )
        
        self.actions['get_pool_events'] = Action(
            name='get_pool_events',
            description='Get pool events for a user',
            parameters=[
                ActionParameter(name='userAddress', type=str, required=True, description='User wallet address'),
                ActionParameter(name='first', type=int, required=False, description='Maximum number of events'),
                ActionParameter(name='skip', type=int, required=False, description='Number of events to skip')
            ]
        )
        
        self.actions['get_pool_by_id'] = Action(
            name='get_pool_by_id',
            description='Get pool information by ID',
            parameters=[
                ActionParameter(name='poolId', type=str, required=True, description='Pool ID')
            ]
        )

        # Add get_pools query action
        self.actions['get_pools'] = Action(
            name='get_pools',
            description='Get all pools with optional filtering and sorting',
            parameters=[
                ActionParameter(name='userAddress', type=str, required=False, description='User wallet address'),
                ActionParameter(name='first', type=int, required=False, description='Maximum number of pools to return'),
                ActionParameter(name='orderBy', type=str, required=False, description='Field to sort by'),
                ActionParameter(name='orderDirection', type=str, required=False, description='Sort direction (asc/desc)'),
                ActionParameter(name='skip', type=int, required=False, description='Number of pools to skip'),
                ActionParameter(name='textSearch', type=str, required=False, description='Text to search for in pool name/symbol')
            ]
        )

    def configure(self) -> bool:
        """Configure the Beets connection"""
        try:
            self._initialize()
            return True
        except Exception as e:
            logger.error(f"Failed to configure Beets connection: {str(e)}")
            return False

    def is_configured(self, verbose: bool = False) -> bool:
        """Check if the connection is properly configured"""
        return True

    def _format_token_amount(self, token_address: str, amount: Union[float, int, str], decimals: int = 18) -> Dict:
        """Format token amount in API expected format"""
        # Convert amount to string to handle large integers
        if isinstance(amount, (float, int)):
            raw_amount = str(int(amount * (10 ** decimals)))
        else:
            raw_amount = amount
            
        return {
            "address": token_address,
            "rawAmount": raw_amount,
            "decimals": decimals
        }

    # Add Liquidity Methods (V3)
    def add_boosted_unbalanced_liquidity_v3(self, amountsIn: List, poolId: str, slippage: float, userAddress: str) -> Dict:
        """Add boosted unbalanced liquidity to a v3 pool"""
        try:
            # Format amounts according to API expectations
            formatted_amounts = []
            for amount in amountsIn:
                if isinstance(amount, dict) and "address" in amount and "amount" in amount:
                    formatted_amounts.append(
                        self._format_token_amount(
                            amount["address"], 
                            amount["amount"],
                            amount.get("decimals", 18)
                        )
                    )
                else:
                    logger.error(f"Invalid amount format: {amount}")
                    raise ValueError("Each amount must contain 'address' and 'amount' fields")
            
            payload = {
                "amountsIn": formatted_amounts,
                "poolId": poolId,
                "slippage": str(slippage),
                "userAddress": userAddress
            }
            
            headers = self._get_headers()
            response = requests.post(f"{self.api_base_url}/api/add-liquidity/v3/boosted/unbalanced", 
                                     json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_details = response.json()  # Extract JSON error message
                error_message = error_details.get("error", "Unknown error")
            except ValueError:
                error_message = response.text  # Fallback to raw response text

            logger.error(f"HTTP error: {http_err} | Response: {error_message}")
            raise BeetsConnectionError(f"HTTP error: {http_err} | Response: {error_message}")

            
        except Exception as e:
            raise BeetsConnectionError(f"Failed to add boosted unbalanced liquidity to v3 pool: {str(e)}")
    
    def add_boosted_proportional_liquidity_v3(self, referenceAmount: Dict, tokensIn: List[str], 
                                              poolId: str, slippage: float, userAddress: str) -> Dict:
        """Add boosted proportional liquidity to a v3 pool"""
        try:
            # Format reference amount according to API expectations
            if isinstance(referenceAmount, dict) and "address" in referenceAmount and "amount" in referenceAmount:
                formatted_ref_amount = self._format_token_amount(
                    referenceAmount["address"],
                    referenceAmount["amount"],
                    referenceAmount.get("decimals", 18)
                )
            else:
                raise ValueError("referenceAmount must contain 'address' and 'amount' fields")
                
            payload = {
                "referenceAmount": formatted_ref_amount,
                "tokensIn": tokensIn,
                "poolId": poolId,
                "slippage": str(slippage),
                "userAddress": userAddress
            }
            
            headers = self._get_headers()
            response = requests.post(f"{self.api_base_url}/api/add-liquidity/v3/boosted/proportional", 
                                     json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_details = response.json()  # Extract JSON error message
                error_message = error_details.get("error", "Unknown error")
            except ValueError:
                error_message = response.text  # Fallback to raw response text

            logger.error(f"HTTP error: {http_err} | Response: {error_message}")
            raise BeetsConnectionError(f"HTTP error: {http_err} | Response: {error_message}")

            
        except Exception as e:
            raise BeetsConnectionError(f"Failed to add boosted proportional liquidity to v3 pool: {str(e)}")
    
    def add_unbalanced_liquidity_v3(self, amountsIn: List, poolId: str, slippage: float, userAddress: str) -> Dict:
        """Add unbalanced liquidity to a v3 pool"""
        try:
            # Format amounts according to API expectations
            formatted_amounts = []
            for amount in amountsIn:
                if isinstance(amount, dict) and "address" in amount and "amount" in amount:
                    formatted_amounts.append(
                        self._format_token_amount(
                            amount["address"], 
                            amount["amount"],
                            amount.get("decimals", 18)
                        )
                    )
                else:
                    raise ValueError("Each amount must contain 'address' and 'amount' fields")
            
            payload = {
                "amountsIn": formatted_amounts,
                "poolId": poolId,
                "slippage": str(slippage),
                "userAddress": userAddress
            }
            
            headers = self._get_headers()
            response = requests.post(f"{self.api_base_url}/api/add-liquidity/v3/unbalanced", 
                                     json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_details = response.json()  # Extract JSON error message
                error_message = error_details.get("error", "Unknown error")
            except ValueError:
                error_message = response.text  # Fallback to raw response text

            logger.error(f"HTTP error: {http_err} | Response: {error_message}")
            raise BeetsConnectionError(f"HTTP error: {http_err} | Response: {error_message}")

            
        except Exception as e:
            raise BeetsConnectionError(f"Failed to add unbalanced liquidity to v3 pool: {str(e)}")
    
    def add_proportional_liquidity_v3(self, referenceAmount: Dict, poolId: str, slippage: float, userAddress: str) -> Dict:
        """Add proportional liquidity to a v3 pool"""
        try:
            # Format reference amount according to API expectations
            if isinstance(referenceAmount, dict) and "address" in referenceAmount and "amount" in referenceAmount:
                formatted_ref_amount = self._format_token_amount(
                    referenceAmount["address"],
                    referenceAmount["amount"],
                    referenceAmount.get("decimals", 18)
                )
            else:
                raise ValueError("referenceAmount must contain 'address' and 'amount' fields")
                
            payload = {
                "referenceAmount": formatted_ref_amount,
                "poolId": poolId,
                "slippage": str(slippage),
                "userAddress": userAddress
            }
            
            headers = self._get_headers()
            response = requests.post(f"{self.api_base_url}/api/add-liquidity/v3/proportional", 
                                     json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_details = response.json()  # Extract JSON error message
                error_message = error_details.get("error", "Unknown error")
            except ValueError:
                error_message = response.text  # Fallback to raw response text

            logger.error(f"HTTP error: {http_err} | Response: {error_message}")
            raise BeetsConnectionError(f"HTTP error: {http_err} | Response: {error_message}")

            
        except Exception as e:
            raise BeetsConnectionError(f"Failed to add proportional liquidity to v3 pool: {str(e)}")
    
    def add_single_token_liquidity_v3(self, bptOut: Dict, tokenIn: str, poolId: str, slippage: float, userAddress: str) -> Dict:
        """Add single token liquidity to a v3 pool"""
        try:
            # Format bptOut amount according to API expectations
            if isinstance(bptOut, dict) and "amount" in bptOut:
                # For BPT tokens, address is typically the pool address
                bpt_address = bptOut.get("address", poolId)
                formatted_bpt_out = self._format_token_amount(
                    bpt_address,
                    bptOut["amount"],
                    bptOut.get("decimals", 18)
                )
            else:
                raise ValueError("bptOut must contain at least an 'amount' field")
                
            payload = {
                "bptOut": formatted_bpt_out,
                "tokenIn": tokenIn,
                "poolId": poolId,
                "slippage": str(slippage),
                "userAddress": userAddress
            }
            
            headers = self._get_headers()
            response = requests.post(f"{self.api_base_url}/api/add-liquidity/v3/single-token", 
                                     json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_details = response.json()  # Extract JSON error message
                error_message = error_details.get("error", "Unknown error")
            except ValueError:
                error_message = response.text  # Fallback to raw response text

            logger.error(f"HTTP error: {http_err} | Response: {error_message}")
            raise BeetsConnectionError(f"HTTP error: {http_err} | Response: {error_message}")

            
        except Exception as e:
            raise BeetsConnectionError(f"Failed to add single token liquidity to v3 pool: {str(e)}")
    
    # Add Liquidity Methods (V2)
    def add_unbalanced_liquidity_v2(self, amountsIn: List, poolId: str, slippage: float, userAddress: str) -> Dict:
        """Add unbalanced liquidity to a v2 pool"""
        try:
            # Format amounts according to API expectations
            formatted_amounts = []
            for amount in amountsIn:
                if isinstance(amount, dict) and "address" in amount and "amount" in amount:
                    formatted_amounts.append(
                        self._format_token_amount(
                            amount["address"], 
                            amount["amount"],
                            amount.get("decimals", 18)
                        )
                    )
                else:
                    raise ValueError("Each amount must contain 'address' and 'amount' fields")
            
            payload = {
                "amountsIn": formatted_amounts,
                "poolId": poolId,
                "slippage": str(slippage),
                "userAddress": userAddress
            }
            
            headers = self._get_headers()
            response = requests.post(f"{self.api_base_url}/api/add-liquidity/v2/unbalanced", 
                                     json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_details = response.json()  # Extract JSON error message
                error_message = error_details.get("error", "Unknown error")
            except ValueError:
                error_message = response.text  # Fallback to raw response text

            logger.error(f"HTTP error: {http_err} | Response: {error_message}")
            raise BeetsConnectionError(f"HTTP error: {http_err} | Response: {error_message}")

            
        except Exception as e:
            raise BeetsConnectionError(f"Failed to add unbalanced liquidity to v2 pool: {str(e)}")
    
    def add_proportional_liquidity_v2(self, referenceAmount: Dict, poolId: str, slippage: float, userAddress: str) -> Dict:
        """Add proportional liquidity to a v2 pool"""
        try:
            # Format reference amount according to API expectations
            if isinstance(referenceAmount, dict) and "address" in referenceAmount and "amount" in referenceAmount:
                formatted_ref_amount = self._format_token_amount(
                    referenceAmount["address"],
                    referenceAmount["amount"],
                    referenceAmount.get("decimals", 18)
                )
            else:
                raise ValueError("referenceAmount must contain 'address' and 'amount' fields")
                
            payload = {
                "referenceAmount": formatted_ref_amount,
                "poolId": poolId,
                "slippage": str(slippage),
                "userAddress": userAddress
            }
            
            headers = self._get_headers()
            response = requests.post(f"{self.api_base_url}/api/add-liquidity/v2/proportional", 
                                     json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_details = response.json()  # Extract JSON error message
                error_message = error_details.get("error", "Unknown error")
            except ValueError:
                error_message = response.text  # Fallback to raw response text

            logger.error(f"HTTP error: {http_err} | Response: {error_message}")
            raise BeetsConnectionError(f"HTTP error: {http_err} | Response: {error_message}")

            
        except Exception as e:
            raise BeetsConnectionError(f"Failed to add proportional liquidity to v2 pool: {str(e)}")
    
    def add_single_token_liquidity_v2(self, bptOut: Dict, tokenIn: str, poolId: str, slippage: float, userAddress: str) -> Dict:
        """Add single token liquidity to a v2 pool"""
        try:
            # Format bptOut amount according to API expectations
            if isinstance(bptOut, dict) and "amount" in bptOut:
                bpt_address = bptOut.get("address", poolId)
                formatted_bpt_out = self._format_token_amount(
                    bpt_address,
                    bptOut["amount"],
                    bptOut.get("decimals", 18)
                )
            else:
                raise ValueError("bptOut must contain at least an 'amount' field")
                
            payload = {
                "bptOut": formatted_bpt_out,
                "tokenIn": tokenIn,
                "poolId": poolId,
                "slippage": str(slippage),
                "userAddress": userAddress
            }
            
            headers = self._get_headers()
            response = requests.post(f"{self.api_base_url}/api/add-liquidity/v2/single-token", 
                                     json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_details = response.json()  # Extract JSON error message
                error_message = error_details.get("error", "Unknown error")
            except ValueError:
                error_message = response.text  # Fallback to raw response text

            logger.error(f"HTTP error: {http_err} | Response: {error_message}")
            raise BeetsConnectionError(f"HTTP error: {http_err} | Response: {error_message}")

            
        except Exception as e:
            raise BeetsConnectionError(f"Failed to add single token liquidity to v2 pool: {str(e)}")
    
    # Remove Liquidity Methods (V3)
    def remove_single_token_exact_out_v3(self, amountOut: Dict, poolId: str, slippage: float, userAddress: str) -> Dict:
        """Remove liquidity from a v3 pool for exact token amount out"""
        try:
            # Format amountOut according to API expectations
            if isinstance(amountOut, dict) and "address" in amountOut and "amount" in amountOut:
                formatted_amount_out = self._format_token_amount(
                    amountOut["address"],
                    amountOut["amount"],
                    amountOut.get("decimals", 18)
                )
            else:
                raise ValueError("amountOut must contain 'address' and 'amount' fields")
                
            payload = {
                "amountOut": formatted_amount_out,
                "poolId": poolId,
                "slippage": str(slippage),
                "userAddress": userAddress
            }
            
            headers = self._get_headers()
            response = requests.post(f"{self.api_base_url}/api/remove-liquidity/v3/single-token-exact-out", 
                                     json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_details = response.json()  # Extract JSON error message
                error_message = error_details.get("error", "Unknown error")
            except ValueError:
                error_message = response.text  # Fallback to raw response text

            logger.error(f"HTTP error: {http_err} | Response: {error_message}")
            raise BeetsConnectionError(f"HTTP error: {http_err} | Response: {error_message}")

            
        except Exception as e:
            raise BeetsConnectionError(f"Failed to remove exact out liquidity from v3 pool: {str(e)}")
    
    def remove_single_token_exact_in_v3(self, bptIn: Dict, tokenOut: str, poolId: str, slippage: float, userAddress: str) -> Dict:
        """Remove exact BPT amount from a v3 pool"""
        try:
            # Format bptIn according to API expectations
            if isinstance(bptIn, dict) and "amount" in bptIn:
                bpt_address = bptIn.get("address", poolId)
                formatted_bpt_in = self._format_token_amount(
                    bpt_address,
                    bptIn["amount"],
                    bptIn.get("decimals", 18)
                )
            else:
                raise ValueError("bptIn must contain at least an 'amount' field")
                
            payload = {
                "bptIn": formatted_bpt_in,
                "tokenOut": tokenOut,
                "poolId": poolId,
                "slippage": str(slippage),
                "userAddress": userAddress
            }
            
            headers = self._get_headers()
            response = requests.post(f"{self.api_base_url}/api/remove-liquidity/v3/single-token-exact-in", 
                                     json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_details = response.json()  # Extract JSON error message
                error_message = error_details.get("error", "Unknown error")
            except ValueError:
                error_message = response.text  # Fallback to raw response text

            logger.error(f"HTTP error: {http_err} | Response: {error_message}")
            raise BeetsConnectionError(f"HTTP error: {http_err} | Response: {error_message}")

            
        except Exception as e:
            raise BeetsConnectionError(f"Failed to remove exact in liquidity from v3 pool: {str(e)}")
    
    def remove_proportional_liquidity_v3(self, bptIn: Dict, poolId: str, slippage: float, userAddress: str) -> Dict:
        """Remove proportional liquidity from a v3 pool"""
        try:
            # Format bptIn according to API expectations
            if isinstance(bptIn, dict) and "amount" in bptIn:
                bpt_address = bptIn.get("address", poolId)
                formatted_bpt_in = self._format_token_amount(
                    bpt_address,
                    bptIn["amount"],
                    bptIn.get("decimals", 18)
                )
            else:
                raise ValueError("bptIn must contain at least an 'amount' field")
                
            payload = {
                "bptIn": formatted_bpt_in,
                "poolId": poolId,
                "slippage": str(slippage),
                "userAddress": userAddress
            }
            
            headers = self._get_headers()
            response = requests.post(f"{self.api_base_url}/api/remove-liquidity/v3/proportional", 
                                     json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_details = response.json()  # Extract JSON error message
                error_message = error_details.get("error", "Unknown error")
            except ValueError:
                error_message = response.text  # Fallback to raw response text

            logger.error(f"HTTP error: {http_err} | Response: {error_message}")
            raise BeetsConnectionError(f"HTTP error: {http_err} | Response: {error_message}")

            
        except Exception as e:
            raise BeetsConnectionError(f"Failed to remove proportional liquidity from v3 pool: {str(e)}")
    
    # Add method for Remove Boosted Proportional Liquidity (V3)
    def remove_boosted_proportional_liquidity_v3(self, bptIn: Dict, poolId: str, slippage: float, userAddress: str) -> Dict:
        """Remove boosted proportional liquidity from a v3 pool"""
        try:
            # Format bptIn according to API expectations
            if isinstance(bptIn, dict) and "amount" in bptIn:
                bpt_address = bptIn.get("address", poolId)
                formatted_bpt_in = self._format_token_amount(
                    bpt_address,
                    bptIn["amount"],
                    bptIn.get("decimals", 18)
                )
            else:
                raise ValueError("bptIn must contain at least an 'amount' field")
                
            payload = {
                "bptIn": formatted_bpt_in,
                "poolId": poolId,
                "slippage": str(slippage),
                "userAddress": userAddress
            }
            
            headers = self._get_headers()
            response = requests.post(f"{self.api_base_url}/api/remove-liquidity/v3/boosted/proportional", 
                                     json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_details = response.json()  # Extract JSON error message
                error_message = error_details.get("error", "Unknown error")
            except ValueError:
                error_message = response.text  # Fallback to raw response text

            logger.error(f"HTTP error: {http_err} | Response: {error_message}")
            raise BeetsConnectionError(f"HTTP error: {http_err} | Response: {error_message}")

            
        except Exception as e:
            raise BeetsConnectionError(f"Failed to remove boosted proportional liquidity from v3 pool: {str(e)}")
    
    # Remove Liquidity Methods (V2)
    def remove_single_token_exact_out_v2(self, amountOut: Dict, poolId: str, slippage: float, userAddress: str) -> Dict:
        """Remove liquidity from a v2 pool for exact token amount out"""
        try:
            # Format amountOut according to API expectations
            if isinstance(amountOut, dict) and "address" in amountOut and "amount" in amountOut:
                formatted_amount_out = self._format_token_amount(
                    amountOut["address"],
                    amountOut["amount"],
                    amountOut.get("decimals", 18)
                )
            else:
                raise ValueError("amountOut must contain 'address' and 'amount' fields")
                
            payload = {
                "amountOut": formatted_amount_out,
                "poolId": poolId,
                "slippage": str(slippage),
                "userAddress": userAddress
            }
            
            headers = self._get_headers()
            response = requests.post(f"{self.api_base_url}/api/remove-liquidity/v2/single-token-exact-out", 
                                     json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_details = response.json()  # Extract JSON error message
                error_message = error_details.get("error", "Unknown error")
            except ValueError:
                error_message = response.text  # Fallback to raw response text

            logger.error(f"HTTP error: {http_err} | Response: {error_message}")
            raise BeetsConnectionError(f"HTTP error: {http_err} | Response: {error_message}")

            
        except Exception as e:
            raise BeetsConnectionError(f"Failed to remove exact out liquidity from v2 pool: {str(e)}")
    
    def remove_single_token_exact_in_v2(self, bptIn: Dict, tokenOut: str, poolId: str, slippage: float, userAddress: str) -> Dict:
        """Remove exact BPT amount from a v2 pool"""
        try:
            # Format bptIn according to API expectations
            if isinstance(bptIn, dict) and "amount" in bptIn:
                bpt_address = bptIn.get("address", poolId)
                formatted_bpt_in = self._format_token_amount(
                    bpt_address,
                    bptIn["amount"],
                    bptIn.get("decimals", 18)
                )
            else:
                raise ValueError("bptIn must contain at least an 'amount' field")
                
            payload = {
                "bptIn": formatted_bpt_in,
                "tokenOut": tokenOut,
                "poolId": poolId,
                "slippage": str(slippage),
                "userAddress": userAddress
            }
            
            headers = self._get_headers()
            response = requests.post(f"{self.api_base_url}/api/remove-liquidity/v2/single-token-exact-in", 
                                     json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_details = response.json()  # Extract JSON error message
                error_message = error_details.get("error", "Unknown error")
            except ValueError:
                error_message = response.text  # Fallback to raw response text

            logger.error(f"HTTP error: {http_err} | Response: {error_message}")
            raise BeetsConnectionError(f"HTTP error: {http_err} | Response: {error_message}")

            
        except Exception as e:
            raise BeetsConnectionError(f"Failed to remove exact in liquidity from v2 pool: {str(e)}")
    
    def remove_proportional_liquidity_v2(self, bptIn: Dict, poolId: str, slippage: float, userAddress: str) -> Dict:
        """Remove proportional liquidity from a v2 pool"""
        try:
            # Format bptIn according to API expectations
            if isinstance(bptIn, dict) and "amount" in bptIn:
                bpt_address = bptIn.get("address", poolId)
                formatted_bpt_in = self._format_token_amount(
                    bpt_address,
                    bptIn["amount"],
                    bptIn.get("decimals", 18)
                )
            else:
                raise ValueError("bptIn must contain at least an 'amount' field")
                
            payload = {
                "bptIn": formatted_bpt_in,
                "poolId": poolId,
                "slippage": str(slippage),
                "userAddress": userAddress
            }
            
            headers = self._get_headers()
            response = requests.post(f"{self.api_base_url}/api/remove-liquidity/v2/proportional", 
                                     json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_details = response.json()  # Extract JSON error message
                error_message = error_details.get("error", "Unknown error")
            except ValueError:
                error_message = response.text  # Fallback to raw response text

            logger.error(f"HTTP error: {http_err} | Response: {error_message}")
            raise BeetsConnectionError(f"HTTP error: {http_err} | Response: {error_message}")

            
        except Exception as e:
            raise BeetsConnectionError(f"Failed to remove proportional liquidity from v2 pool: {str(e)}")
    
    # Add method for Remove Unbalanced Liquidity (V2)
    def remove_unbalanced_liquidity_v2(self, amountsOut: List, poolId: str, slippage: float, userAddress: str) -> Dict:
        """Remove unbalanced liquidity from a v2 pool with specific token amounts"""
        try:
            # Format amounts according to API expectations
            formatted_amounts = []
            for amount in amountsOut:
                if isinstance(amount, dict) and "address" in amount and "amount" in amount:
                    formatted_amounts.append(
                        self._format_token_amount(
                            amount["address"], 
                            amount["amount"],
                            amount.get("decimals", 18)
                        )
                    )
                else:
                    raise ValueError("Each amount must contain 'address' and 'amount' fields")
            
            payload = {
                "amountsOut": formatted_amounts,
                "poolId": poolId,
                "slippage": str(slippage),
                "userAddress": userAddress
            }
            
            headers = self._get_headers()
            response = requests.post(f"{self.api_base_url}/api/remove-liquidity/v2/unbalanced", 
                                     json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_details = response.json()  # Extract JSON error message
                error_message = error_details.get("error", "Unknown error")
            except ValueError:
                error_message = response.text  # Fallback to raw response text

            logger.error(f"HTTP error: {http_err} | Response: {error_message}")
            raise BeetsConnectionError(f"HTTP error: {http_err} | Response: {error_message}")

            
        except Exception as e:
            raise BeetsConnectionError(f"Failed to remove unbalanced liquidity from v2 pool: {str(e)}")

    # New Token Query Methods
    def get_token_by_symbol(self, symbol: str) -> Dict:
        """Get token information by symbol"""
        try:
            headers = self._get_headers()
            response = requests.get(f"{self.api_base_url}/api/queries/token/symbol/{symbol}", 
                                   headers=headers)
            response.raise_for_status()
            logger.info(response)
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_details = response.json()  # Extract JSON error message
                error_message = error_details.get("error", "Unknown error")
            except ValueError:
                error_message = response.text  # Fallback to raw response text

            logger.error(f"HTTP error: {http_err} | Response: {error_message}")
            raise BeetsConnectionError(f"HTTP error: {http_err} | Response: {error_message}")

        except Exception as e:
            logger.info(str(e))
            raise BeetsConnectionError(f"Failed to get token by symbol: {str(e)}")
    
    def get_token_by_address(self, address: str) -> Dict:
        """Get token information by address"""
        try:
            headers = self._get_headers()
            response = requests.get(f"{self.api_base_url}/api/queries/token/address/{address}", 
                                   headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_details = response.json()  # Extract JSON error message
                error_message = error_details.get("error", "Unknown error")
            except ValueError:
                error_message = response.text  # Fallback to raw response text

            logger.error(f"HTTP error: {http_err} | Response: {error_message}")
            raise BeetsConnectionError(f"HTTP error: {http_err} | Response: {error_message}")

        except Exception as e:
            raise BeetsConnectionError(f"Failed to get token by address: {str(e)}")
    
    # New Pool Query Methods
    def get_pool_events(self, userAddress: str, first: Optional[int] = None, skip: Optional[int] = None) -> Dict:
        """Get pool events for a user"""
        try:
            params = {}
            if first is not None:
                params['first'] = first
            if skip is not None:
                params['skip'] = skip
                
            headers = self._get_headers()
            response = requests.get(f"{self.api_base_url}/api/queries/pool/events/{userAddress}", 
                                   params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_details = response.json()  # Extract JSON error message
                error_message = error_details.get("error", "Unknown error")
            except ValueError:
                error_message = response.text  # Fallback to raw response text

            logger.error(f"HTTP error: {http_err} | Response: {error_message}")
            raise BeetsConnectionError(f"HTTP error: {http_err} | Response: {error_message}")

        except Exception as e:
            raise BeetsConnectionError(f"Failed to get pool events: {str(e)}")
    
    def get_pool_by_id(self, poolId: str) -> Dict:
        """Get pool information by ID"""
        try:
            headers = self._get_headers()
            response = requests.get(f"{self.api_base_url}/api/queries/pool/{poolId}", 
                                   headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_details = response.json()  # Extract JSON error message
                error_message = error_details.get("error", "Unknown error")
            except ValueError:
                error_message = response.text  # Fallback to raw response text

            logger.error(f"HTTP error: {http_err} | Response: {error_message}")
            raise BeetsConnectionError(f"HTTP error: {http_err} | Response: {error_message}")

        except Exception as e:
            raise BeetsConnectionError(f"Failed to get pool by ID: {str(e)}")
        
    
    # Add get_pools method
    def get_pools(self, userAddress: Optional[str] = None, first: Optional[int] = None, 
                  orderBy: Optional[str] = None, orderDirection: Optional[str] = None, 
                  skip: Optional[int] = None, textSearch: Optional[str] = None) -> Dict:
        """Get all pools with optional filtering and sorting"""
        try:
            params = {
                "userAddress": userAddress,
                "first": first,
                "orderBy": orderBy,
                "orderDirection": orderDirection,
                "skip": skip,
                "textSearch": textSearch
            }
            
            headers = self._get_headers()
            response = requests.get(f"{self.api_base_url}/api/queries/pools", 
                                  params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_details = response.json()  # Extract JSON error message
                error_message = error_details.get("error", "Unknown error")
            except ValueError:
                error_message = response.text  # Fallback to raw response text

            logger.error(f"HTTP error: {http_err} | Response: {error_message}")
            raise BeetsConnectionError(f"HTTP error: {http_err} | Response: {error_message}")

        except Exception as e:
            logger.error(f"Failed to get pools: {str(e)}")
            raise BeetsConnectionError(f"Failed to get pools: {str(e)}")
    
    def swap(self, tokenIn: str, tokenOut: str, slippage: float = 0.005, userAddress: str = None) -> Dict:
        """Swap tokens using Beets"""
        try:
            payload = {
                "tokenIn": tokenIn,
                "tokenOut": tokenOut,
                "userAddress": userAddress,
                "slippage": str(slippage)
            }
            
            headers = self._get_headers()
            response = requests.post(f"{self.api_base_url}/api/swap", 
                                     json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            try:
                error_details = response.json()  # Extract JSON error message
                error_message = error_details.get("error", "Unknown error")
            except ValueError:
                error_message = response.text  # Fallback to raw response text

            logger.error(f"HTTP error: {http_err} | Response: {error_message}")
            raise BeetsConnectionError(f"HTTP error: {http_err} | Response: {error_message}")

            
        except Exception as e:
            raise BeetsConnectionError(f"Failed to swap tokens: {str(e)}")

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        return headers

    def perform_action(self, action_name: str, **kwargs) -> Any:
        """Execute a Beets action with validation"""
        if action_name not in self.actions:
            raise KeyError(f"Unknown action: {action_name}")

        if not self.is_configured(verbose=True):
            raise BeetsConnectionError("Beets is not properly configured")

        action = self.actions[action_name]
        validation_errors = action.validate_params(kwargs)
        if validation_errors:
            raise BeetsConnectionError(f"Invalid parameters: {', '.join(validation_errors)}")

        method = getattr(self, action_name)
        return method(**kwargs)

