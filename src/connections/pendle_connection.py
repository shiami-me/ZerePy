import logging
import requests
from typing import Dict, List, Any, Optional, Union
from .base_connection import BaseConnection, Action, ActionParameter

logger = logging.getLogger("connections.pendle_connection")

class PendleConnectionError(Exception):
    """Base exception for Pendle connection errors"""
    pass

class PendleConnection(BaseConnection):
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_url = "https://api-v2.pendle.finance"
        self.sdk_url = f"{self.api_url}/core/"
        self._initialize()

    def _initialize(self):
        """Initialize Pendle connection"""
        try:
            # No API key needed for Pendle public APIs
            pass
        except Exception as e:
            raise PendleConnectionError(f"Failed to initialize Pendle connection: {str(e)}")

    def is_llm_provider(self) -> bool:
        return False

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the configuration parameters"""
        return config

    def configure(self) -> bool:
        """Configure the Pendle connection"""
        try:
            self._initialize()
            return True
        except Exception as e:
            logger.error(f"Failed to configure Pendle connection: {str(e)}")
            return False

    def is_configured(self, verbose: bool = False) -> bool:
        """Check if the connection is properly configured"""
        try:
            # No special configuration needed for Pendle
            return True
        except Exception as e:
            if verbose:
                logger.error(f"Configuration check failed: {str(e)}")
            return False
            
    def register_actions(self) -> None:
        self.actions['get_markets'] = Action(
            name='get_markets',
            description='Get all available Pendle markets for a given chain ID',
            parameters=[
            ]
        )
        self.actions['get_assets'] = Action(
            name='get_assets',
            description='Get all available Pendle assets for a given chain ID',
            parameters=[
            ]
        )
        self.actions['parse_token_id'] = Action(
            name='parse_token_id',
            description='Parse token ID in format "146-0x..." to get the clean address',
            parameters=[
                ActionParameter(name='token_id', type=str, required=True, description='Token ID in format "chainId-address"'),
            ]
        )
        
        # Add liquidity actions
        self.actions['add_liquidity'] = Action(
            name='add_liquidity',
            description='Add liquidity to a Pendle market with a single token input',
            parameters=[
                ActionParameter(name='market_address', type=str, required=True, description='Pendle market address'),
                ActionParameter(name='token_in', type=str, required=True, description='Input token address'),
                ActionParameter(name='amount_in', type=str, required=True, description='Amount of input token'),
                ActionParameter(name='slippage', type=float, required=False, description='Slippage tolerance (default: 0.01)'),
                ActionParameter(name='zpi', type=bool, required=False, description='Zero Price Impact flag (default: False)'),
                ActionParameter(name='user_address', type=str, required=True, description='User address'),
                ActionParameter(name='enable_aggregator', type=bool, required=False, description='Enable aggregator (default: False)'),
            ]
        )
        self.actions['add_liquidity_dual'] = Action(
            name='add_liquidity_dual',
            description='Add liquidity to a Pendle market with dual inputs (token and PT)',
            parameters=[
                ActionParameter(name='market_address', type=str, required=True, description='Pendle market address'),
                ActionParameter(name='token_in', type=str, required=True, description='Input token address'),
                ActionParameter(name='amount_token_in', type=str, required=True, description='Amount of input token'),
                ActionParameter(name='amount_pt_in', type=str, required=True, description='Amount of principal token input'),
                ActionParameter(name='slippage', type=float, required=False, description='Slippage tolerance (default: 0.01)'),
                ActionParameter(name='user_address', type=str, required=True, description='User address'),
                ActionParameter(name='enable_aggregator', type=bool, required=False, description='Enable aggregator (default: False)'),
            ]
        )
        self.actions['remove_liquidity'] = Action(
            name='remove_liquidity',
            description='Remove liquidity from a Pendle market with single token output',
            parameters=[
                ActionParameter(name='market_address', type=str, required=True, description='Pendle market address'),
                ActionParameter(name='token_out', type=str, required=True, description='Output token address'),
                ActionParameter(name='amount_in', type=str, required=True, description='Amount of LP token to remove'),
                ActionParameter(name='slippage', type=float, required=False, description='Slippage tolerance (default: 0.01)'),
                ActionParameter(name='user_address', type=str, required=True, description='User address'),
                ActionParameter(name='enable_aggregator', type=bool, required=False, description='Enable aggregator (default: False)'),
            ]
        )
        self.actions['remove_liquidity_dual'] = Action(
            name='remove_liquidity_dual',
            description='Remove liquidity from a Pendle market to token and PT',
            parameters=[
                ActionParameter(name='market_address', type=str, required=True, description='Pendle market address'),
                ActionParameter(name='token_out', type=str, required=True, description='Output token address'),
                ActionParameter(name='amount_in', type=str, required=True, description='Amount of LP token to remove'),
                ActionParameter(name='slippage', type=float, required=False, description='Slippage tolerance (default: 0.01)'),
                ActionParameter(name='user_address', type=str, required=True, description='User address'),
                ActionParameter(name='enable_aggregator', type=bool, required=False, description='Enable aggregator (default: False)'),
            ]
        )
        self.actions['swap'] = Action(
            name='swap',
            description='Execute a swap in a Pendle market',
            parameters=[
                ActionParameter(name='market_address', type=str, required=True, description='Pendle market address'),
                ActionParameter(name='token_in', type=str, required=True, description='Input token address'),
                ActionParameter(name='token_out', type=str, required=True, description='Output token address'),
                ActionParameter(name='amount_in', type=str, required=True, description='Amount of input token'),
                ActionParameter(name='slippage', type=float, required=False, description='Slippage tolerance (default: 0.01)'),
                ActionParameter(name='user_address', type=str, required=True, description='User address'),
                ActionParameter(name='enable_aggregator', type=bool, required=False, description='Enable aggregator (default: False)'),
            ]
        )
        
        # Add mint/redeem operations
        self.actions['mint_sy'] = Action(
            name='mint_sy',
            description='Mint SY tokens from a base token',
            parameters=[
                ActionParameter(name='sy', type=str, required=True, description='SY token address'),
                ActionParameter(name='token_in', type=str, required=True, description='Input token address'),
                ActionParameter(name='amount_in', type=str, required=True, description='Amount of input token'),
                ActionParameter(name='slippage', type=float, required=False, description='Slippage tolerance (default: 0.01)'),
                ActionParameter(name='user_address', type=str, required=True, description='User address'),
                ActionParameter(name='enable_aggregator', type=bool, required=False, description='Enable aggregator (default: False)'),
            ]
        )
        self.actions['mint_py'] = Action(
            name='mint_py',
            description='Mint PT and YT from SY or a base token',
            parameters=[
                ActionParameter(name='yt', type=str, required=True, description='YT token address'),
                ActionParameter(name='token_in', type=str, required=True, description='Input token address'),
                ActionParameter(name='amount_in', type=str, required=True, description='Amount of input token'),
                ActionParameter(name='slippage', type=float, required=False, description='Slippage tolerance (default: 0.01)'),
                ActionParameter(name='user_address', type=str, required=True, description='User address'),
                ActionParameter(name='enable_aggregator', type=bool, required=False, description='Enable aggregator (default: False)'),
            ]
        )
        self.actions['redeem_sy'] = Action(
            name='redeem_sy',
            description='Redeem SY tokens to a base token',
            parameters=[
                ActionParameter(name='sy', type=str, required=True, description='SY token address'),
                ActionParameter(name='token_out', type=str, required=True, description='Output token address'),
                ActionParameter(name='amount_in', type=str, required=True, description='Amount of SY token to redeem'),
                ActionParameter(name='slippage', type=float, required=False, description='Slippage tolerance (default: 0.01)'),
                ActionParameter(name='user_address', type=str, required=True, description='User address'),
                ActionParameter(name='enable_aggregator', type=bool, required=False, description='Enable aggregator (default: False)'),
            ]
        )
        self.actions['redeem_py'] = Action(
            name='redeem_py',
            description='Redeem PT and YT to SY or a base token',
            parameters=[
                ActionParameter(name='yt', type=str, required=True, description='YT token address'),
                ActionParameter(name='token_out', type=str, required=True, description='Output token address'),
                ActionParameter(name='amount_in', type=str, required=True, description='Amount of PT and YT to redeem'),
                ActionParameter(name='slippage', type=float, required=False, description='Slippage tolerance (default: 0.01)'),
                ActionParameter(name='user_address', type=str, required=True, description='User address'),
                ActionParameter(name='enable_aggregator', type=bool, required=False, description='Enable aggregator (default: False)'),
            ]
        )

    def _call_sdk(self, path: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Helper method to call the Pendle SDK API
        """
        try:
            url = f"{self.sdk_url}{path}"
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise PendleConnectionError(f"Failed to call Pendle SDK: {str(e)}")

    def get_markets(self) -> List[Dict[str, Any]]:
        """
        Fetches all markets for a given chain ID and transforms the response
        from list format to proper market objects
        """
        try:
            url = f"{self.api_url}/bff/v3/markets/all?chainId=146&select=all"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            return self._transform_markets_response(data)
        except Exception as e:
            raise PendleConnectionError(f"Failed to fetch Pendle markets: {str(e)}")

    def get_assets(self) -> List[Dict[str, Any]]:
        """
        Fetches all assets for a given chain ID and transforms the response
        from list format to proper asset objects
        """
        try:
            url = f"{self.api_url}/bff/v3/assets/all?chainId=146"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            return self._transform_assets_response(data)
        except Exception as e:
            raise PendleConnectionError(f"Failed to fetch Pendle assets: {str(e)}")

    def parse_token_id(self, token_id: str) -> Dict[str, Any]:
        """
        Parse token ID in format "146-0x..." to get the clean address
        """
        try:
            # Check if it's in the format of CHAINID-ADDRESS
            if '-' in token_id:
                chain_id_str, address = token_id.split('-')
                return {
                    "chain_id": 146,
                    "address": address
                }
            
            # If no dash, assume it's just an address
            return {
                "chain_id": 146,
                "address": token_id
            }
        except Exception as e:
            raise PendleConnectionError(f"Error parsing token ID: {str(e)}")

    def _transform_markets_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Transforms the list-based API response into an array of market objects
        """
        markets = []
        chain_id_list = response.get('chainIdList', [])
        
        for i in range(len(chain_id_list)):
            # Skip if there's no liquidity data
            if not response.get('liquidityList') or i >= len(response.get('liquidityList')):
                continue
                
            market = {
                'chainId': response.get('chainIdList', [])[i] if i < len(response.get('chainIdList', [])) else None,
                'address': response.get('addressList', [])[i] if i < len(response.get('addressList', [])) else None,
                'symbol': response.get('symbolList', [])[i] if i < len(response.get('symbolList', [])) else None,
                'expiry': response.get('expiryList', [])[i] if i < len(response.get('expiryList', [])) else None,
                'icon': response.get('iconList', [])[i] if i < len(response.get('iconList', [])) else None,
                'pt': response.get('ptList', [])[i] if i < len(response.get('ptList', [])) else None,
                'yt': response.get('ytList', [])[i] if i < len(response.get('ytList', [])) else None,
                'sy': response.get('syList', [])[i] if i < len(response.get('syList', [])) else None,
                'accountingAsset': response.get('accountingAssetList', [])[i] if i < len(response.get('accountingAssetList', [])) else None,
                'underlyingAsset': response.get('underlyingAssetList', [])[i] if i < len(response.get('underlyingAssetList', [])) else None,
                'rewardTokens': response.get('rewardTokensList', [])[i] if i < len(response.get('rewardTokensList', [])) else [],
                'inputTokens': response.get('inputTokensList', [])[i] if i < len(response.get('inputTokensList', [])) else [],
                'outputTokens': response.get('outputTokensList', [])[i] if i < len(response.get('outputTokensList', [])) else [],
                'protocol': response.get('protocolList', [])[i] if i < len(response.get('protocolList', [])) else None,
                'underlyingPool': response.get('underlyingPoolList', [])[i] if i < len(response.get('underlyingPoolList', [])) else None,
                'liquidity': response.get('liquidityList', [])[i] if i < len(response.get('liquidityList', [])) else None,
                'tradingVolume': response.get('tradingVolumeList', [])[i] if i < len(response.get('tradingVolumeList', [])) else None,
                'underlyingApy': response.get('underlyingApyList', [])[i] if i < len(response.get('underlyingApyList', [])) else None,
                'impliedApy': response.get('impliedApyList', [])[i] if i < len(response.get('impliedApyList', [])) else None,
                'ytFloatingApy': response.get('ytFloatingApyList', [])[i] if i < len(response.get('ytFloatingApyList', [])) else None,
                'ptDiscount': response.get('ptDiscountList', [])[i] if i < len(response.get('ptDiscountList', [])) else None,
                'isNew': response.get('isNewList', [])[i] if i < len(response.get('isNewList', [])) else None,
                'isFeatured': response.get('isFeaturedList', [])[i] if i < len(response.get('isFeaturedList', [])) else None,
                'isActive': response.get('isActiveList', [])[i] if i < len(response.get('isActiveList', [])) else None,
            }
            
            markets.append(market)
        
        return markets

    def _transform_assets_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Transforms the list-based assets API response into an array of asset objects
        """
        assets = []
        chain_id_list = response.get('chainIdList', [])
        
        for i in range(len(chain_id_list)):
            asset = {
                'chainId': response.get('chainIdList', [])[i] if i < len(response.get('chainIdList', [])) else None,
                'address': response.get('addressList', [])[i] if i < len(response.get('addressList', [])) else None,
                'symbol': response.get('symbolList', [])[i] if i < len(response.get('symbolList', [])) else None,
                'icon': response.get('iconList', [])[i] if i < len(response.get('iconList', [])) else None,
                'decimals': response.get('decimalsList', [])[i] if i < len(response.get('decimalsList', [])) else None,
                'price': response.get('priceList', [])[i] if i < len(response.get('priceList', [])) else None,
                'type': response.get('typeList', [])[i] if i < len(response.get('typeList', [])) else None,
                'underlyingPool': response.get('underlyingPoolList', [])[i] if i < len(response.get('underlyingPoolList', [])) else None,
                'zappable': response.get('zappableList', [])[i] if i < len(response.get('zappableList', [])) else None,
                'expiry': response.get('expiryList', [])[i] if i < len(response.get('expiryList', [])) else None,
            }
            
            assets.append(asset)
        
        return assets

    def add_liquidity(self, market_address: str, token_in: str, amount_in: str, 
                      slippage: float = 0.01, zpi: bool = False, user_address: str = None,
                      enable_aggregator: bool = False) -> Dict[str, Any]:
        """
        Add liquidity to a Pendle market with a single token input
        """
        try:
            # Format amount to Wei (18 decimals)
            
            params = {
                "receiver": user_address,
                "slippage": slippage,
                "tokenIn": token_in,
                "amountIn": amount_in,
                "zpi": zpi,
                "enableAggregator": enable_aggregator
            }
            
            result = self._call_sdk(f"v1/sdk/146/markets/{market_address}/add-liquidity", params)
            
            logger.info(f"Amount LP Out: {result['data']['amountLpOut']}")
            logger.info(f"Price impact: {result['data']['priceImpact']}")
            
            if zpi and 'amountYtOut' in result['data']:
                logger.info(f"Amount YT Out: {result['data']['amountYtOut']}")
            
            return result
        except Exception as e:
            raise PendleConnectionError(f"Failed to add liquidity: {str(e)}")

    def add_liquidity_dual(self, market_address: str, token_in: str, 
                           amount_token_in: str, amount_pt_in: str, slippage: float = 0.01,
                           user_address: str = None, enable_aggregator: bool = False) -> Dict[str, Any]:
        """
        Add liquidity to a Pendle market with dual inputs (token and PT)
        """
        try:
            # Format amounts to Wei (18 decimals)
            
            params = {
                "receiver": user_address,
                "slippage": slippage,
                "tokenIn": token_in,
                "amountTokenIn": amount_token_in,
                "amountPtIn": amount_pt_in,
                "enableAggregator": enable_aggregator
            }
            
            result = self._call_sdk(f"v1/sdk/146/markets/{market_address}/add-liquidity-dual", params)
            
            logger.info(f"Amount LP Out: {result['data']['amountOut']}")
            logger.info(f"Price impact: {result['data']['priceImpact']}")
            
            return result
        except Exception as e:
            raise PendleConnectionError(f"Failed to add dual liquidity: {str(e)}")

    def remove_liquidity(self, market_address: str, token_out: str, amount_in: str,
                         slippage: float = 0.01, user_address: str = None,
                         enable_aggregator: bool = False) -> Dict[str, Any]:
        """
        Remove liquidity from a Pendle market with single token output
        """
        try:
            # Format amount to Wei (18 decimals)
            
            params = {
                "receiver": user_address,
                "slippage": slippage,
                "tokenOut": token_out,
                "amountIn": amount_in,
                "enableAggregator": enable_aggregator
            }
            
            result = self._call_sdk(f"v1/sdk/146/markets/{market_address}/remove-liquidity", params)
            
            logger.info(f"Amount Out: {result['data']['amountOut']}")
            logger.info(f"Price impact: {result['data']['priceImpact']}")
            
            return result
        except Exception as e:
            raise PendleConnectionError(f"Failed to remove liquidity: {str(e)}")

    def remove_liquidity_dual(self, market_address: str, token_out: str, amount_in: str,
                              slippage: float = 0.01, user_address: str = None,
                              enable_aggregator: bool = False) -> Dict[str, Any]:
        """
        Remove liquidity from a Pendle market to token and PT
        """
        try:
            # Format amount to Wei (18 decimals)
            
            params = {
                "receiver": user_address,
                "slippage": slippage,
                "tokenOut": token_out,
                "amountIn": amount_in,
                "enableAggregator": enable_aggregator
            }
            
            result = self._call_sdk(f"v1/sdk/146/markets/{market_address}/remove-liquidity-dual", params)
            
            logger.info(f"Amount Token Out: {result['data']['amountTokenOut']}")
            logger.info(f"Amount PT Out: {result['data']['amountPtOut']}")
            logger.info(f"Price impact: {result['data']['priceImpact']}")
            
            return result
        except Exception as e:
            raise PendleConnectionError(f"Failed to remove dual liquidity: {str(e)}")

    def swap(self, market_address: str, token_in: str, token_out: str, amount_in: str,
             slippage: float = 0.01, user_address: str = None,
             enable_aggregator: bool = False) -> Dict[str, Any]:
        """
        Execute a swap in a Pendle market
        """
        try:
            # Format amount to Wei (18 decimals)
            
            params = {
                "receiver": user_address,
                "slippage": slippage,
                "tokenIn": token_in,
                "tokenOut": token_out,
                "amountIn": amount_in,
                "enableAggregator": enable_aggregator
            }
            
            result = self._call_sdk(f"v1/sdk/146/markets/{market_address}/swap", params)
            
            logger.info(f"Amount Out: {result['data']['amountOut']}")
            logger.info(f"Price impact: {result['data']['priceImpact']}")
            
            return result
        except Exception as e:
            raise PendleConnectionError(f"Failed to swap tokens: {str(e)}")

    def mint_sy(self, sy: str, token_in: str, amount_in: str,
                slippage: float = 0.01, user_address: str = None,
                enable_aggregator: bool = False) -> Dict[str, Any]:
        """
        Mint SY tokens from a base token
        """
        try:
            # Format amount to Wei (18 decimals)
            
            params = {
                "receiver": user_address,
                "sy": sy,
                "tokenIn": token_in,
                "amountIn": amount_in,
                "slippage": slippage,
                "enableAggregator": enable_aggregator
            }
            
            result = self._call_sdk(f"v1/sdk/146/mint-sy", params)
            
            logger.info(f"Amount SY Out: {result['data']['amountOut']}")
            logger.info(f"Price impact: {result['data']['priceImpact']}")
            
            return result
        except Exception as e:
            raise PendleConnectionError(f"Failed to mint SY: {str(e)}")

    def mint_py(self, yt: str, token_in: str, amount_in: str,
                slippage: float = 0.01, user_address: str = None,
                enable_aggregator: bool = False) -> Dict[str, Any]:
        """
        Mint PT and YT from SY or a base token
        """
        try:
            # Format amount to Wei (18 decimals)
            
            params = {
                "receiver": user_address,
                "yt": yt,
                "tokenIn": token_in,
                "amountIn": amount_in,
                "slippage": slippage,
                "enableAggregator": enable_aggregator
            }
            
            result = self._call_sdk(f"v1/sdk/146/mint", params)
            
            logger.info(f"Amount PT & YT Out: {result['data']['amountOut']}")
            logger.info(f"Price impact: {result['data']['priceImpact']}")
            
            return result
        except Exception as e:
            raise PendleConnectionError(f"Failed to mint PT and YT: {str(e)}")

    def redeem_sy(self, sy: str, token_out: str, amount_in: str,
                  slippage: float = 0.01, user_address: str = None,
                  enable_aggregator: bool = False) -> Dict[str, Any]:
        """
        Redeem SY tokens to a base token
        """
        try:
            # Format amount to Wei (18 decimals)
            
            params = {
                "receiver": user_address,
                "sy": sy,
                "tokenOut": token_out,
                "amountIn": amount_in,
                "slippage": slippage,
                "enableAggregator": enable_aggregator
            }
            
            result = self._call_sdk(f"v1/sdk/146/redeem-sy", params)
            
            logger.info(f"Amount Out: {result['data']['amountOut']}")
            logger.info(f"Price impact: {result['data']['priceImpact']}")
            
            return result
        except Exception as e:
            raise PendleConnectionError(f"Failed to redeem SY: {str(e)}")

    def redeem_py(self, yt: str, token_out: str, amount_in: str,
                  slippage: float = 0.01, user_address: str = None,
                  enable_aggregator: bool = False) -> Dict[str, Any]:
        """
        Redeem PT and YT to SY or a base token
        """
        try:
            # Format amount to Wei (18 decimals)
            
            params = {
                "receiver": user_address,
                "yt": yt,
                "tokenOut": token_out,
                "amountIn": amount_in,
                "slippage": slippage,
                "enableAggregator": enable_aggregator
            }
            
            result = self._call_sdk(f"v1/sdk/146/redeem", params)
            
            logger.info(f"Amount Out: {result['data']['amountOut']}")
            logger.info(f"Price impact: {result['data']['priceImpact']}")
            
            return result
        except Exception as e:
            raise PendleConnectionError(f"Failed to redeem PT and YT: {str(e)}")

    def perform_action(self, action_name: str, kwargs) -> Any:
        """Execute a Pendle action with validation"""
        if action_name not in self.actions:
            raise KeyError(f"Unknown action: {action_name}")
        
        if not self.is_configured(verbose=True):
            raise PendleConnectionError("Pendle connection is not properly configured")

        action = self.actions[action_name]
        validation_errors = action.validate_params(kwargs)
        if validation_errors:
            raise PendleConnectionError(f"Invalid parameters: {', '.join(validation_errors)}")

        method_name = action_name.replace('-', '_')
        method = getattr(self, method_name)
        return method(**kwargs)
