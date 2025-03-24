import json
import logging
from langchain.tools import BaseTool

logger = logging.getLogger("tools.pendle_tools")

class PendleMarketsInfoTool(BaseTool):
    name: str = "pendle_markets"
    description: str = """
    pendle_markets: Get information about available Pendle markets, including liquidity, APYs, and token pairs.
    
    Examples:
      - Get all Pendle markets on Sonic: pendle_markets() 
    """
    
    def __init__(self, agent):
        super().__init__()
        self._agent = agent
    
    def _run(self) -> str:
        try:
            markets = self._agent.connection_manager.connections["pendle"].get_markets()
            
            # Format the market data for better readability
            formatted_markets = []
            for market in markets:
                # Skip inactive markets
                if not market.get('isActive', True):
                    continue
                    
                formatted_market = {
                    'market': market.get('symbol', ''),
                    'expires': (market.get('expiry', 0)),
                    'liquidity': f"${float(market.get('liquidity', 0)):,.2f}",
                    'implied_apy': f"{float(market.get('impliedApy', 0)) * 100:.2f}%",
                    'yield_apy': f"{float(market.get('ytFloatingApy', 0)) * 100:.2f}%",
                    'pt_discount': f"{float(market.get('ptDiscount', 0)) * 100:.2f}%",
                    'address': market.get('address', ''),
                    'pt': market.get('pt', ''),
                    'yt': market.get('yt', ''),
                    'sy': market.get('sy', ''),
                    'trading_volume': f"${float(market.get('tradingVolume', 0)):,.2f}",
                    'underlying_asset': market.get('underlyingAsset', ''),
                    'is_featured': market.get('isFeatured', False),
                }
                formatted_markets.append(formatted_market)
            
            # Sort markets by liquidity (descending)
            formatted_markets.sort(key=lambda x: float(x['liquidity'].replace('$', '').replace(',', '')), reverse=True)
            
            result = {
                'markets_count': len(formatted_markets),
                'markets': formatted_markets
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error fetching Pendle markets: {str(e)}"

class PendleAssetsInfoTool(BaseTool):
    name: str = "pendle_assets"
    description: str = """
    pendle_assets: Get information about tokens available in the Pendle protocol on a specific chain.
    
    Examples:
      - Get all Pendle assets on Sonic: pendle_assets()
    
    """
    def __init__(self, agent):
        super().__init__()
        self._agent = agent
    
    def _run(self) -> str:
        try:
            assets = self._agent.connection_manager.connections["pendle"].get_assets()
            
            # Format the asset data for better readability
            formatted_assets = []
            for asset in assets:
                # Skip assets without symbols
                if not asset.get('symbol'):
                    continue
                    
                price_str = "Unknown"
                if asset.get('price') is not None:
                    price_str = f"${float(asset.get('price', 0)):,.6f}"
                
                formatted_asset = {
                    'symbol': asset.get('symbol', ''),
                    'type': asset.get('type', ''),
                    'address': asset.get('address', ''),
                    'price': price_str,
                    'decimals': asset.get('decimals', 18),
                    'expiry': asset.get('expiry'),
                    'zappable': asset.get('zappable', False),
                }
                formatted_assets.append(formatted_asset)
            
            # Group assets by type
            asset_types = {}
            for asset in formatted_assets:
                asset_type = asset.get('type', 'unknown')
                if asset_type not in asset_types:
                    asset_types[asset_type] = []
                asset_types[asset_type].append(asset)
            
            result = {
                'assets_count': len(formatted_assets),
                'asset_types': asset_types
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error fetching Pendle assets: {str(e)}"

class PendleMarketDetailsTool(BaseTool):
    name: str = "pendle_market_details"
    description: str = """
    pendle_market_details: Get detailed information about a specific Pendle market.
    
    Examples:
      - Get details for a market with address 0x1234...: pendle_market_details(market_address="0x1234...")
    
    Args:
        market_address: The address of the Pendle market
    """
    
    def __init__(self, agent):
        super().__init__()
        self._agent = agent
    
    def _run(self, market_address: str) -> str:
        try:
            # Get all markets and find the specific one
            markets = self._agent.connection_manager.connections["pendle"].get_markets()
            
            target_market = None
            for market in markets:
                if market.get('address', '').lower() == market_address.lower():
                    target_market = market
                    break
            
            if not target_market:
                return f"Error: Market with address {market_address} not found"
            
            # Format the detailed market data
            from datetime import datetime
            
            expiry_str = "Perpetual"
            if target_market.get('expiry'):
                dt = datetime.fromtimestamp(target_market.get('expiry'))
                expiry_str = dt.strftime("%Y-%m-%d")
            
            market_details = {
                'market': target_market.get('symbol', ''),
                'address': target_market.get('address', ''),
                'expires': expiry_str,
                'status': "Active" if target_market.get('isActive', True) else "Inactive",
                'liquidity': f"${float(target_market.get('liquidity', 0)):,.2f}",
                'trading_volume': f"${float(target_market.get('tradingVolume', 0)):,.2f}",
                'token_info': {
                    'underlying_asset': target_market.get('underlyingAsset', ''),
                    'accounting_asset': target_market.get('accountingAsset', ''),
                    'pt_address': target_market.get('pt', ''),
                    'yt_address': target_market.get('yt', ''),
                    'sy_address': target_market.get('sy', ''),
                },
                'yields': {
                    'implied_apy': f"{float(target_market.get('impliedApy', 0)) * 100:.2f}%",
                    'yt_floating_apy': f"{float(target_market.get('ytFloatingApy', 0)) * 100:.2f}%",
                    'underlying_apy': f"{float(target_market.get('underlyingApy', 0)) * 100:.2f}%",
                    'pt_discount': f"{float(target_market.get('ptDiscount', 0)) * 100:.2f}%",
                },
                'input_tokens': target_market.get('inputTokens', []),
                'output_tokens': target_market.get('outputTokens', []),
                'featured': target_market.get('isFeatured', False),
                'protocol': target_market.get('protocol', ''),
                'reward_tokens': target_market.get('rewardTokens', [])
            }
            
            return json.dumps(market_details, indent=2)
            
        except Exception as e:
            return f"Error getting Pendle market details: {str(e)}"

class PendleAddLiquidityTool(BaseTool):
    name: str = "pendle_add_liquidity"
    description: str = """
    pendle_add_liquidity: Add liquidity to a Pendle market using user-friendly token symbols.
    
    Examples:
      - Add 100 USDC to Pendle market stS: pendle_add_liquidity(market_symbol="stS", token_symbol="USDC", amount_in="100", user_address="0x789...")
      - Add 50 stS to Pendle market stS with 0.5% slippage: pendle_add_liquidity(market_symbol="stS", token_symbol="stS", token_type="SY", amount_in="50", slippage=0.005, user_address="0x789...")
      - Add 100 USDC to Pendle market stS with Zero Price Impact (ZPI): pendle_add_liquidity(market_symbol="stS", token_symbol="USDC", amount_in="100", zpi=True, user_address="0x789...")
    
    Args:
        market_symbol: Symbol of the Pendle market (e.g., "stS", "USDC", etc.)
        token_symbol: Symbol of the token to provide as liquidity (e.g., "USDC", "stS", etc.)
        token_type: Optional type of token (e.g., "SY", "PT", "YT") if needed to distinguish between tokens with the same symbol
        amount_in: The amount of tokens to provide
        user_address: The address of the user providing liquidity
        slippage: Maximum slippage tolerance (default: 0.01 = 1%)
        zpi: Zero Price Impact flag (default: False)
    """
    
    def __init__(self, agent):
        super().__init__()
        self._agent = agent
    
    def _run(self, market_symbol: str, token_symbol: str, amount_in: str, user_address: str,
             token_type: str = None, slippage: float = 0.01, zpi: bool = False) -> str:
        try:
            # Step 1: Find the market by symbol
            markets = self._agent.connection_manager.connections["pendle"].get_markets()
            target_market = None
            
            for market in markets:
                if market.get('symbol', '').lower() == market_symbol.lower():
                    target_market = market
                    break
            
            if not target_market:
                return f"Error: No Pendle market found with symbol '{market_symbol}'"
            
            market_address = target_market['address']
            
            # Step 2: Find the token by symbol and type
            assets = self._agent.connection_manager.connections["pendle"].get_assets()
            matching_tokens = []
            for asset in assets:
                if asset.get('symbol', '').lower() == token_symbol.lower():
                    # If token_type is specified, check if it matches
                    if token_type:
                        if asset.get('type', '').lower() == token_type.lower():
                            matching_tokens.append(asset)
                    else:
                        if asset.get('type', '').lower() != 'sy':
                            matching_tokens.append(asset)
            
            if not matching_tokens:
                return f"Error: No token found with symbol '{token_symbol}'"
            
            # If multiple tokens match, use the first one or the one with matching type
            token_asset = matching_tokens[0]
            token_address = token_asset['address']
            token_decimals = token_asset.get('decimals', 18)
            amount_in = int(float(amount_in) * 10**token_decimals)
            # Step 3: Determine if we should enable the aggregator
            enable_aggregator = False
            if token_address:
                # Check if the token is in the market's input tokens
                input_tokens = [self._agent.connection_manager.connections["pendle"].parse_token_id(t)['address'].lower() 
                               for t in target_market.get('tokensMintSy', [])]
                input_tokens.append(self._agent.connection_manager.connections["pendle"].parse_token_id(target_market.get('sy'))['address'].lower())
                enable_aggregator = token_address.lower() not in input_tokens
            
            # Step 4: Call the Pendle connection to add liquidity
            result = self._agent.connection_manager.connections["pendle"].add_liquidity(
                market_address=market_address,
                token_in=token_address,
                amount_in=amount_in,
                slippage=slippage,
                zpi=zpi,
                user_address=user_address,
                enable_aggregator=enable_aggregator
            )
            
            # Format the output for better readability
            output = {
                "success": True,
                "type": "Add Liquidity",
                "market": market_symbol,
                "market_address": market_address,
                "token_in": token_symbol,
                "token_address": token_address,
                "token_decimals": token_decimals,
                "token_type": token_type or token_asset.get('type', 'Unknown'),
                "amount_in": amount_in,
                "user_address": user_address,
                "lp_tokens_received": result["data"]["amountLpOut"],
                "price_impact": f"{result['data']['priceImpact']:.4f}%",
                "aggregator_used": enable_aggregator,
                "transaction": {
                    "to": result["tx"]["to"],
                    "data": result["tx"]["data"],
                    "value": result["tx"].get("value")
                }
            }
            
            # Add YT output if using Zero Price Impact (ZPI)
            if zpi and "amountYtOut" in result["data"]:
                output["yt_tokens_received"] = (result["data"]["amountYtOut"])
            
            return json.dumps(output, indent=2)
            
        except Exception as e:
            return f"Error adding liquidity to Pendle market: {str(e)}"
    
class PendleAddLiquidityDualTool(BaseTool):
    name: str = "pendle_add_liquidity_dual"
    description: str = """
    pendle_add_liquidity_dual: Add liquidity to a Pendle market with both token and PT (Principal Token) using user-friendly symbol names.
    
    Examples:
      - Add 100 USDC and 50 PT to Pendle market stS: pendle_add_liquidity_dual(market_symbol="stS", token_symbol="USDC", amount_token_in="100", amount_pt_in="50", user_address="0x789...")
      - Add 200 WETH and 100 PT to Pendle market stETH with 0.5% slippage: pendle_add_liquidity_dual(market_symbol="stETH", token_symbol="WETH", token_type="UNDERLYING", amount_token_in="200", amount_pt_in="100", slippage=0.005, user_address="0x789...")
    
    Args:
        market_symbol: Symbol of the Pendle market (e.g., "stS", "stETH", etc.)
        token_symbol: Symbol of the input token (not PT)
        token_type: Optional type of input token (e.g., "SY", "UNDERLYING") if needed to distinguish
        amount_token_in: The amount of the input token to provide
        amount_pt_in: The amount of PT (Principal Token) to provide
        user_address: The address of the user providing liquidity
        slippage: Maximum slippage tolerance (default: 0.01 = 1%)
    """
    
    def __init__(self, agent):
        super().__init__()
        self._agent = agent
    
    def _run(self, market_symbol: str, token_symbol: str, amount_token_in: str, amount_pt_in: str, 
             user_address: str, token_type: str = None, slippage: float = 0.01) -> str:
        try:
            # Step 1: Find the market by symbol
            markets = self._agent.connection_manager.connections["pendle"].get_markets()
            target_market = None
            
            for market in markets:
                if market.get('symbol', '').lower() == market_symbol.lower():
                    target_market = market
                    break
            
            if not target_market:
                return f"Error: No Pendle market found with symbol '{market_symbol}'"
            
            market_address = target_market['address']
            
            # Step 2: Find the token by symbol and type
            assets = self._agent.connection_manager.connections["pendle"].get_assets()
            matching_tokens = []
            pt_decimals = 18

            for asset in assets:
                # Check if PT address matches
                if asset.get('address', '').lower() == self._agent.connection_manager.connections["pendle"].parse_token_id(target_market['pt'])["address"].lower():
                    pt_decimals = asset.get('decimals', 18)

                # Check symbol match
                if asset.get('symbol', '').lower() != token_symbol.lower():
                    continue  # Skip if symbol doesn't match

                asset_type = asset.get('type', '').lower()

                # If token_type is specified, check if it matches
                if token_type:
                    if asset_type == token_type.lower():
                        matching_tokens.append(asset)
                else:
                    # Default behavior: skip 'sy' tokens
                    if asset_type != 'sy':
                        matching_tokens.append(asset)
            
            if not matching_tokens:
                return f"Error: No token found with symbol '{token_symbol}'"
            
            # If multiple tokens match, use the first one or the one with matching type
            token_asset = matching_tokens[0]
            token_address = token_asset['address']
            token_decimals = token_asset.get('decimals', 18)
            # Step 3: Get PT address from the market
            pt_address = self._agent.connection_manager.connections["pendle"].parse_token_id(target_market.get('pt', ''))['address']
            if not pt_address:
                return f"Error: Could not find PT token for market '{market_symbol}'"
            
            # Step 4: Determine if we should enable the aggregator
            enable_aggregator = False
            if token_address:
                # Check if the token is in the market's input tokens
                input_tokens = [self._agent.connection_manager.connections["pendle"].parse_token_id(t)['address']
                               for t in target_market.get('tokensMintSy', [])]
                input_tokens.append(self._agent.connection_manager.connections["pendle"].parse_token_id(target_market.get('sy'))['address'])
                if token_address not in input_tokens:
                    return f"Error: Token '{token_symbol}' is not a valid input token for market '{market_symbol}'"
            amount_token_in = int(float(amount_token_in) * 10**token_decimals)
            amount_pt_in = int(float(amount_pt_in) * 10**pt_decimals)
            # Step 5: Call the Pendle connection to add dual liquidity
            result = self._agent.connection_manager.connections["pendle"].add_liquidity_dual(
                market_address=market_address,
                token_in=token_address,
                amount_token_in=amount_token_in,
                amount_pt_in=amount_pt_in,
                slippage=slippage,
                user_address=user_address,
                enable_aggregator=enable_aggregator
            )
            
            # Format the output for better readability
            output = {
                "success": True,
                "type": "Add Dual Liquidity",
                "market": market_symbol,
                "market_address": market_address,
                "token_in": token_symbol,
                "token_address": token_address,
                "token_decimals": token_decimals,
                "pt_address": pt_address,
                "pt_decimals": pt_decimals,
                "token_type": token_type or token_asset.get('type', 'Unknown'),
                "pt_token": "PT",
                "amount_token_in": amount_token_in,
                "amount_pt_in": amount_pt_in,
                "user_address": user_address,
                "lp_tokens_received": result["data"]["amountOut"],
                "price_impact": f"{result['data']['priceImpact']:.4f}%",
                "aggregator_used": enable_aggregator,
                "transaction": {
                    "to": result["tx"]["to"],
                    "data": result["tx"]["data"],
                    "value": result["tx"].get("value")
                }
            }
            
            return json.dumps(output, indent=2)
            
        except Exception as e:
            return f"Error adding dual liquidity to Pendle market: {str(e)}"

class PendleRemoveLiquidityTool(BaseTool):
    name: str = "pendle_remove_liquidity"
    description: str = """
    pendle_remove_liquidity: Remove liquidity from a Pendle market and receive a single token as output using user-friendly token symbols.
    
    Examples:
      - Remove 50 LP tokens from Pendle market stS and get USDC: pendle_remove_liquidity(market_symbol="stS", token_out_symbol="USDC", amount_in="50", user_address="0x789...")
      - Remove 100 LP tokens from Pendle market stETH and get ETH with 0.5% slippage: pendle_remove_liquidity(market_symbol="stETH", token_out_symbol="ETH", token_out_type="UNDERLYING", amount_in="100", slippage=0.005, user_address="0x789...")
      - Remove 25 LP tokens from Pendle market stS and get SY: pendle_remove_liquidity(market_symbol="stS", token_out_symbol="stS", token_out_type="SY", amount_in="25", user_address="0x789...")
    
    Args:
        market_symbol: Symbol of the Pendle market (e.g., "stS", "stETH", etc.)
        token_out_symbol: Symbol of the token you want to receive
        token_out_type: Optional type of output token (e.g., "SY", "UNDERLYING") if needed to distinguish
        amount_in: The amount of LP tokens to remove
        user_address: The address of the user removing liquidity
        slippage: Maximum slippage tolerance (default: 0.01 = 1%)
    """
    
    def __init__(self, agent):
        super().__init__()
        self._agent = agent
    
    def _run(self, market_symbol: str, token_out_symbol: str, amount_in: str, user_address: str,
             token_out_type: str = None, slippage: float = 0.01) -> str:
        try:
            # Step 1: Find the market by symbol
            markets = self._agent.connection_manager.connections["pendle"].get_markets()
            target_market = None
            
            for market in markets:
                if market.get('symbol', '').lower() == market_symbol.lower():
                    target_market = market
                    break
            
            if not target_market:
                return f"Error: No Pendle market found with symbol '{market_symbol}'"
            
            market_address = target_market['address']
            market_output_tokens = target_market.get('tokensRedeemSy', [])
            market_output_tokens.append(target_market.get('sy', ''))
            # Step 2: Find the output token by symbol and type
            assets = self._agent.connection_manager.connections["pendle"].get_assets()
            
            # Special handling for PT, YT, and SY tokens from the market
            if token_out_symbol.lower() == "pt":
                token_out = target_market.get('pt', '')
            elif token_out_symbol.lower() == "yt":
                token_out = target_market.get('yt', '')
            elif token_out_symbol.lower() == "sy" or (token_out_symbol.lower() == market_symbol.lower() and token_out_type and token_out_type.lower() == "sy"):
                token_out = target_market.get('sy', '')
            else:
                # Regular token lookup
                matching_tokens_out = []
                for asset in assets:
                    if asset.get('symbol', '').lower() == token_out_symbol.lower():
                        # If token_type is specified, check if it matches
                        if token_out_type:
                            if asset.get('type', '').lower() == token_out_type.lower():
                                matching_tokens_out.append(asset)
                        else:
                            if asset.get('type', '').lower() != 'sy':
                                matching_tokens_out.append(asset)
                
                if not matching_tokens_out:
                    return f"Error: No token found with symbol '{token_out_symbol}'"
                
                # If multiple tokens match, use the first one or the one with matching type
                token_out_asset = matching_tokens_out[0]
                token_out = token_out_asset.get('address', '')
            
            # Step 3: Determine if we should enable the aggregator
            # Enable aggregator if the output token is not directly in the market's output tokens
            output_token_address = None
            if isinstance(token_out, str) and token_out.startswith('146-'):
                output_token_address = token_out.split('-')[1].lower()
            else:
                output_token_address = token_out.lower()
                
            enable_aggregator = True
            for market_token in market_output_tokens:
                try:
                    parsed_token = self._agent.connection_manager.connections["pendle"].parse_token_id(market_token)
                    if parsed_token['address'].lower() == output_token_address:
                        enable_aggregator = False
                        break
                except:
                    continue
            amount_in = int(float(amount_in) * 10**18)
            # Step 4: Call the Pendle connection to remove liquidity
            result = self._agent.connection_manager.connections["pendle"].remove_liquidity(
                market_address=market_address,
                token_out=token_out,
                amount_in=amount_in,
                slippage=slippage,
                user_address=user_address,
                enable_aggregator=enable_aggregator
            )
            
            # Format the output for better readability
            output = {
                "success": True,
                "type": "Remove Liquidity",
                "market": market_symbol,
                "market_address": market_address,
                "market_decimals": 18,
                "token_out": token_out_symbol,
                "token_out_type": token_out_type or "Standard",
                "lp_tokens_removed": amount_in,
                "user_address": user_address,
                "tokens_received": (result["data"]["amountOut"]),
                "price_impact": f"{result['data']['priceImpact']:.4f}%",
                "aggregator_used": enable_aggregator,
                "transaction": {
                    "to": result["tx"]["to"],
                    "data": result["tx"]["data"],
                    "value": result["tx"].get("value")
                }
            }
            
            return json.dumps(output, indent=2)
            
        except Exception as e:
            return f"Error removing liquidity from Pendle market: {str(e)}"

class PendleRemoveLiquidityDualTool(BaseTool):
    name: str = "pendle_remove_liquidity_dual"
    description: str = """
    pendle_remove_liquidity_dual: Remove liquidity from a Pendle market and receive both token and PT using user-friendly token symbols.
    
    Examples:
      - Remove 50 LP tokens from Pendle market stS and get both USDC and PT: pendle_remove_liquidity_dual(market_symbol="stS", token_out_symbol="USDC", amount_in="50", user_address="0x789...")
      - Remove 100 LP tokens from Pendle market stETH and get both ETH and PT with 0.5% slippage: pendle_remove_liquidity_dual(market_symbol="stETH", token_out_symbol="ETH", token_out_type="UNDERLYING", amount_in="100", slippage=0.005, user_address="0x789...")
    
    Args:
        market_symbol: Symbol of the Pendle market (e.g., "stS", "stETH", etc.)
        token_out_symbol: Symbol of the token you want to receive (PT will be received automatically)
        token_out_type: Optional type of output token (e.g., "SY", "UNDERLYING") if needed to distinguish
        amount_in: The amount of LP tokens to remove
        user_address: The address of the user removing liquidity
        slippage: Maximum slippage tolerance (default: 0.01 = 1%)
    """
    
    def __init__(self, agent):
        super().__init__()
        self._agent = agent
    
    def _run(self, market_symbol: str, token_out_symbol: str, amount_in: str, user_address: str,
             token_out_type: str = None, slippage: float = 0.01) -> str:
        try:
            # Step 1: Find the market by symbol
            markets = self._agent.connection_manager.connections["pendle"].get_markets()
            target_market = None
            
            for market in markets:
                if market.get('symbol', '').lower() == market_symbol.lower():
                    target_market = market
                    break
            
            if not target_market:
                return f"Error: No Pendle market found with symbol '{market_symbol}'"
            
            market_address = target_market['address']
            market_output_tokens = target_market.get('tokensRedeemSy', [])
            market_output_tokens.append(target_market.get('sy', ''))
            # Step 2: Find the output token by symbol and type
            assets = self._agent.connection_manager.connections["pendle"].get_assets()
            
            # Special handling for SY tokens from the market
            if token_out_symbol.lower() == "sy" or (token_out_symbol.lower() == market_symbol.lower() and token_out_type and token_out_type.lower() == "sy"):
                token_out = target_market.get('sy', '')
            else:
                # Regular token lookup
                matching_tokens_out = []
                for asset in assets:
                    if asset.get('symbol', '').lower() == token_out_symbol.lower():
                        # If token_type is specified, check if it matches
                        if token_out_type:
                            if asset.get('type', '').lower() == token_out_type.lower():
                                matching_tokens_out.append(asset)
                        else:
                            if asset.get('type', '').lower() != 'sy':
                                matching_tokens_out.append(asset)
                
                if not matching_tokens_out:
                    return f"Error: No token found with symbol '{token_out_symbol}'"
                
                # If multiple tokens match, use the first one or the one with matching type
                token_out_asset = matching_tokens_out[0]
                token_out = token_out_asset.get('address', '')
            
            # Step 3: Determine if we should enable the aggregator
            # Enable aggregator if the output token is not directly in the market's output tokens
            output_token_address = None
            if isinstance(token_out, str) and token_out.startswith('146-'):
                output_token_address = token_out.split('-')[1].lower()
            else:
                output_token_address = token_out.lower()
                
            enable_aggregator = True
            for market_token in market_output_tokens:
                try:
                    parsed_token = self._agent.connection_manager.connections["pendle"].parse_token_id(market_token)
                    if parsed_token['address'].lower() == output_token_address:
                        enable_aggregator = False
                        break
                except:
                    continue
            if enable_aggregator:
                return f"Error: Token '{token_out_symbol}' is not a valid output token for market '{market_symbol}'"
            amount_in = int(float(amount_in) * 10**18)
            # Step 4: Call the Pendle connection to remove dual liquidity
            result = self._agent.connection_manager.connections["pendle"].remove_liquidity_dual(
                market_address=market_address,
                token_out=token_out,
                amount_in=amount_in,
                slippage=slippage,
                user_address=user_address,
                enable_aggregator=enable_aggregator
            )
            
            # Format the output for better readability
            output = {
                "success": True,
                "type": "Remove Dual Liquidity",
                "market": market_symbol,
                "market_address": market_address,
                "market_decimals": 18,
                "token_out": token_out_symbol,
                "token_out_type": token_out_type or "Standard",
                "lp_tokens_removed": amount_in,
                "user_address": user_address,
                "tokens_received": (result["data"]["amountTokenOut"]),
                "pt_tokens_received": (result["data"]["amountPtOut"]),
                "price_impact": f"{result['data']['priceImpact']:.4f}%",
                "aggregator_used": enable_aggregator,
                "transaction": {
                    "to": result["tx"]["to"],
                    "data": result["tx"]["data"],
                    "value": result["tx"].get("value")
                }
            }
            
            return json.dumps(output, indent=2)
            
        except Exception as e:
            return f"Error removing dual liquidity from Pendle market: {str(e)}"

class PendleSwapTool(BaseTool):
    name: str = "pendle_swap"
    description: str = """
    pendle_swap: Swap tokens in a Pendle market using user-friendly token symbols.
    
    Examples:
      - Swap 100 USDC for PT in the stS market: pendle_swap(market_symbol="stS", token_in_symbol="USDC", token_out_symbol="PT", amount_in="100", user_address="0x789...")
      - Swap 50 YT for stS SY with 0.5% slippage: pendle_swap(market_symbol="stS", token_in_symbol="YT", token_out_symbol="stS", token_out_type="SY", amount_in="50", slippage=0.005, user_address="0x789...")
      - Swap 10 USDC for YT in the stETH market: pendle_swap(market_symbol="stETH", token_in_symbol="USDC", token_out_symbol="YT", amount_in="10", user_address="0x789...")
    
    Args:
        market_symbol: Symbol of the Pendle market (e.g., "stS", "stETH", etc.)
        token_in_symbol: Symbol of the input token
        token_out_symbol: Symbol of the output token
        token_in_type: Optional type of input token (e.g., "SY", "PT", "YT") if needed
        token_out_type: Optional type of output token (e.g., "SY", "PT", "YT") if needed
        amount_in: The amount of the input token to swap
        user_address: The address of the user performing the swap
        slippage: Maximum slippage tolerance (default: 0.01 = 1%)
    """
    
    def __init__(self, agent):
        super().__init__()
        self._agent = agent
    
    def _run(self, market_symbol: str, token_in_symbol: str, token_out_symbol: str, 
             amount_in: str, user_address: str, token_in_type: str = None, 
             token_out_type: str = None, slippage: float = 0.01) -> str:
        try:
            # Step 1: Find the market by symbol
            markets = self._agent.connection_manager.connections["pendle"].get_markets()
            target_market = None
            
            for market in markets:
                if market.get('symbol', '').lower() == market_symbol.lower():
                    target_market = market
                    break
            
            if not target_market:
                return f"Error: No Pendle market found with symbol '{market_symbol}'"
            
            market_address = target_market['address']
            market_input_tokens = target_market.get('tokensMintSy', [])
            market_output_tokens = target_market.get('tokensRedeemSy', [])
            market_input_tokens.append(target_market.get('sy', ''))
            market_output_tokens.append(target_market.get('sy', ''))
            # Step 2: Find the input token by symbol and type
            assets = self._agent.connection_manager.connections["pendle"].get_assets()
            token_decimals = 18
            
            # Special handling for PT and YT tokens
            if token_in_symbol.lower() == "pt":
                token_in = target_market.get('pt', '')
            elif token_in_symbol.lower() == "yt":
                token_in = target_market.get('yt', '')
            else:
                # Regular token lookup
                matching_tokens_in = []
                for asset in assets:
                    if asset.get('symbol', '').lower() == token_in_symbol.lower():
                        # If token_type is specified, check if it matches
                        if token_in_type:
                            if asset.get('type', '').lower() == token_in_type.lower():
                                matching_tokens_in.append(asset)
                        else:
                            if asset.get('type', '').lower() != 'sy':
                                matching_tokens_in.append(asset)
                
                if not matching_tokens_in:
                    return f"Error: No token found with symbol '{token_in_symbol}'"
                
                # If multiple tokens match, use the first one or the one with matching type
                token_in_asset = matching_tokens_in[0]
                token_in = token_in_asset.get('address', '')
            
            # Step 3: Find the output token by symbol and type
            if token_out_symbol.lower() == "pt":
                token_out = target_market.get('pt', '')
            elif token_out_symbol.lower() == "yt":
                token_out = target_market.get('yt', '')
            else:
                # Regular token lookup
                matching_tokens_out = []
                for asset in assets:
                    if asset.get('symbol', '').lower() == token_out_symbol.lower():
                        # If token_type is specified, check if it matches
                        if token_out_type:
                            if asset.get('type', '').lower() == token_out_type.lower():
                                matching_tokens_out.append(asset)
                        else:
                            if asset.get('type', '').lower() != 'sy':
                                matching_tokens_out.append(asset)
                
                if not matching_tokens_out:
                    return f"Error: No token found with symbol '{token_out_symbol}'"
                
                # If multiple tokens match, use the first one or the one with matching type
                token_out_asset = matching_tokens_out[0]
                token_out = token_out_asset.get('address', '')
            
            # Step 4: Determine if we should enable the aggregator
            # Enable aggregator if the input token is not directly in the market's input tokens
            input_token_address = None
            output_token_address = None
            if token_in.startswith('146-'):
                input_token_address = token_in.split('-')[1].lower()
            else:
                input_token_address = token_in.lower()
            
            for asset in assets:
                if asset.get('address', '').lower() == input_token_address:
                    token_decimals = asset.get('decimals', 18)
                    break
            if token_out.startswith('146-'):
                output_token_address = token_out.split('-')[1].lower()
            else:
                output_token_address = token_out.lower()
            amount_in = int(float(amount_in) * 10**token_decimals)
            enable_aggregator = True
            for market_token in market_input_tokens:
                try:
                    parsed_token = self._agent.connection_manager.connections["pendle"].parse_token_id(market_token)
                    if parsed_token['address'].lower() == input_token_address:
                        enable_aggregator = False
                        break
                except:
                    continue
            
            # Step 5: Call the Pendle connection to perform the swap
            result = self._agent.connection_manager.connections["pendle"].swap(
                market_address=market_address,
                token_in=input_token_address,
                token_out=output_token_address,
                amount_in=amount_in,
                slippage=slippage,
                user_address=user_address,
                enable_aggregator=enable_aggregator
            )
            
            # Format the output for better readability
            output = {
                "success": True,
                "type": "Swap",
                "market": market_symbol,
                "market_address": market_address,
                "token_in": token_in_symbol,
                "token_out": token_out_symbol,
                "token_address": input_token_address,
                "token_decimals": token_decimals,
                "token_in_type": token_in_type or "Standard",
                "token_out_type": token_out_type or "Standard",
                "amount_in": amount_in,
                "user_address": user_address,
                "amount_out": (result["data"]["amountOut"]),
                "price_impact": f"{result['data']['priceImpact']:.4f}%",
                "aggregator_used": enable_aggregator,
                "transaction": {
                    "to": result["tx"]["to"],
                    "data": result["tx"]["data"],
                    "value": result["tx"].get("value")
                }
            }
            
            return json.dumps(output, indent=2)
            
        except Exception as e:
            return f"Error swapping tokens in Pendle market: {str(e)}"

class PendleMintSyTool(BaseTool):
    name: str = "pendle_mint_sy"
    description: str = """
    pendle_mint_sy: Mint SY (Standardized Yield) tokens from a base token using user-friendly token symbols.
    
    Examples:
      - Mint stS SY tokens from 100 USDC: pendle_mint_sy(sy_symbol="stS", token_in_symbol="USDC", amount_in="100", user_address="0x789...")
      - Mint wstETH SY tokens from 10 ETH with 0.5% slippage: pendle_mint_sy(sy_symbol="wstETH", token_in_symbol="ETH", amount_in="10", slippage=0.005, user_address="0x789...")
    
    Args:
        sy_symbol: Symbol of the SY token to mint (e.g., "stS", "wstETH", etc.)
        token_in_symbol: Symbol of the input token
        token_in_type: Optional type of input token if needed to distinguish
        amount_in: The amount of the input token to convert to SY
        user_address: The address of the user minting SY
        slippage: Maximum slippage tolerance (default: 0.01 = 1%)
    """
    
    def __init__(self, agent):
        super().__init__()
        self._agent = agent
    
    def _run(self, sy_symbol: str, token_in_symbol: str, amount_in: str, user_address: str,
             token_in_type: str = None, slippage: float = 0.01) -> str:
        try:
            # Step 1: Find the SY token by symbol
            assets = self._agent.connection_manager.connections["pendle"].get_assets()
            sy_assets = [asset for asset in assets if asset.get('type') == 'SY']
            
            target_sy = None
            for asset in sy_assets:
                if asset.get('symbol', '').lower() == sy_symbol.lower():
                    target_sy = asset
                    break
            
            if not target_sy:
                return f"Error: No SY token found with symbol '{sy_symbol}'"
            
            sy_address = target_sy['address']
            
            # Step 2: Find the input token by symbol and type
            matching_tokens_in = []
            for asset in assets:
                if asset.get('symbol', '').lower() == token_in_symbol.lower():
                    # If token_in_type is specified, check if it matches
                    if token_in_type:
                        if asset.get('type', '').lower() == token_in_type.lower():
                            matching_tokens_in.append(asset)
                    else:
                        if asset.get('type', '').lower() != 'sy':
                            matching_tokens_in.append(asset)
            
            if not matching_tokens_in:
                return f"Error: No token found with symbol '{token_in_symbol}'"
            
            # If multiple tokens match, use the first one or the one with matching type
            token_in_asset = matching_tokens_in[0]
            token_in_decimals = token_in_asset.get('decimals', 18)
            token_in_address = token_in_asset['address']
            
            # Step 3: Determine if we should enable the aggregator
            # Check if the token is directly supported for minting the SY
            enable_aggregator = True
            markets = self._agent.connection_manager.connections["pendle"].get_markets()
            for market in markets:
                if market.get('sy') == sy_address:
                    input_tokens = [
                        self._agent.connection_manager.connections["pendle"].parse_token_id(t)['address'].lower() 
                        for t in market.get('tokensMintSy', [])
                    ]
                    if token_in_address.lower() in input_tokens:
                        enable_aggregator = False
                        break
            amount_in = int(float(amount_in) * 10**token_in_decimals)
            # Step 4: Call the Pendle connection to mint SY
            result = self._agent.connection_manager.connections["pendle"].mint_sy(
                sy=sy_address,
                token_in=token_in_address,
                amount_in=amount_in,
                slippage=slippage,
                user_address=user_address,
                enable_aggregator=enable_aggregator
            )
            
            # Format the output for better readability
            output = {
                "success": True,
                "type": "Mint SY",
                "sy_symbol": sy_symbol,
                "sy_address": sy_address,
                "token_in": token_in_symbol,
                "token_in_type": token_in_type or token_in_asset.get('type', 'Unknown'),
                "token_in_address": token_in_address,
                "token_in_decimals": token_in_decimals,
                "amount_in": amount_in,
                "user_address": user_address,
                "sy_tokens_received": (result["data"]["amountOut"]),
                "price_impact": f"{result['data']['priceImpact']:.4f}%",
                "aggregator_used": enable_aggregator,
                "transaction": {
                    "to": result["tx"]["to"],
                    "data": result["tx"]["data"],
                    "value": result["tx"].get("value")
                }
            }
            
            return json.dumps(output, indent=2)
            
        except Exception as e:
            return f"Error minting SY tokens: {str(e)}"

class PendleMintPyTool(BaseTool):
    name: str = "pendle_mint_py"
    description: str = """
    pendle_mint_py: Mint PT (Principal Token) and YT (Yield Token) from SY or a base token using user-friendly token symbols.
    
    Examples:
      - Mint PT and YT for the stS market from 100 USDC: pendle_mint_py(market_symbol="stS", token_in_symbol="USDC", amount_in="100", user_address="0x789...")
      - Mint PT and YT for the wstETH market from 50 stETH with 0.5% slippage: pendle_mint_py(market_symbol="wstETH", token_in_symbol="stETH", token_in_type="UNDERLYING", amount_in="50", slippage=0.005, user_address="0x789...")
      - Mint PT and YT using SY tokens directly: pendle_mint_py(market_symbol="stS", token_in_symbol="stS", token_in_type="SY", amount_in="100", user_address="0x789...")
    
    Args:
        market_symbol: Symbol of the Pendle market (e.g., "stS", "wstETH", etc.)
        token_in_symbol: Symbol of the input token
        token_in_type: Optional type of input token (e.g., "SY", "UNDERLYING") if needed to distinguish
        amount_in: The amount of the input token to convert to PT/YT
        user_address: The address of the user minting PT/YT
        slippage: Maximum slippage tolerance (default: 0.01 = 1%)
    """
    
    def __init__(self, agent):
        super().__init__()
        self._agent = agent
    
    def _run(self, market_symbol: str, token_in_symbol: str, amount_in: str, user_address: str,
             token_in_type: str = None, slippage: float = 0.01) -> str:
        try:
            # Step 1: Find the market by symbol
            markets = self._agent.connection_manager.connections["pendle"].get_markets()
            target_market = None
            
            for market in markets:
                if market.get('symbol', '').lower() == market_symbol.lower():
                    target_market = market
                    break
            
            if not target_market:
                return f"Error: No Pendle market found with symbol '{market_symbol}'"
            
            # Get the YT address from the market
            yt_address = target_market.get('yt', '')
            if not yt_address:
                return f"Error: Could not find YT token for market '{market_symbol}'"
            
            # Step 2: Find the input token by symbol and type
            assets = self._agent.connection_manager.connections["pendle"].get_assets()
            
            # Special handling for SY token from the market
            if token_in_symbol.lower() == "sy" or (token_in_symbol.lower() == market_symbol.lower() and token_in_type and token_in_type.lower() == "sy"):
                token_in = target_market.get('sy', '')
                if not token_in:
                    return f"Error: Could not find SY token for market '{market_symbol}'"
            else:
                # Regular token lookup
                matching_tokens_in = []
                for asset in assets:
                    if asset.get('symbol', '').lower() == token_in_symbol.lower():
                        # If token_type is specified, check if it matches
                        if token_in_type:
                            if asset.get('type', '').lower() == token_in_type.lower():
                                matching_tokens_in.append(asset)
                        else:
                            if asset.get('type', '').lower() != 'sy':
                                matching_tokens_in.append(asset)
                
                if not matching_tokens_in:
                    return f"Error: No token found with symbol '{token_in_symbol}'"
                
                # If multiple tokens match, use the first one or the one with matching type
                token_in_asset = matching_tokens_in[0]
                token_in = token_in_asset.get('address', '')
            
            # Step 3: Determine if we should enable the aggregator
            # Check if the token is directly supported for minting with YT
            enable_aggregator = True
            input_tokens = [
                self._agent.connection_manager.connections["pendle"].parse_token_id(t)['address'].lower() 
                for t in target_market.get('tokensMintSy', [])
            ]
            input_tokens.append(self._agent.connection_manager.connections["pendle"].parse_token_id(target_market.get('sy'))['address'].lower())
            
            input_token_address = None
            if isinstance(token_in, str) and token_in.startswith('146-'):
                input_token_address = token_in.split('-')[1].lower()
            else:
                input_token_address = token_in.lower()
            
            if isinstance(yt_address, str) and yt_address.startswith('146-'):
                input_yt_address = yt_address.split('-')[1].lower()
            else:
                input_yt_address = yt_address.lower()
                
            if input_token_address.lower() in input_tokens:
                enable_aggregator = False
            
            input_token_decimals = 18
            for asset in assets:
                if asset.get('address', '').lower() == input_token_address:
                    input_token_decimals = asset.get('decimals', 18)
                    break
            amount_in = int(float(amount_in) * 10**input_token_decimals)
            
            # Step 4: Call the Pendle connection to mint PT and YT
            result = self._agent.connection_manager.connections["pendle"].mint_py(
                yt=input_yt_address,
                token_in=input_token_address,
                amount_in=amount_in,
                slippage=slippage,
                user_address=user_address,
                enable_aggregator=enable_aggregator
            )
            
            # Format the output for better readability
            output = {
                "success": True,
                "type": "Mint PT and YT",
                "market": market_symbol,
                "yt_address": input_yt_address,
                "pt_address": target_market.get('pt', ''),
                "token_in": token_in_symbol,
                "token_in_type": token_in_type or (
                    "SY" if token_in == target_market.get('sy', '') else 
                    (matching_tokens_in[0].get('type', 'Unknown') if 'matching_tokens_in' in locals() and matching_tokens_in else "Unknown")
                ),
                "token_in_address": input_token_address,
                "token_in_decimals": input_token_decimals,
                "amount_in": amount_in,
                "user_address": user_address,
                "tokens_received": (result["data"]["amountOut"]),
                "price_impact": f"{result['data']['priceImpact']:.4f}%",
                "aggregator_used": enable_aggregator,
                "transaction": {
                    "to": result["tx"]["to"],
                    "data": result["tx"]["data"],
                    "value": result["tx"].get("value")
                },
                "note": "Equal amounts of PT and YT tokens are minted"
            }
            
            return json.dumps(output, indent=2)
            
        except Exception as e:
            return f"Error minting PT and YT tokens: {str(e)}"

class PendleRedeemSyTool(BaseTool):
    name: str = "pendle_redeem_sy"
    description: str = """
    pendle_redeem_sy: Redeem SY (Standardized Yield) tokens to a base token using user-friendly token symbols.
    
    Examples:
      - Redeem 100 stS SY tokens to USDC: pendle_redeem_sy(sy_symbol="stS", token_out_symbol="USDC", amount_in="100", user_address="0x789...")
      - Redeem 50 wstETH SY tokens to WETH with 0.5% slippage: pendle_redeem_sy(sy_symbol="wstETH", token_out_symbol="WETH", amount_in="50", slippage=0.005, user_address="0x789...")
      - Redeem 25 stS SY tokens to a specific token type: pendle_redeem_sy(sy_symbol="stS", token_out_symbol="USDC", token_out_type="UNDERLYING", amount_in="25", user_address="0x789...")
    
    Args:
        sy_symbol: Symbol of the SY token to redeem (e.g., "stS", "wstETH", etc.)
        token_out_symbol: Symbol of the output/base token you want to receive
        token_out_type: Optional type of output token if needed to distinguish
        amount_in: The amount of SY tokens to redeem
        user_address: The address of the user redeeming SY
        slippage: Maximum slippage tolerance (default: 0.01 = 1%)
    """
    
    def __init__(self, agent):
        super().__init__()
        self._agent = agent
    
    def _run(self, sy_symbol: str, token_out_symbol: str, amount_in: str, user_address: str,
             token_out_type: str = None, slippage: float = 0.01) -> str:
        try:
            # Step 1: Find the SY token by symbol
            assets = self._agent.connection_manager.connections["pendle"].get_assets()
            sy_assets = [asset for asset in assets if asset.get('type', '').lower() == 'sy']
            
            target_sy = None
            for asset in sy_assets:
                if asset.get('symbol', '').lower() == sy_symbol.lower():
                    target_sy = asset
                    break
            
            if not target_sy:
                return f"Error: No SY token found with symbol '{sy_symbol}'"
            
            sy_address = target_sy['address']
            sy_decimals = target_sy.get('decimals', 18)
            # Step 2: Find the output token by symbol and type
            matching_tokens_out = []
            for asset in assets:
                if asset.get('symbol', '').lower() == token_out_symbol.lower():
                    # If token_out_type is specified, check if it matches
                    if token_out_type:
                        if asset.get('type', '').lower() == token_out_type.lower():
                            matching_tokens_out.append(asset)
                    else:
                        if asset.get('type', '').lower() != 'sy':
                            matching_tokens_out.append(asset)
            
            if not matching_tokens_out:
                return f"Error: No token found with symbol '{token_out_symbol}'"
            
            # If multiple tokens match, use the first one or the one with matching type
            token_out_asset = matching_tokens_out[0]
            token_out_address = token_out_asset['address']
            
            # Step 3: Determine if we should enable the aggregator
            # Check if the token is directly supported for redeeming from this SY
            enable_aggregator = True
            markets = self._agent.connection_manager.connections["pendle"].get_markets()
            for market in markets:
                if market.get('sy') == sy_address:
                    output_tokens = [
                        self._agent.connection_manager.connections["pendle"].parse_token_id(t)['address'].lower() 
                        for t in market.get('tokensRedeemSy', [])
                    ]
                    
                    if token_out_address.lower() in output_tokens:
                        enable_aggregator = False
                        break
            amount_in = int(float(amount_in) * 10**sy_decimals)
            # Step 4: Call the Pendle connection to redeem SY
            result = self._agent.connection_manager.connections["pendle"].redeem_sy(
                sy=sy_address,
                token_out=token_out_address,
                amount_in=amount_in,
                slippage=slippage,
                user_address=user_address,
                enable_aggregator=enable_aggregator
            )
            
            # Format the output for better readability
            output = {
                "success": True,
                "type": "Redeem SY",
                "sy_symbol": sy_symbol,
                "sy_address": sy_address,
                "sy_decimals": sy_decimals,
                "token_out": token_out_symbol,
                "token_out_type": token_out_type or token_out_asset.get('type', 'Unknown'),
                "amount_in": amount_in,
                "user_address": user_address,
                "tokens_received": (result["data"]["amountOut"]),
                "price_impact": f"{result['data']['priceImpact']:.4f}%",
                "aggregator_used": enable_aggregator,
                "transaction": {
                    "to": result["tx"]["to"],
                    "data": result["tx"]["data"],
                    "value": result["tx"].get("value")
                }
            }
            
            return json.dumps(output, indent=2)
            
        except Exception as e:
            return f"Error redeeming SY tokens: {str(e)}"

class PendleRedeemPyTool(BaseTool):
    name: str = "pendle_redeem_py"
    description: str = """
    pendle_redeem_py: Redeem PT (Principal Token) and YT (Yield Token) to SY or a base token using user-friendly token symbols.
    
    Examples:
      - Redeem PT+YT from stS market to USDC: pendle_redeem_py(market_symbol="stS", token_out_symbol="USDC", amount_in="100", user_address="0x789...")
      - Redeem PT+YT from wstETH market to ETH with 0.5% slippage: pendle_redeem_py(market_symbol="wstETH", token_out_symbol="ETH", token_out_type="UNDERLYING", amount_in="50", slippage=0.005, user_address="0x789...")
      - Redeem PT+YT directly to SY: pendle_redeem_py(market_symbol="stS", token_out_symbol="stS", token_out_type="SY", amount_in="25", user_address="0x789...")
    
    Args:
        market_symbol: Symbol of the Pendle market containing the PT/YT tokens
        token_out_symbol: Symbol of the output token you want to receive
        token_out_type: Optional type of output token if needed to distinguish
        amount_in: The amount of PT/YT tokens to redeem (must have equal amounts of both)
        user_address: The address of the user redeeming PT/YT
        slippage: Maximum slippage tolerance (default: 0.01 = 1%)
    """
    
    def __init__(self, agent):
        super().__init__()
        self._agent = agent
    
    def _run(self, market_symbol: str, token_out_symbol: str, amount_in: str, user_address: str,
             token_out_type: str = None, slippage: float = 0.01) -> str:
        try:
            # Step 1: Find the market by symbol
            markets = self._agent.connection_manager.connections["pendle"].get_markets()
            target_market = None
            
            for market in markets:
                if market.get('symbol', '').lower() == market_symbol.lower():
                    target_market = market
                    break
            
            if not target_market:
                return f"Error: No Pendle market found with symbol '{market_symbol}'"
            
            # Get YT address from the market
            yt_address = target_market.get('yt', '')
            if not yt_address:
                return f"Error: Could not find YT token for market '{market_symbol}'"
            
            # Step 2: Find the output token by symbol and type
            assets = self._agent.connection_manager.connections["pendle"].get_assets()
            
            # Special handling for SY token from the market
            if token_out_symbol.lower() == "sy" or (token_out_symbol.lower() == market_symbol.lower() and token_out_type and token_out_type.lower() == "sy"):
                token_out = target_market.get('sy', '')
                if not token_out:
                    return f"Error: Could not find SY token for market '{market_symbol}'"
                token_out_asset = next((asset for asset in assets if asset.get('address') == target_market.get('sy')), None)
            else:
                # Regular token lookup
                matching_tokens_out = []
                for asset in assets:
                    if asset.get('symbol', '').lower() == token_out_symbol.lower():
                        # If token_out_type is specified, check if it matches
                        if token_out_type:
                            if asset.get('type', '').lower() == token_out_type.lower():
                                matching_tokens_out.append(asset)
                        else:
                            matching_tokens_out.append(asset)
                
                if not matching_tokens_out:
                    return f"Error: No token found with symbol '{token_out_symbol}'"
                
                # If multiple tokens match, use the first one or the one with matching type
                token_out_asset = matching_tokens_out[0]
                token_out = token_out_asset.get('address', '')
            
            # Step 3: Determine if we should enable the aggregator
            # Check if the output token is directly supported for redeeming
            enable_aggregator = True
            output_tokens = [
                self._agent.connection_manager.connections["pendle"].parse_token_id(t)['address'].lower() 
                for t in target_market.get('tokensRedeemSy', [])
            ]
            output_tokens.append(self._agent.connection_manager.connections["pendle"].parse_token_id(target_market.get('sy'))['address'].lower())
            
            output_token_address = None
            if isinstance(token_out, str) and token_out.startswith('146-'):
                output_token_address = token_out.split('-')[1].lower()
            else:
                output_token_address = token_out.lower()
            if isinstance(yt_address, str) and yt_address.startswith('146-'):
                input_yt_address = yt_address.split('-')[1].lower()
            else:
                input_yt_address = yt_address.lower()
            input_pt_address = target_market.get('pt', '')
            if input_pt_address and input_pt_address.startswith('146-'):
                input_pt_address = input_pt_address.split('-')[1].lower()
            else:
                input_pt_address = input_pt_address.lower()
            if output_token_address.lower() in output_tokens:
                enable_aggregator = False
            amount_in = int(float(amount_in) * 10**18)
            # Step 4: Call the Pendle connection to redeem PT and YT
            result = self._agent.connection_manager.connections["pendle"].redeem_py(
                yt=input_yt_address,
                token_out=token_out,
                amount_in=amount_in,
                slippage=slippage,
                user_address=user_address,
                enable_aggregator=enable_aggregator
            )
            
            # Format the output for better readability
            output = {
                "success": True,
                "type": "Redeem PT and YT",
                "market": market_symbol,
                "yt_address": input_yt_address,
                "pt_address": input_pt_address,
                "token_out": token_out_symbol,
                "token_out_type": token_out_type or (
                    "SY" if token_out == target_market.get('sy', '') else
                    (token_out_asset.get('type', 'Unknown') if token_out_asset else "Unknown")
                ),
                "amount_in": amount_in,
                "user_address": user_address,
                "tokens_received": (result["data"]["amountOut"]),
                "price_impact": f"{result['data']['priceImpact']:.4f}%",
                "aggregator_used": enable_aggregator,
                "transaction": {
                    "to": result["tx"]["to"],
                    "data": result["tx"]["data"],
                    "value": result["tx"].get("value")
                },
                "note": "Equal amounts of PT and YT tokens are needed for redemption"
            }
            
            return json.dumps(output, indent=2)
            
        except Exception as e:
            return f"Error redeeming PT and YT tokens: {str(e)}"

def get_all_pendle_tools(agent):
    """Get all Pendle-related tools"""
    return [
        PendleMarketsInfoTool(agent),
        PendleAssetsInfoTool(agent),
        PendleMarketDetailsTool(agent),
        PendleAddLiquidityTool(agent),
        PendleAddLiquidityDualTool(agent),
        PendleRemoveLiquidityTool(agent),
        PendleRemoveLiquidityDualTool(agent),
        PendleSwapTool(agent),
        PendleMintSyTool(agent),
        PendleMintPyTool(agent),
        PendleRedeemSyTool(agent),
        PendleRedeemPyTool(agent),
    ]