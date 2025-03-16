from typing import Optional, Tuple
import json
import logging
import requests
from langchain.tools import BaseTool
from .sonic_tools import SonicBalanceCheckTool

logger = logging.getLogger("tools.silo_tools")

def get_silo_config_address(token_0: str, token_1: str, id: Optional[int] = None) -> Tuple[str, bool, str]:
    """Get silo config address for given token pair from API.
    
    Args:
        token_0: First token symbol
        token_1: Second token symbol
        
    Returns:
        Tuple of (config_address, is_token0_silo0) where is_token0_silo0 indicates if token_0 
        corresponds to silo0 in the pair
    """
    url = "https://shiami.me/api/silo/markets"
    headers = {
        "Content-Type": "application/json"
    }
    if id is not None:
        id = int(id)
    response = requests.get(url, headers=headers)
    data = response.json()
    for market in data["markets"]:
        market["id"] = int(market["id"])
        if id is not None and market["id"] != id:
            continue
        silo0_symbol = market["silo0"]["symbol"]
        silo1_symbol = market["silo1"]["symbol"]
        token0_contract = market["silo0"]["tokenAddress"]
        token1_contract = market["silo1"]["tokenAddress"]
        token0_decimals = market["silo0"]["decimals"]
        token1_decimals = market["silo1"]["decimals"]
        # Check both orderings of the pair
        if market["id"] == id:
            if silo0_symbol == token_0:
                return market["configAddress"], True, token0_contract, token0_decimals
            elif silo1_symbol == token_0:
                return market["configAddress"], False, token1_contract, token1_decimals
        if (silo0_symbol == token_0 and (silo1_symbol == token_1)):
            return market["configAddress"], True, token0_contract, token0_decimals
        elif (silo0_symbol == token_1 and silo1_symbol == token_0):
            return market["configAddress"], False, token1_contract, token1_decimals
            
    raise ValueError(f"No silo found for token pair {token_0}/{token_1}")

def get_position_details(silo_address: str, sender: str) -> dict:
    """Get details of a user's position in a Silo pair.
    
    Args:
        silo_address: Address of the Silo contract
        sender: Address of the user
    
    Returns:
        Dict of position details
    """
    url = "https://shiami.me/api/silo/user/position"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "siloAddress": silo_address,
        "userAddress": sender
    }

    response = requests.get(url, headers=headers, params=payload)
    response.raise_for_status()
    data = response.json()
    return data["position"]

def get_silo_pools(tokens: str = None) -> dict:
    """Get available Silo pools, optionally filtered by tokens.
    
    Args:
        tokens: Comma-separated list of token symbols to filter by (e.g. "S,USDC")
        
    Returns:
        Dict containing markets data
    """
    url = "https://shiami.me/api/silo/filter"
    headers = {
        "Content-Type": "application/json"
    }
    params = {}
    if tokens:
        params["tokens"] = tokens
    
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    # Process market data to format it nicely
    formatted_markets = []
    for item in data["markets"]:
        # Format first token in the pair
        silo0Logo = item["silo0"]["logos"]["trustWallet"] or item["silo0"]["logos"]["coinGecko"] or item["silo0"]["logos"]["coinMarketCap"] or { "large": "https://coin-images.coingecko.com/coins/images/52857/large/wrapped_sonic.png?1734536585" }
        silo1Logo = item["silo1"]["logos"]["trustWallet"] or item["silo1"]["logos"]["coinGecko"] or item["silo1"]["logos"]["coinMarketCap"] or { "large": "https://coin-images.coingecko.com/coins/images/52857/large/wrapped_sonic.png?1734536585" }
        formatted_markets.append({
            "id": item["id"],
            "reviewed": item["isVerified"],
            "market": item["silo0"]["symbol"] + "/" + item["silo1"]["symbol"],
            "silo0": {
                "market": item["silo0"]["symbol"],
                "deposit_apr": f"{(float(item['silo0']['collateralBaseApr']) / 10**18) * 100:.2f}%",
                "borrow_apr": f"{(float(item['silo0']['debtBaseApr']) / 10**18) * 100:.2f}%",
                "isBorrowable": not item["silo0"]["isNonBorrowable"],
                "token0": item["silo0"]["symbol"],
                "token1": item["silo1"]["symbol"],
                "max_ltv": int(item["silo0"]["maxLtv"]) / 10**18,
                "lt": int(item["silo0"]["lt"]) / 10**18,
                "liquidity": int(item["silo0"]["liquidity"]) / 10**(item["silo0"]["decimals"]),
                "logo": silo0Logo["large"]
            },
            "silo1": {
                "market": item["silo1"]["symbol"],
                "deposit_apr": f"{(float(item['silo1']['collateralBaseApr']) / 10**18) * 100:.2f}%",
                "borrow_apr": f"{(float(item['silo1']['debtBaseApr']) / 10**18) * 100:.2f}%",
                "isBorrowable": not item["silo1"]["isNonBorrowable"],
                "token0": item["silo1"]["symbol"],
                "token1": item["silo0"]["symbol"],
                "max_ltv": int(item["silo1"]["maxLtv"]) / 10**18,
                "lt": int(item["silo1"]["lt"]) / 10**18,
                "liquidity": int(item["silo1"]["liquidity"]) / 10**(item["silo1"]["decimals"]),
                "logo": silo1Logo["large"]
            }
        })
    
    return {
        "requestedTokens": tokens,
        "markets": formatted_markets,
        "count": len(formatted_markets),
        "timestamp": data.get("timestamp")
    }

class SiloPositionTool(BaseTool):
    name: str = "silo_position"
    description: str = """
    silo_position: Get details of a user's position in a Silo pair.
    Ex - get position details for Sonic/USDC pair. Then token_0: S, token_1: USDC, sender: <user_address>
         get position details for S/USDC market with ID 1. Then id: 1, token_0: S, token_1: USDC, sender: <user_address>
    Args:
        token_0: Symbol of the token in the Silo pair
        token_1: Symbol of the other token in the Silo pair
        sender: Address of the user(connected wallet)
        id: Optional market ID to specify a specific market (useful when multiple markets exist for the same token pair)
    """

    def __init__(self, agent, llm):
        super().__init__()
        self._agent = agent
        self._llm = llm

    def _run(self, token_0: str, token_1: str, sender: str, id: Optional[int] = None) -> str:
        try:
            silo_config_address, is_token0_silo0, _, decimals0 = get_silo_config_address(token_0, token_1, id)
            _, _, _, decimals1 = get_silo_config_address(token_1, token_0, id)
            token_idx = 0 if is_token0_silo0 else 1
            silo0_address = self._agent.connection_manager.connections["silo"]._get_silo_address(silo_config_address, token_idx)
            
            silo1_address = self._agent.connection_manager.connections["silo"]._get_silo_address(silo_config_address, 1 - token_idx)
            position0 = get_position_details(silo0_address, sender)
            position1 = get_position_details(silo1_address, sender)
            result = {
                "token0": token_0,
                "token1": token_1,
                "position0": {
                    "regularDeposit": str(float(position0["regularDeposit"]) / 10**decimals0),
                    "protectedDeposit": str(float(position0["protectedDeposit"]) / 10**decimals0),
                    "totalCollateral": str(float(position0["totalCollateral"]) / 10**decimals0),
                    "borrowedAmount": str(float(position0["borrowedAmount"]) / 10**decimals0),
                    "loanToValue": str(float(position0["loanToValue"]) / 10**18),
                    "maxBorrowAmount": str(float(position0["maxBorrowAmount"]) / 10**decimals0),
                    "decimals": decimals0
                },
                "position1": {
                    "regularDeposit": str(float(position1["regularDeposit"]) / 10**decimals1),
                    "protectedDeposit": str(float(position1["protectedDeposit"]) / 10**decimals1),
                    "totalCollateral": str(float(position1["totalCollateral"]) / 10**decimals1),
                    "borrowedAmount": str(float(position1["borrowedAmount"]) / 10**decimals1),
                    "loanToValue": str(float(position1["loanToValue"]) / 10**18),
                    "maxBorrowAmount": str(float(position1["maxBorrowAmount"]) / 10**decimals1),
                    "decimals": decimals1
                }
            }
            return result
        except Exception as e:
            return f"Error getting position details: {str(e)}"

class SiloDepositTool(BaseTool):
    name: str = "silo_deposit"
    description: str = """
    silo_deposit: Deposit tokens into a Silo smart contract. Supports both Collateral (1) and Protected (0) deposits.
    Ex - deposit Collateral 1000 USDC into Sonic/USDC pair. Then token_0: USDC, token_1: Sonic, collateral_type: 1 (collateral), amount: 1000.0
        deposit Protected 100 Sonic into Sonic/USDC pair. Then token_0: Sonic, token_1: USDC, collateral_type: 0 (protected), amount: 100.0
        deposit Collateral 1000 USDC into market with ID 1. Then id: 1, token_0: USDC, token_1: Sonic, collateral_type: 1, amount: 1000.0
    Args:
        token_0: Symbol of the token to deposit
        token_1: Symbol of the other token in the Silo pair
        amount: Amount of assets to deposit
        collateral_type: Type of collateral (0 for Protected, 1 for Collateral)
        sender: Address of the sender (optional)
        id: Optional market ID to specify a specific market
    """
    
    def __init__(self, agent, llm):
        super().__init__()
        self._agent = agent
        self._llm = llm
    
    def _run(self, token_0: str, token_1: str, amount: float, 
             collateral_type: int = 0, sender: Optional[str] = None, id: Optional[int] = None) -> str:
        try:
            silo_config_address, is_token0_silo0, token_address, decimals = get_silo_config_address(token_0, token_1, id)
            # Get appropriate silo address based on token position
            token_idx = 0 if is_token0_silo0 else 1
            silo_address = self._agent.connection_manager.connections["silo"]._get_silo_address(silo_config_address, token_idx)
            if token_0.upper() == "S":
                token_0 = "wS"
                balance = json.loads(SonicBalanceCheckTool(self._agent)._run(sender, token_0))
                if float(balance["balance"]) < amount:
                    return {"error": f"Insufficient balance: {balance['balance']} wS. Silo accepts only wrapped Sonic tokens. Would you like me to wrap some?"}
            result = self._agent.connection_manager.connections["silo"].deposit(
                silo_address=silo_address,
                amount=amount,
                collateral_type=int(collateral_type),
                sender=sender,
                decimals=decimals
            )
            result["type"] = "deposit"
            result["tokenAddress"] = token_address
            result["status"] = "Initiated. Continue in the frontend."
            result["sender"] = sender
            return result
        except Exception as e:
            return f"Error depositing tokens: {str(e)}"

class SiloBorrowTool(BaseTool):
    name: str = "silo_borrow"
    description: str = """
    silo_borrow: Borrow tokens from a Silo smart contract.
    Ex - borrow 1000 USDC from Sonic/USDC pair. Then token_0: USDC, token_1: Sonic, amount: 1000.0
        borrow 100 Sonic from Sonic/USDC pair. Then token_0: Sonic, token_1: USDC, amount: 100.0
        borrow 1000 USDC from market with ID 1. Then id: 1, token_0: USDC, token_1: Sonic, amount: 1000.0
    Args:
        token_0: Symbol of the token to borrow
        token_1: Symbol of the other token in the Silo pair
        amount: Amount of tokens to borrow
        sender: Address of the sender
        receiver: Address to receive the borrowed assets (optional, defaults to sender)
        id: Optional market ID to specify a specific market
    """
    
    def __init__(self, agent, llm):
        super().__init__()
        self._agent = agent
        self._llm = llm
    
    def _run(self, token_0: str, token_1: str, amount: float, sender: str, 
             receiver: Optional[str] = None, id: Optional[int] = None) -> str:
        try:
            silo_config_address, is_token0_silo0, token_address, decimals = get_silo_config_address(token_0, token_1, id)
            # Get appropriate silo address based on token position
            token_idx = 0 if is_token0_silo0 else 1  
            silo_address = self._agent.connection_manager.connections["silo"]._get_silo_address(silo_config_address, token_idx)
            result = self._agent.connection_manager.connections["silo"].borrow(
                silo_address=silo_address,
                amount=amount,
                sender=sender,
                receiver=receiver,
                decimals=decimals
            )
            result["type"] = "borrow"
            result["tokenAddress"] = token_address
            result["status"] = "Initiated. Continue in the frontend."
            result["sender"] = sender
            return result
        except Exception as e:
            return f"Error borrowing tokens: {str(e)}"

class SiloRepayTool(BaseTool):
    name: str = "silo_repay"
    description: str = """
    silo_repay: Repay borrowed tokens to a Silo smart contract.
    Ex - repay 1000 USDC into Sonic/USDC pair. Then token_0: USDC, token_1: Sonic, amount: 1000.0
        repay 100 Sonic into Sonic/USDC pair. Then token_0: Sonic, token_1: USDC, amount: 100.0
        repay 1000 USDC into market with ID 1. Then id: 1, token_0: USDC, token_1: Sonic, amount: 1000.0
    Args:
        token_0: Symbol of the token to repay
        token_1: Symbol of the other token in the Silo pair
        amount: Amount of tokens to repay
        sender: Address of the sender (optional)
        id: Optional market ID to specify a specific market
    """
    
    def __init__(self, agent, llm):
        super().__init__()
        self._agent = agent
        self._llm = llm
    
    def _run(self, token_0: str, token_1: str, amount: float, sender: Optional[str] = None, id: Optional[int] = None) -> str:
        try:
            silo_config_address, is_token0_silo0, token_address, decimals = get_silo_config_address(token_0, token_1, id)
            # Get appropriate silo address based on token position
            token_idx = 0 if is_token0_silo0 else 1
            silo_address = self._agent.connection_manager.connections["silo"]._get_silo_address(silo_config_address, token_idx)
            if token_0.upper() == "S":
                token_0 = "wS"
                balance = json.loads(SonicBalanceCheckTool(self._agent)._run(sender, token_0))
                if float(balance["balance"]) < amount:
                    return {"error": f"Insufficient balance: {balance['balance']} wS. Silo accepts only wrapped Sonic tokens. Would you like me to wrap some?"}
            result = self._agent.connection_manager.connections["silo"].repay(
                silo_address=silo_address,
                amount=amount,
                sender=sender,
                decimals=decimals
            )
            result["type"] = "repay"
            result["tokenAddress"] = token_address
            result["status"] = "Initiated. Continue in the frontend."
            result["sender"] = sender
            return result
        except Exception as e:
            return f"Error repaying tokens: {str(e)}"

class SiloWithdrawTool(BaseTool):
    name: str = "silo_withdraw"
    description: str = """
    silo_withdraw: Withdraw tokens from a Silo smart contract. Supports both Collateral (1) and Protected (0) withdrawals.
    Ex - withdraw Collateral 1000 USDC from Sonic/USDC pair. Then token_0: USDC, token_1: Sonic, collateral_type: 1 (collateral), amount: 1000.0
        withdraw Protected 100 Sonic from Sonic/USDC pair. Then token_0: Sonic, token_1: USDC, collateral_type: 0 (protected), amount: 100.0
        withdraw Collateral 1000 USDC from market with ID 1. Then id: 1, token_0: USDC, token_1: Sonic, collateral_type: 1, amount: 1000.0
    Args:
        token_0: Symbol of the token to withdraw
        token_1: Symbol of the other token in the Silo pair
        amount: Amount of tokens to withdraw
        receiver: Address to receive the withdrawn assets (optional)
        collateral_type: Type of collateral (1 for Collateral, 0 for Protected)
        sender: Address of the sender (optional)
        id: Optional market ID to specify a specific market
    """
    
    def __init__(self, agent, llm):
        super().__init__()
        self._agent = agent
        self._llm = llm
    
    def _run(self, token_0: str, token_1: str, amount: float, receiver: Optional[str] = None, 
             collateral_type: int = 0, sender: Optional[str] = None, id: Optional[int] = None) -> str:
        try:
            silo_config_address, is_token0_silo0, token_address, decimals = get_silo_config_address(token_0, token_1, id)
            # Get appropriate silo address based on token position
            token_idx = 0 if is_token0_silo0 else 1
            silo_address = self._agent.connection_manager.connections["silo"]._get_silo_address(silo_config_address, token_idx)
            result = self._agent.connection_manager.connections["silo"].withdraw(
                silo_address=silo_address,
                amount=amount,
                receiver=receiver,
                collateral_type=int(collateral_type),
                sender=sender,
                decimals=decimals
            )
            result["type"] = "withdraw"
            result["tokenAddress"] = token_address
            result["status"] = "Initiated. Continue in the frontend."
            result["sender"] = sender
            return result
        except Exception as e:
            return f"Error withdrawing tokens: {str(e)}"
# silo connection has claim_rewards function that takes sender address as the arg
class SiloClaimRewardsTool(BaseTool):
    name: str = "silo_claim_rewards"
    description: str = """
    silo_claim_rewards: Claim rewards from a Silo smart contract.
    Ex - claim my silo rewards. Then sender: connected wallet <user_address>
    Args:
        sender: Address of the user(connected wallet)
    """
    
    def __init__(self, agent, llm):
        super().__init__()
        self._agent = agent
        self._llm = llm
    
    def _run(self, sender: str) -> str:
        try:
            result = self._agent.connection_manager.connections["silo"].claim_rewards(
                sender=sender
            )
            result["type"] = "claim_rewards"
            result["status"] = "Initiated. Continue in the frontend."
            result["sender"] = sender
            return result
        except Exception as e:
            return f"Error claiming rewards: {str(e)}"

class SiloPoolsTool(BaseTool):
    name: str = "silo_markets"
    description: str = """
    silo_markets: Get available Silo lending/borrowing markets, optionally filtered by tokens.
    
    Ex - get all Silo pools. Then tokens: "" (empty string or omit)
        get markets containing Sonic. Then tokens: "S" 
        get pools for Sonic and USDC. Then tokens: "S,USDC"
        get pools for ETH, S, USDC. Then tokens: "ETH,S,USDC"
        
    Args:
        tokens: Comma-separated list of token symbols to filter by (optional)
    """
    
    def __init__(self, agent, llm):
        super().__init__()
        self._agent = agent
        self._llm = llm
    
    def _run(self, tokens: Optional[str] = None) -> str:
        try:
            result = get_silo_pools(tokens)
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error getting Silo pools: {str(e)}"

class SiloLoopingStrategyTool(BaseTool):
    return_direct: bool = True
    name: str = "silo_looping_opportunities"
    description: str = """
    silo_looping_opportunities: Find optimal yield farming strategies through Silo deposit-borrow loops.
    
    This tool identifies Silo markets with favorable deposit/borrow spreads, calculates potential yields
    at different leverage levels, and provides step-by-step instructions.
    Also output max_leverage and apr, along with the outputs specific to the loops, like net_apr after n loops.
    
    Ex - find looping strategies for all markets. Then initial_amount: 1000 (default)
        find looping strategies for USDC. Then token: "USDC", initial_amount: 1000
        find top 5 looping strategies for S and stS. max loops 50. Then token: "S,stS", limit: 5, max_loops: 50
        find top 5 looping strategies. Then limit: 5, initial_amount: 10000.
        simulate looping strategy for S and USDC. Then token: "S,USDC", initial_amount: 1000, min_loops: 2, max_loops: 50
    
    Args:
        initial_amount: Initial capital to simulate looping with (default: 1000)
        token: Optional tokens symbol to filter by
        limit: Maximum number of strategies to return (default: 10)
        min_loops: Minimum number of loops to consider (default: 2)
        max_loops: Maximum number of loops to consider. Max Value = 50 (default: 50)
    """
    
    def __init__(self, agent, llm):
        super().__init__()
        self._agent = agent
        self._llm = llm
    
    def _calculate_loop_yields(self, deposit_apr, borrow_apr, initial_amount, loops, max_ltv):
        """Calculate yields for different loop iterations considering max LTV."""
        results = []
        
        total_deposit = initial_amount
        total_borrowed = 0
        max_leverage = 1 / (1 - max_ltv)
        max_yield = ((deposit_apr * max_leverage) - (borrow_apr * (max_leverage - 1)))
        for i in range(loops + 1):
            if i == 0:
                leverage = 1
                net_apr = deposit_apr
            else:
                # Calculate how much can be borrowed based on max LTV
                borrow_capacity = total_deposit * max_ltv
                borrowable_amount = borrow_capacity - total_borrowed
                loop_amount = borrowable_amount * 0.95  # Apply safety factor
                if loop_amount <= 0:
                    break  # Stop looping if no more can be borrowed
                
                total_deposit += loop_amount
                total_borrowed += loop_amount
                
                leverage = 1 + (total_borrowed / initial_amount)
                
                net_apr = ((total_deposit * deposit_apr) - (total_borrowed * borrow_apr)) / initial_amount

            results.append({
                "loops": i,
                "leverage": leverage,
                "net_apr": net_apr * 100,  # Convert to percentage
                "total_deposit": total_deposit,
                "total_borrowed": total_borrowed,
                "max_leverage": max_leverage,
                "max_yield": max_yield * 100
            })
        
        return results

    def _find_looping_opportunities(self, initial_amount=1000, token=None, limit=10, min_loops=2, max_loops=50):
        """Find and analyze looping opportunities across Silo markets."""
        # Get all available markets
        markets_data = json.loads(SiloPoolsTool(self._agent, self._llm)._run(tokens=token))
        markets = markets_data.get("markets", [])
        
        opportunities = []
        
        for market in markets:
            # Analyze both sides of the market (token0/token1 and token1/token0)
            market_id = market.get("id")
            is_verified = market.get("reviewed", False)
            
            silo0 = market.get("silo0", {})
            silo1 = market.get("silo1", {})
            
            # First direction: deposit silo0 token, borrow silo1 token
            if silo1.get("isBorrowable", False):
                token0 = silo0.get("market")  # Use market as token symbol
                token1 = silo1.get("market")  # Use market as token symbol
                # Convert APR strings to floats properly
                deposit_apr = float(silo0.get("deposit_apr", "0%").rstrip("%"))
                borrow_apr = float(silo1.get("borrow_apr", "0%").rstrip("%"))
                deposit_token_logo = silo0.get("logo")
                borrow_token_logo = silo1.get("logo")
                # Get liquidity (max borrowable amount)
                liquidity = silo1.get("liquidity", 0)
                if deposit_apr > borrow_apr:
                    spread = deposit_apr - borrow_apr
                    loop_results = self._calculate_loop_yields(
                        deposit_apr/100, borrow_apr/100, initial_amount, max_loops, silo0.get("max_ltv", 0.7)
                    )
                    
                    best_loop = max(loop_results, key=lambda x: x["net_apr"])
                    if best_loop["loops"] >= min_loops:
                        opportunities.append({
                            "market_id": market_id,
                            "market_name": f"{token0}/{token1}",
                            "verified": is_verified,
                            "deposit_token": token0,
                            "borrow_token": token1,
                            "deposit_token_logo": deposit_token_logo,
                            "borrow_token_logo": borrow_token_logo,
                            "deposit_apr": deposit_apr,
                            "borrow_apr": borrow_apr,
                            "apr_spread": spread,
                            "best_loops": best_loop["loops"],
                            "max_leverage": best_loop["max_leverage"],
                            "max_yield": best_loop["max_yield"],
                            "initial_amount": initial_amount,
                            "available_liquidity": liquidity,
                            "loop_results": loop_results
                        })
            
            # Second direction: deposit silo1 token, borrow silo0 token
            if silo0.get("isBorrowable", False):
                token0 = silo1.get("market")  # Use market as token symbol
                token1 = silo0.get("market")  # Use market as token symbol
                # Convert APR strings to floats properly
                deposit_apr = float(silo1.get("deposit_apr", "0%").rstrip("%"))
                borrow_apr = float(silo0.get("borrow_apr", "0%").rstrip("%"))
                deposit_token_logo = silo1.get("logo")
                borrow_token_logo = silo0.get("logo")
                # Get liquidity (max borrowable amount)
                liquidity = silo0.get("liquidity", 0)
                
                if deposit_apr > borrow_apr:
                    spread = deposit_apr - borrow_apr
                    loop_results = self._calculate_loop_yields(
                        deposit_apr/100, borrow_apr/100, initial_amount, max_loops, silo1.get("max_ltv", 0.7)
                    )
                    
                    best_loop = max(loop_results, key=lambda x: x["net_apr"])
                    
                    if best_loop["loops"] >= min_loops:
                        opportunities.append({
                            "market_id": market_id,
                            "market_name": f"{token0}/{token1}",
                            "verified": is_verified,
                            "deposit_token": token0,
                            "borrow_token": token1,
                            "deposit_token_logo": deposit_token_logo,
                            "borrow_token_logo": borrow_token_logo,
                            "deposit_apr": deposit_apr,
                            "borrow_apr": borrow_apr,
                            "apr_spread": spread,
                            "best_loops": best_loop["loops"],
                            "max_leverage": best_loop["max_leverage"],
                            "max_yield": best_loop["max_yield"],
                            "initial_amount": initial_amount,
                            "available_liquidity": liquidity,
                            "loop_results": loop_results
                        })
        
        # Sort opportunities by max yield (descending)
        sorted_opportunities = sorted(
            opportunities, 
            key=lambda x: x["max_yield"], 
            reverse=True
        )
        
        # Return top N opportunities
        return sorted_opportunities[:limit]
    
    def _format_strategy(self, opportunity):
        """Format a looping strategy into a readable output."""
        strategy = {
            "market_id": opportunity["market_id"],
            "market": opportunity["market_name"],
            "verified": opportunity["verified"],
            "deposit_token": opportunity["deposit_token"],
            "borrow_token": opportunity["borrow_token"],
            "strategy_overview": {
                "deposit_token_logo": opportunity["deposit_token_logo"],
                "borrow_token_logo": opportunity["borrow_token_logo"],
                "deposit_apr": f"{opportunity['deposit_apr']:.2f}%",
                "borrow_apr": f"{opportunity['borrow_apr']:.2f}%",
                "spread": f"{opportunity['apr_spread']:.2f}%",
                "best_loops": opportunity["best_loops"],
                "max_leverage": f"{opportunity['max_leverage']:.2f}x",
                "max_yield": f"{opportunity['max_yield']:.2f}% APR",
                "initial_investment": f"${opportunity['initial_amount']:.2f}",
                "available_liquidity": f"{opportunity['available_liquidity']:.2f}"
            },
            "execution_steps": [
                f"Deposit {opportunity['deposit_token']} on Silo",
                f"Borrow {opportunity['borrow_token']}",
                f"Swap {opportunity['borrow_token']} for {opportunity['deposit_token']}",
                "Repeat steps 1-3 for desired leverage (see yield table)"
            ],
            "yield_table": []
        }
        
        # Add yield data for different loop iterations
        for loop in opportunity["loop_results"]:
            strategy["yield_table"].append({
                "loops": loop["loops"],
                "leverage": f"{loop['leverage']:.2f}x",
                "net_apr": f"{loop['net_apr']:.2f}%",
                "total_deposit": f"${loop['total_deposit']:.2f}",
                "total_borrowed": f"${loop['total_borrowed']:.2f}"
            })
        
        # Add risk considerations
        strategy["risk_considerations"] = [
            "Price impact and slippage may reduce actual yields",
            "Market volatility can increase liquidation risk at higher leverage",
            "APRs may change over time based on market conditions",
            f"Consider maintaining a buffer below maximum leverage for safety",
            f"Available liquidity of {opportunity['available_liquidity']:.2f} {opportunity['borrow_token']} limits maximum borrowing"
        ]
        
        return strategy
    
    def _run(self, initial_amount: float = 1000, token: Optional[str] = None, 
             limit: int = 10, min_loops: int = 2, max_loops: int = 50) -> str:
        logger.info(f"Finding looping strategies with initial amount: {initial_amount}, token: {token}, limit: {limit}, min_loops: {min_loops}, max_loops: {max_loops}")
        try:
            opportunities = self._find_looping_opportunities(
                initial_amount=initial_amount,
                token=token,
                limit=int(limit),
                min_loops=int(min_loops),
                max_loops=int(max_loops)
            )
            
            if not opportunities:
                return "No profitable looping strategies found with the current parameters."
            
            result = {
                "message": f"Output is based on initial investment of ${initial_amount:.2f} and maximum {max_loops} loops",
                "strategies_count": len(opportunities),
                "initial_amount": initial_amount,
                "filtered_token": token,
                "strategies": [self._format_strategy(opp) for opp in opportunities],
                "note": "The outputs only consider the base APR, for any additional rewards check https://v2.silo.finance/"
            }
            
            return json.dumps(result, indent=2)
        
        except Exception as e:
            return f"Error finding looping strategies: {str(e)}"


def get_silo_tools(agent, llm) -> list:
    """Get all Silo-related tools"""
    return [
        SiloPositionTool(agent, llm),
        SiloDepositTool(agent, llm),
        SiloBorrowTool(agent, llm),
        SiloRepayTool(agent, llm),
        SiloWithdrawTool(agent, llm),
        SiloClaimRewardsTool(agent, llm),
        SiloPoolsTool(agent, llm),
        SiloLoopingStrategyTool(agent, llm)
    ]