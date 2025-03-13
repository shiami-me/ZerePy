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
            },
            "silo1": {
                "market": item["silo1"]["symbol"],
                "deposit_apr": f"{(float(item['silo1']['collateralBaseApr']) / 10**18) * 100:.2f}%",
                "borrow_apr": f"{(float(item['silo1']['debtBaseApr']) / 10**18) * 100:.2f}%",
                "isBorrowable": not item["silo1"]["isNonBorrowable"],
                "token0": item["silo1"]["symbol"],
                "token1": item["silo0"]["symbol"],
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
            silo_config_address, is_token0_silo0, _, decimals0 = get_silo_config_address(token_0, token_1, int(id))
            _, _, _, decimals1 = get_silo_config_address(token_1, token_0, int(id))
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
            silo_config_address, is_token0_silo0, token_address, decimals = get_silo_config_address(token_0, token_1, int(id))
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
            silo_config_address, is_token0_silo0, token_address, decimals = get_silo_config_address(token_0, token_1, int(id))
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
            silo_config_address, is_token0_silo0, token_address, decimals = get_silo_config_address(token_0, token_1, int(id))
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
    silo_withdraw: Withdraw tokens from a Silo smart contract. Supports both Collateral (0) and Protected (1) withdrawals.
    Ex - withdraw Collateral 1000 USDC from Sonic/USDC pair. Then token_0: USDC, token_1: Sonic, collateral_type: 0 (collateral), amount: 1000.0
        withdraw Protected 100 Sonic from Sonic/USDC pair. Then token_0: Sonic, token_1: USDC, collateral_type: 1 (protected), amount: 100.0
        withdraw Collateral 1000 USDC from market with ID 1. Then id: 1, token_0: USDC, token_1: Sonic, collateral_type: 0, amount: 1000.0
    Args:
        token_0: Symbol of the token to withdraw
        token_1: Symbol of the other token in the Silo pair
        amount: Amount of tokens to withdraw
        receiver: Address to receive the withdrawn assets (optional)
        collateral_type: Type of collateral (0 for Collateral, 1 for Protected)
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
            silo_config_address, is_token0_silo0, token_address, decimals = get_silo_config_address(token_0, token_1, int(id))
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

class SiloPoolsTool(BaseTool):
    name: str = "silo_pools"
    description: str = """
    silo_pools: Get available Silo lending/borrowing pools, optionally filtered by tokens.
    
    Ex - get all Silo pools. Then tokens: "" (empty string or omit)
        get pools containing Sonic. Then tokens: "S" 
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

def get_silo_tools(agent, llm) -> list:
    """Get all Silo-related tools"""
    return [
        SiloPositionTool(agent, llm),
        SiloDepositTool(agent, llm),
        SiloBorrowTool(agent, llm),
        SiloRepayTool(agent, llm),
        SiloWithdrawTool(agent, llm),
        SiloPoolsTool(agent, llm)
    ]