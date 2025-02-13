from typing import Optional, Tuple
import json
import logging
import requests
from langchain.tools import BaseTool

logger = logging.getLogger("tools.silo_tools")

def get_silo_config_address(token_0: str, token_1: str) -> Tuple[str, bool, str]:
    """Get silo config address for given token pair from API.
    
    Args:
        token_0: First token symbol
        token_1: Second token symbol
        
    Returns:
        Tuple of (config_address, is_token0_silo0) where is_token0_silo0 indicates if token_0 
        corresponds to silo0 in the pair
    """
    url = "https://v2.silo.finance/api/display-markets"
    payload = {
        "isApeMode": False,
        "isCurated": True,
        "protocolKey": "sonic",
        "search": None,
        "sort": None
    }
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    data = response.json()
    for market in data:
        silo0_symbol = market["silo0"]["symbol"]
        silo1_symbol = market["silo1"]["symbol"]
        token0_contract = market["silo0"]["tokenAddress"]
        token1_contract = market["silo1"]["tokenAddress"]
        token0_decimals = market["silo0"]["decimals"]
        token1_decimals = market["silo1"]["decimals"]
        
        # Check both orderings of the pair
        if (silo0_symbol == token_0 and silo1_symbol == token_1):
            return market["configAddress"], True, token0_contract, token0_decimals
        elif (silo0_symbol == token_1 and silo1_symbol == token_0):
            return market["configAddress"], False, token1_contract, token1_decimals
            
    raise ValueError(f"No silo found for token pair {token_0}/{token_1}")

class SiloDepositTool(BaseTool):
    name: str = "silo_deposit"
    description: str = """
    silo_deposit: Deposit tokens into a Silo smart contract. Supports both Collateral (0) and Protected (1) deposits.
    Ex - deposit Collateral 1000 USDC into Sonic/USDC pair. Then token_0: USDC, token_1: Sonic, collateral_type: 0 (collateral), amount: 1000.0
        deposit Protected 100 Sonic into Sonic/USDC pair. Then token_0: Sonic, token_1: USDC, collateral_type: 1 (protected), amount: 100.0
    Args:
        token_0: Symbol of the token to deposit
        token_1: Symbol of the other token in the Silo pair
        amount: Amount of assets to deposit
        collateral_type: Type of collateral (0 for Collateral, 1 for Protected)
        sender: Address of the sender (optional)
    """
    
    def __init__(self, agent, llm):
        super().__init__()
        self._agent = agent
        self._llm = llm
    
    def _run(self, token_0: str, token_1: str, amount: float, 
             collateral_type: int = 0, sender: Optional[str] = None) -> str:
        try:
            silo_config_address, is_token0_silo0, token_address, decimals = get_silo_config_address(token_0, token_1)
            # Get appropriate silo address based on token position
            token_idx = 0 if is_token0_silo0 else 1
            silo_address = self._agent.connection_manager.connections["silo"]._get_silo_address(silo_config_address, token_idx)
            result = self._agent.connection_manager.connections["silo"].deposit(
                silo_address=silo_address,
                amount=amount,
                collateral_type=int(collateral_type),
                sender=sender,
                decimals=decimals
            )
            result["type"] = "deposit"
            result["tokenAddress"] = token_address
            return result
        except Exception as e:
            return f"Error depositing tokens: {str(e)}"

class SiloBorrowTool(BaseTool):
    name: str = "silo_borrow"
    description: str = """
    silo_borrow: Borrow tokens from a Silo smart contract.
    Ex - borrow 1000 USDC from Sonic/USDC pair. Then token_0: USDC, token_1: Sonic, amount: 1000.0
        borrow 100 Sonic from Sonic/USDC pair. Then token_0: Sonic, token_1: USDC, amount: 100.0
    Args:
        token_0: Symbol of the token to borrow
        token_1: Symbol of the other token in the Silo pair
        amount: Amount of tokens to borrow
        sender: Address of the sender
        receiver: Address to receive the borrowed assets (optional, defaults to sender)
    """
    
    def __init__(self, agent, llm):
        super().__init__()
        self._agent = agent
        self._llm = llm
    
    def _run(self, token_0: str, token_1: str, amount: float, sender: str, receiver: Optional[str] = None) -> str:
        try:
            silo_config_address, is_token0_silo0, token_address, decimals = get_silo_config_address(token_0, token_1)
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
            return result
        except Exception as e:
            return f"Error borrowing tokens: {str(e)}"

class SiloRepayTool(BaseTool):
    name: str = "silo_repay"
    description: str = """
    silo_repay: Repay borrowed tokens to a Silo smart contract.
    Ex - repay 1000 USDC into Sonic/USDC pair. Then token_0: USDC, token_1: Sonic, amount: 1000.0
        repay 100 Sonic into Sonic/USDC pair. Then token_0: Sonic, token_1: USDC, amount: 100.0
    Args:
        token_0: Symbol of the token to repay
        token_1: Symbol of the other token in the Silo pair
        amount: Amount of tokens to repay
        sender: Address of the sender (optional)
    """
    
    def __init__(self, agent, llm):
        super().__init__()
        self._agent = agent
        self._llm = llm
    
    def _run(self, token_0: str, token_1: str, amount: float, sender: Optional[str] = None) -> str:
        try:
            silo_config_address, is_token0_silo0, token_address, decimals = get_silo_config_address(token_0, token_1)
            # Get appropriate silo address based on token position
            token_idx = 0 if is_token0_silo0 else 1
            silo_address = self._agent.connection_manager.connections["silo"]._get_silo_address(silo_config_address, token_idx)
            result = self._agent.connection_manager.connections["silo"].repay(
                silo_address=silo_address,
                amount=amount,
                sender=sender,
                decimals=decimals
            )
            result["type"] = "repay"
            result["tokenAddress"] = token_address
            return result
        except Exception as e:
            return f"Error repaying tokens: {str(e)}"

class SiloWithdrawTool(BaseTool):
    name: str = "silo_withdraw"
    description: str = """
    silo_withdraw: Withdraw tokens from a Silo smart contract. Supports both Collateral (0) and Protected (1) withdrawals.
    Ex - withdraw Collateral 1000 USDC from Sonic/USDC pair. Then token_0: USDC, token_1: Sonic, collateral_type: 0 (collateral), amount: 1000.0
        withdraw Protected 100 Sonic from Sonic/USDC pair. Then token_0: Sonic, token_1: USDC, collateral_type: 1 (protected), amount: 100.0
    Args:
        token_0: Symbol of the token to withdraw
        token_1: Symbol of the other token in the Silo pair
        amount: Amount of tokens to withdraw
        receiver: Address to receive the withdrawn assets (optional)
        collateral_type: Type of collateral (0 for Collateral, 1 for Protected)
        sender: Address of the sender (optional)
    """
    
    def __init__(self, agent, llm):
        super().__init__()
        self._agent = agent
        self._llm = llm
    
    def _run(self, token_0: str, token_1: str, amount: float, receiver: Optional[str] = None, 
             collateral_type: int = 0, sender: Optional[str] = None) -> str:
        try:
            silo_config_address, is_token0_silo0, token_address, decimals = get_silo_config_address(token_0, token_1)
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
            return result
        except Exception as e:
            return f"Error withdrawing tokens: {str(e)}"

def get_silo_tools(agent, llm) -> list:
    """Get all Silo-related tools"""
    return [
        SiloDepositTool(agent, llm),
        SiloBorrowTool(agent, llm),
        SiloRepayTool(agent, llm),
        SiloWithdrawTool(agent, llm)
    ]