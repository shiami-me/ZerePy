import logging
from typing import Any, Dict, Optional, List
from web3 import Web3
from web3.exceptions import ContractLogicError
from web3.types import TxReceipt
from web3.middleware import geth_poa_middleware

from src.constants.networks import SONIC_NETWORKS
from src.constants.abi import SILO_ABI, SILO_CONFIG, SILO_INCENTIVE_ABI
from dotenv import load_dotenv

from .base_connection import BaseConnection, Action, ActionParameter

logger = logging.getLogger("connections.silo_connection")

class SiloConnectionError(Exception):
    """Base exception for Silo connection errors"""
    pass

class SiloConnection(BaseConnection):
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.web3 = None
        
        # Get network configuration
        network = config.get("network", "mainnet")
        if network not in SONIC_NETWORKS:
            raise ValueError(f"Invalid network '{network}'. Must be one of: {', '.join(SONIC_NETWORKS.keys())}")
        
        network_config = SONIC_NETWORKS[network]
        self.explorer = network_config["scanner_url"]
        self.rpc_url = network_config["rpc_url"]
        self.silo_config = SILO_CONFIG
        self.silo_abi = SILO_ABI
        self.silo_incentives = SILO_INCENTIVE_ABI
        
        self._initialize_web3()


    def _get_explorer_link(self, tx_hash: str) -> str:
        """Generate block explorer link for transaction"""
        return f"{self.explorer}/tx/{tx_hash}"

    def _initialize_web3(self):
        """Initialize Web3 connection"""
        try:
            if not self.rpc_url:
                raise SiloConnectionError("RPC URL not configured")
            
            self.web3 = Web3(Web3.HTTPProvider(self.rpc_url))
            # Add POA middleware for networks that require it
            self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            if not self.web3.is_connected():
                raise SiloConnectionError("Failed to connect to RPC endpoint")

        except Exception as e:
            raise SiloConnectionError(f"Failed to initialize Web3: {str(e)}")

    def is_llm_provider(self) -> bool:
        return False

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the configuration parameters"""
                
        return config

    def register_actions(self) -> None:
        
        self.actions['deposit'] = Action(
            name='deposit',
            description='Deposit assets into Silo',
            parameters=[
                ActionParameter(name='silo_address', type=str, required=True, description='Address of the Silo contract'),
                ActionParameter(name='amount', type=float, required=True, description='Amount of assets to deposit'),
                ActionParameter(name='collateral_type', type=int, required=True, description='Type of collateral (0 for Collateral, 1 for Protected)'),
                ActionParameter(name='sender', type=str, required=True, description='Address of the sender'),
                ActionParameter(name='decimals', type=int, required=False, description='Decimals of the token')
            ]
        )

        self.actions['borrow'] = Action(
            name='borrow',
            description='Borrow assets from Silo',
            parameters=[
                ActionParameter(name='silo_address', type=str, required=True, description='Address of the Silo contract'),
                ActionParameter(name='amount', type=float, required=True, description='Amount to borrow'),
                ActionParameter(name='sender', type=str, required=True, description='Address of the sender'),
                ActionParameter(name='receiver', type=str, required=False, description='Address to receive the borrowed assets'),
                ActionParameter(name='decimals', type=int, required=False, description='Decimals of the token')
            ]
        )

        self.actions['repay'] = Action(
            name='repay',
            description='Repay borrowed assets',
            parameters=[
                ActionParameter(name='silo_address', type=str, required=True, description='Address of the Silo contract'),
                ActionParameter(name='amount', type=float, required=True, description='Amount to repay'),
                ActionParameter(name='sender', type=str, required=False, description='Address of the sender'),
                ActionParameter(name='decimals', type=int, required=False, description='Decimals of the token')
            ]
        )

        self.actions['withdraw'] = Action(
            name='withdraw',
            description='Withdraw assets from Silo',
            parameters=[
                ActionParameter(name='silo_address', type=str, required=True, description='Address of the Silo contract'),
                ActionParameter(name='amount', type=float, required=True, description='Amount to withdraw'),
                ActionParameter(name='receiver', type=str, required=False, description='Address to receive the borrowed assets (defaults to sender)'),
                ActionParameter(name='collateral_type', type=int, required=True, description='Type of collateral (0 for Collateral, 1 for Protected)'),
                ActionParameter(name='sender', type=str, required=True, description='Address of the sender'),
                ActionParameter(name='decimals', type=int, required=False, description='Decimals of the token')
            ]
        )
        
        self.actions['borrow-same-asset'] = Action(
            name='borrow-same-asset',
            description='Borrow same asset that was used as collateral from Silo',
            parameters=[
                ActionParameter(name='silo_address', type=str, required=True, description='Address of the Silo contract'),
                ActionParameter(name='amount', type=float, required=True, description='Amount to borrow'),
                ActionParameter(name='sender', type=str, required=True, description='Address of the sender'),
                ActionParameter(name='receiver', type=str, required=False, description='Address to receive the borrowed assets (defaults to sender)'),
                ActionParameter(name='decimals', type=int, required=False, description='Decimals of the token')
            ]
        )

    def configure(self) -> bool:
        """Configure the Silo connection"""
        try:
            if not self.web3:
                self._initialize_web3()

            return True
        except Exception as e:
            logger.error(f"Failed to configure Silo connection: {str(e)}")
            return False

    def is_configured(self, verbose: bool = False) -> bool:
        """Check if the connection is properly configured"""
        is_valid = (
            self.web3 is not None and 
            self.web3.is_connected()
        )
        
        if verbose and not is_valid:
            if not self.web3:
                logger.error("Web3 not initialized")
            elif not self.web3.is_connected():
                logger.error("Web3 not connected")
                
        return is_valid
    
    def _get_silo_address(self, config_address: str, token: int) -> str:
        """Get Silo address for a given token"""
        if not self.is_configured():
            raise SiloConnectionError("Silo connection not properly configured")

        try:
            # Get SiloConfig contract
            silo_config_contract = self.web3.eth.contract(
                address=config_address,
                abi=self.silo_config
            )

            # Get Silo address
            silo0, silo1 = silo_config_contract.functions.getSilos().call()
            if token == 0:
                return silo0
            elif token == 1:
                return silo1
        except Exception as e:
            raise SiloConnectionError(f"Failed to get Silo address: {str(e)}")

    def deposit(self, silo_address: str, amount: float,  
                     collateral_type: int = 0, sender: str = None, decimals: int = 18) -> Dict:
        """
        Deposit assets into a Silo
        
        Args:
            silo_address: Address of the Silo contract
            amount: Amount of assets to deposit
            token_address: Address of the token to deposit
            collateral_type: Type of collateral (0 for Collateral, 1 for Protected)
            sender: Address of the sender (optional)
        """
        if amount == 0:
            raise SiloConnectionError("Cannot deposit zero amount")
        try:
            if not self.is_configured():
                raise SiloConnectionError("Silo connection not properly configured")

            # Convert amount to Wei
            amount_wei = int(amount * (10**decimals))
            
            # Get Silo contract
            silo_contract = self.web3.eth.contract(
                address=silo_address,
                abi=self.silo_abi
            )
            
            # Build transaction
            tx = silo_contract.functions.deposit(
                amount_wei,
                sender,
                collateral_type
            ).build_transaction({
                'from': sender,
                'gas': 500000,  # Estimated gas limit
                'nonce': self.web3.eth.get_transaction_count(
                    sender
                )
            })
            tx["amount"] = int(amount_wei)
            return tx
            
        except ContractLogicError as e:
            raise SiloConnectionError(f"Contract error during deposit: {str(e)}")
        except Exception as e:
            raise SiloConnectionError(f"Failed to deposit: {str(e)}")

    def borrow(self, silo_address: str, amount: float, sender: str,
                    receiver: str = None, decimals: int = 18) -> Dict:
        """
        Borrow assets from a Silo
        
        Args:
            silo_address: Address of the Silo contract
            amount: Amount to borrow
            receiver: Address to receive the borrowed assets
            sender: Address of the sender
            
        Raises:
            SiloConnectionError: If borrow is not possible due to:
                - Zero amount (InputZeroShares)
                - Above max LTV limit (AboveMaxLtv) 
                - Not enough liquidity (NotEnoughLiquidity)
                - Account not solvent
                - Borrow not possible for other reasons
        """
        if amount == 0:
            raise SiloConnectionError("InputZeroShares: Cannot borrow zero amount")
        # Get Silo contract
        silo_contract = self.web3.eth.contract(
            address=silo_address,
            abi=self.silo_abi
        )

        if not self.is_configured():
            raise SiloConnectionError("Silo connection not properly configured")

        # Check if account is solvent
        if not silo_contract.functions.isSolvent(sender).call():
            raise SiloConnectionError("Account is not solvent")

        # Convert amount to Wei
        amount_wei = int(amount * (10**decimals))
        # Check max borrow amount
        max_borrow = silo_contract.functions.maxBorrow(sender).call()
        if amount_wei > max_borrow:
            raise SiloConnectionError(f"AboveMaxLtv: Borrow amount {amount} exceeds maximum allowed: {max_borrow}")
        receiver = receiver or sender
        try:
            # Build transaction
            tx = silo_contract.functions.borrow(
                amount_wei,
                receiver,
                sender
            ).build_transaction({
                'from': sender,
                'gas': 500000,  # Estimated gas limit
                'nonce': self.web3.eth.get_transaction_count(sender)
            })
            return tx
            
            # TODO ADD MORE CHECKS LIKE LIQUIDTIY
            
        except ContractLogicError as e:
            raise SiloConnectionError(f"Contract error during borrow: {str(e)}")
        except Exception as e:
            raise SiloConnectionError(f"Failed to borrow: {str(e)}")

    def borrow_same_asset(self, silo_address: str, amount: float, sender: str,
                          receiver: str = None, decimals: int = 18) -> Dict:
        """
        Borrow same asset from a Silo. This method is used when borrowing the same asset
        that was used as collateral.
        
        Args:
            silo_address: Address of the Silo contract
            amount: Amount to borrow
            sender: Address of the sender
            receiver: Address to receive the borrowed assets (defaults to sender)
            
        Raises:
            SiloConnectionError: If borrow is not possible due to:
                - Zero amount (InputZeroShares)
                - Above max LTV limit (AboveMaxLtv)
                - Not enough liquidity (NotEnoughLiquidity)
                - Account not solvent
                - Borrow not possible for other reasons
        """
        if amount == 0:
            raise SiloConnectionError("InputZeroShares: Cannot borrow zero amount")

        # Get Silo contract
        silo_contract = self.web3.eth.contract(
            address=silo_address,
            abi=self.silo_abi
        )

        if not self.is_configured():
            raise SiloConnectionError("Silo connection not properly configured")

        # Check if account is solvent
        if not silo_contract.functions.isSolvent(sender).call():
            raise SiloConnectionError("Account is not solvent")

        # Convert amount to Wei
        amount_wei = int(amount * (10**decimals))

        # Check max borrow amount for same asset
        max_borrow = silo_contract.functions.maxBorrowSameAsset(sender).call()
        if amount_wei > max_borrow:
            raise SiloConnectionError(f"AboveMaxLtv: Borrow amount {amount} exceeds maximum allowed: {self.web3.from_wei(max_borrow, decimals)}")
        receiver = receiver or sender

        try:
            # Build transaction
            tx = silo_contract.functions.borrowSameAsset(
                amount_wei,
                receiver,
                sender
            ).build_transaction({
                'from': sender,
                'gas': 500000,  # Estimated gas limit
                'nonce': self.web3.eth.get_transaction_count(sender)
            })
            # TODO ADD MORE CHECKS LIKE LIQUIDTIY

            return tx
            
        except ContractLogicError as e:
            raise SiloConnectionError(f"Contract error during borrow: {str(e)}")
        except Exception as e:
            raise SiloConnectionError(f"Failed to borrow same asset: {str(e)}")

    def repay(self, silo_address: str, amount: float, sender: str = None, decimals: int = 18) -> Dict:
        """
        Repay borrowed assets
        
        Args:
            silo_address: Address of the Silo contract
            amount: Amount to repay
            sender: Address of the sender (optional)
        """
        if amount == 0:
            raise SiloConnectionError("Cannot repay zero amount")

        # Get Silo contract
        silo_contract = self.web3.eth.contract(
            address=silo_address,
            abi=self.silo_abi
        )

        # Get max repay amount
        max_repay = silo_contract.functions.maxRepay(
            sender
        ).call()

        # Convert amount to Wei
        amount_wei = int(amount * (10**decimals))
        
        # If amount is greater than max_repay, adjust to max_repay
        if amount_wei > max_repay:
            amount_wei = max_repay
        # Preview repay to get shares
        shares = silo_contract.functions.previewRepay(amount_wei).call()
        if shares == 0:
            raise SiloConnectionError("Cannot repay zero shares")
        try:
            if not self.is_configured():
                raise SiloConnectionError("Silo connection not properly configured")

            # Get Silo contract
            silo_contract = self.web3.eth.contract(
                address=silo_address,
                abi=self.silo_abi
            )
            
            # Build transaction
            tx = silo_contract.functions.repay(
                amount_wei,
                sender
            ).build_transaction({
                'from': sender,
                'gas': 500000,  # Estimated gas limit
                'nonce': self.web3.eth.get_transaction_count(
                    sender
                )
            })
            
            return tx
            
        except ContractLogicError as e:
            raise SiloConnectionError(f"Contract error during repay: {str(e)}")
        except Exception as e:
            raise SiloConnectionError(f"Failed to repay: {str(e)}")

    def withdraw(self, silo_address: str, amount: float, receiver: str = None, 
                      collateral_type: int = 0, sender: str = None, decimals: int = 18) -> Dict:
        """
        Withdraw assets from a Silo
        
        Args:
            silo_address: Address of the Silo contract
            amount: Amount to withdraw
            receiver: Address to receive the withdrawn assets
            collateral_type: Type of collateral (0 for Collateral, 1 for Protected)
            sender: Address of the sender (optional)
            
        Raises:
            SiloConnectionError: If withdrawal not possible or amount exceeds max allowed
        """
        try:
            if not self.is_configured():
                raise SiloConnectionError("Silo connection not properly configured")

            if amount == 0:
                raise SiloConnectionError("Cannot withdraw zero amount")
                
            if collateral_type not in [0, 1]:  # 0: Collateral, 1: Protected
                raise SiloConnectionError("Invalid collateral type")

            sender = sender
            receiver = receiver or sender

            # Convert amount to Wei
            amount_wei = int(amount * (10**decimals))
            
            # Get Silo contract
            silo_contract = self.web3.eth.contract(
                address=silo_address,
                abi=self.silo_abi
            )
            
            # Check max withdraw amount for specific collateral type
            max_withdraw = silo_contract.functions.maxWithdraw(
                sender,
                collateral_type
            ).call()
            
            if amount_wei > max_withdraw:
                raise SiloConnectionError(f"Withdrawal amount exceeds maximum allowed for collateral type {collateral_type}: {str(max_withdraw / (10**decimals))}")
            
            # Build transaction
            tx = silo_contract.functions.withdraw(
                amount_wei,
                receiver,
                sender,
                collateral_type
            ).build_transaction({
                'from': sender,
                'gas': 500000,
                'nonce': self.web3.eth.get_transaction_count(sender)
            })
            
            return tx
            
        except ContractLogicError as e:
            raise SiloConnectionError(f"Contract error during withdraw: {str(e)}")
        except Exception as e:
            raise SiloConnectionError(f"Failed to withdraw: {str(e)}")
    
    def claim_rewards(self, sender: str) -> Dict:
        """
        Claim rewards from a Silo

        Args:
            sender: Address of the sender
        """
        try:
            if not self.is_configured():
                raise SiloConnectionError("Silo connection not properly configured")

            # Get Silo contract
            silo_contract = self.web3.eth.contract(
                address="0x2D3d269334485d2D876df7363e1A50b13220a7D8",
                abi=self.silo_incentives
            )

            # Build transaction
            tx = silo_contract.functions.claimRewards(
                sender
            ).build_transaction({
                'from': sender,
                'gas': 500000,
                'nonce': self.web3.eth.get_transaction_count(sender)
            })

            return tx

        except ContractLogicError as e:
            raise SiloConnectionError(f"Contract error during claim rewards: {str(e)}")
        except Exception as e:
            raise SiloConnectionError(f"Failed to claim rewards: {str(e)}")

    def perform_action(self, action_name: str, kwargs) -> Any:
        """Execute a Silo action with validation"""
        if action_name not in self.actions:
            raise KeyError(f"Unknown action: {action_name}")

        load_dotenv()
        
        if not self.is_configured(verbose=True):
            raise SiloConnectionError("Silo is not properly configured")

        action = self.actions[action_name]
        validation_errors = action.validate_params(kwargs)
        if validation_errors:
            raise SiloConnectionError(f"Invalid parameters: {', '.join(validation_errors)}")

        method_name = action_name.replace('-', '_')
        method = getattr(self, method_name)
        return method(**kwargs)