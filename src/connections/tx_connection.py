import logging
import requests
import os
import traceback
from typing import Dict, Any, List, Optional
from web3 import Web3
from web3.middleware import geth_poa_middleware
from .base_connection import BaseConnection, Action, ActionParameter
from src.constants.networks import SONIC_NETWORKS

logger = logging.getLogger("connections.tx_connection")

class TxConnectionError(Exception):
    """Base exception for transaction connection errors"""
    pass

class TxConnection(BaseConnection):
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base_url = config.get("api_base_url", "https://api.sonicscan.org/api")
        self.api_key = config.get("api_key", os.environ.get("SONICSCAN_API_KEY", ""))
        self.web3 = None
        
        # Get network configuration using the silo_connection approach
        network = config.get("network", "mainnet")
        if network not in SONIC_NETWORKS:
            logger.warning(f"Invalid network '{network}'. Using mainnet as default.")
            network = "mainnet"
            
        network_config = SONIC_NETWORKS[network]
        self.explorer = network_config["scanner_url"]
        self.rpc_url = network_config["rpc_url"]
        self.network = network
        self.chain_id = network_config.get("chain_id")
        
        self._initialize()

    def _initialize(self):
        """Initialize Transaction connection"""
        if not self.api_key:
            logger.warning("No API key provided for SonicScan API. Some requests may be rate-limited.")
        logger.info(f"Initialized Transaction connection with API URL: {self.api_base_url}")
        
        # Initialize Web3 connection
        try:
            self._initialize_web3()
        except Exception as e:
            logger.warning(f"Failed to initialize Web3 connection: {str(e)}. Token balance features will be unavailable.")

    def _initialize_web3(self):
        """Initialize Web3 connection"""
        try:
            if not self.rpc_url:
                raise TxConnectionError("RPC URL not configured")
            
            self.web3 = Web3(Web3.HTTPProvider(self.rpc_url))
            # Add POA middleware for networks that require it
            self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            if not self.web3.is_connected():
                raise TxConnectionError("Failed to connect to RPC endpoint")
            
            chain_id = self.web3.eth.chain_id
            logger.info(f"Web3 connection initialized to chain ID: {chain_id} ({self.network})")
            
            # Verify chain ID if specified in the network config
            if self.chain_id and chain_id != self.chain_id:
                logger.warning(f"Connected chain ID {chain_id} doesn't match expected chain ID {self.chain_id} for network {self.network}")
                
        except Exception as e:
            self.web3 = None
            raise TxConnectionError(f"Failed to initialize Web3: {str(e)}")

    def _get_explorer_link(self, tx_hash: str) -> str:
        """Generate block explorer link for transaction"""
        return f"{self.explorer}/tx/{tx_hash}"

    def is_llm_provider(self) -> bool:
        return False

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the configuration parameters"""
        # Add API URL if not provided
        if "api_base_url" not in config:
            config["api_base_url"] = "https://api.sonicscan.org/api"
            
        # Check for API key in config or environment
        if "api_key" not in config:
            config["api_key"] = os.environ.get("SONICSCAN_API_KEY", "")
            
        # Set default network if not provided
        if "network" not in config:
            config["network"] = "mainnet"
            
        return config

    def register_actions(self) -> None:
        # Get transaction list action
        self.actions['get_tx_list'] = Action(
            name='get_tx_list',
            description='Get a list of normal transactions for an address',
            parameters=[
                ActionParameter(name='address', type=str, required=True, description='Address to get transactions for'),
                ActionParameter(name='startblock', type=int, required=False, description='Starting block number'),
                ActionParameter(name='endblock', type=int, required=False, description='Ending block number'),
                ActionParameter(name='page', type=int, required=False, description='Page number'),
                ActionParameter(name='offset', type=int, required=False, description='Max records to return'),
                ActionParameter(name='sort', type=str, required=False, description='Sort order (asc/desc)'),
            ]
        )
        
        # Get token transfers action
        self.actions['get_token_tx_list'] = Action(
            name='get_token_tx_list',
            description='Get a list of token transfers for an address',
            parameters=[
                ActionParameter(name='address', type=str, required=True, description='Address to get transfers for'),
                ActionParameter(name='contractaddress', type=str, required=False, description='Filter by token contract address'),
                ActionParameter(name='offset', type=int, required=False, description='Max records to return'),
                ActionParameter(name='sort', type=str, required=False, description='Sort order (asc/desc)'),
            ]
        )
        
        # Add new action for token balances
        self.actions['get_token_balances'] = Action(
            name='get_token_balances',
            description='Get all token balances for an address',
            parameters=[
                ActionParameter(name='address', type=str, required=True, description='Address to get token balances for'),
                ActionParameter(name='include_zero_balances', type=bool, required=False, description='Include tokens with zero balance'),
            ]
        )

    def configure(self) -> bool:
        """Configure the Transaction connection"""
        try:
            self._initialize()
            return True
        except Exception as e:
            logger.error(f"Failed to configure Transaction connection: {str(e)}")
            return False

    def is_configured(self, verbose: bool = False) -> bool:
        """Check if the connection is properly configured"""
        if not self.api_base_url:
            if verbose:
                logger.error("API base URL is not configured")
            return False
        return True

    def get_tx_list(self, address: str, startblock: int = 0, endblock: int = 99999999, 
                    page: int = 1, offset: int = 10, sort: str = "asc") -> Dict:
        """
        Get a list of normal transactions by address
        """
        try:
            params = {
                "module": "account",
                "action": "txlist",
                "address": address,
                "startblock": startblock,
                "endblock": endblock,
                "page": page,
                "offset": offset,
                "sort": sort,
            }
            
            if self.api_key:
                params["apikey"] = self.api_key
                
            response = requests.get(self.api_base_url, params=params)
            response.raise_for_status()
            
            result = response.json()
            if result.get("status") == "0" and result.get("message") == "No transactions found":
                return {
                    "status": "success", 
                    "message": "No transactions found", 
                    "result": []
                }
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise TxConnectionError(f"API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to get transaction list: {str(e)}")
            raise TxConnectionError(f"Failed to get transaction list: {str(e)}")

    def get_token_tx_list(self, address: str, contractaddress: str = None, 
                         offset: int = 100, sort: str = "asc") -> Dict:
        """
        Get a list of token transfers for an address
        """
        try:
            params = {
                "module": "account",
                "action": "tokentx",
                "address": address,
                "startblock": 0,
                "endblock": 999999999,
                "page": 1,
                "offset": offset,
                "sort": sort,
            }
            
            if contractaddress:
                params["contractaddress"] = contractaddress
                
            if self.api_key:
                params["apikey"] = self.api_key
                
            response = requests.get(self.api_base_url, params=params)
            response.raise_for_status()
            
            result = response.json()
            if result.get("status") == "0" and result.get("message") == "No transactions found":
                return {
                    "status": "success", 
                    "message": "No token transfers found", 
                    "result": []
                }
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise TxConnectionError(f"API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to get token transfers: {str(e)}")
            raise TxConnectionError(f"Failed to get token transfers: {str(e)}")

    def get_token_balances(self, address: str, include_zero_balances: bool = False) -> Dict:
        """
        Get all token balances for an address using web3
        
        This method follows this pattern:
        1. Get all transfer logs to/from the address
        2. Extract unique token addresses from those logs
        3. Query each token contract for balance, symbol and decimals
        4. Return formatted balances
        """
        try:
            if self.web3 is None:
                self._initialize_web3()
                if self.web3 is None:
                    return {"error": "Web3 connection not available", "status": "error"}
            
            if not self.web3.is_connected():
                return {"error": "Failed to connect to blockchain", "status": "error"}
            
            w3 = self.web3  # Use the initialized web3 instance
            
            # Ensure address is checksum format
            try:
                address = w3.to_checksum_address(address)
            except Exception as e:
                return {"error": f"Invalid address format: {str(e)}", "status": "error"}
            # Get native token balance (ETH, BNB, etc.)
            native_balance = w3.eth.get_balance(address)
            native_balance_adjusted = w3.from_wei(native_balance, 'ether')
            
            # Define minimal ERC20 ABI for balanceOf, symbol and decimals functions
            erc20_abi = [
                {
                    "constant": True,
                    "inputs": [{"name": "_owner", "type": "address"}],
                    "name": "balanceOf",
                    "outputs": [{"name": "balance", "type": "uint256"}],
                    "type": "function"
                },
                {
                    "constant": True,
                    "inputs": [],
                    "name": "symbol",
                    "outputs": [{"name": "", "type": "string"}],
                    "type": "function"
                },
                {
                    "constant": True,
                    "inputs": [],
                    "name": "decimals",
                    "outputs": [{"name": "", "type": "uint8"}],
                    "type": "function"
                },
                {
                    "constant": True,
                    "inputs": [],
                    "name": "name",
                    "outputs": [{"name": "", "type": "string"}],
                    "type": "function"
                }
            ]
            
            # Get token addresses from transactions
            token_tx_response = self.get_token_tx_list(address=address, offset=100)
            token_addresses = set()
            
            if token_tx_response.get("status") == "1" and token_tx_response.get("result"):
                for tx in token_tx_response.get("result", []):
                    if tx.get("contractAddress"):
                        try:
                            token_addr = w3.to_checksum_address(tx.get("contractAddress"))
                            token_addresses.add(token_addr)
                        except Exception:
                            continue
            
            logger.info(f"Found {len(token_addresses)} token addresses from transaction history")
            
            # Query each token contract for balance information
            token_balances = []
            
            # Add native token info
            chain_id = w3.eth.chain_id
            native_symbol = "S"
            native_name = "Sonic"
            
            native_token_info = {
                "token": "Native",
                "symbol": native_symbol,
                "name": native_name,
                "balance": str(native_balance_adjusted),
                "raw_balance": str(native_balance),
                "decimals": 18,
                "chain_id": chain_id
            }
            token_balances.append(native_token_info)
            
            # Process ERC-20 tokens
            for token_address in token_addresses:
                try:
                    token_contract = w3.eth.contract(address=token_address, abi=erc20_abi)
                    
                    # Get balance, symbol and decimals in parallel
                    balance = token_contract.functions.balanceOf(address).call()
                    
                    # Skip tokens with zero balance unless explicitly included
                    if balance == 0 and not include_zero_balances:
                        continue
                        
                    # Try to get symbol and decimals
                    try:
                        symbol = token_contract.functions.symbol().call()
                        if isinstance(symbol, bytes):
                            symbol = symbol.decode('utf-8')
                    except (Exception, ValueError) as e:
                        symbol = "UNKNOWN"
                    
                    try:
                        decimals = token_contract.functions.decimals().call()
                    except (Exception, ValueError) as e:
                        decimals = 18  # Default to 18 if we can't get decimals
                    
                    try:
                        name = token_contract.functions.name().call()
                        if isinstance(name, bytes):
                            name = name.decode('utf-8')
                    except (Exception, ValueError) as e:
                        name = "Unknown Token"
                    
                    # Calculate human-readable balance
                    adjusted_balance = balance / (10 ** decimals)
                    
                    token_balances.append({
                        "token": token_address,
                        "symbol": symbol,
                        "name": name,
                        "balance": str(adjusted_balance),
                        "raw_balance": str(balance),
                        "decimals": decimals,
                        "chain_id": chain_id,
                        "network": self.network,
                        "explorer_url": f"{self.explorer}/token/{token_address}"
                    })
                    
                except Exception as e:
                    logger.error(f"Error fetching data for token {token_address}: {str(e)}")
                    continue
            
            # Sort by balance value (descending)
            token_balances.sort(key=lambda x: float(x["balance"]) if x["balance"] != "0" else 0, reverse=True)
            logger.info(token_balances)
            return {
                "status": "success",
                "message": f"Found {len(token_balances)} tokens with {'non-zero ' if not include_zero_balances else ''}balance",
                "result": token_balances
            }
                
        except Exception as e:
            logger.error(f"Failed to get token balances: {str(e)}")
            logger.error(traceback.format_exc())
            raise TxConnectionError(f"Failed to get token balances: {str(e)}")

    def perform_action(self, action_name: str, kwargs) -> Any:
        """Execute a transaction action with validation"""
        if action_name not in self.actions:
            raise KeyError(f"Unknown action: {action_name}")
            
        logger.info(f"Performing action: {action_name} with params: {kwargs}")
        
        if not self.is_configured(verbose=True):
            raise TxConnectionError("Transaction service is not properly configured")

        action = self.actions[action_name]
        validation_errors = action.validate_params(kwargs)
        if validation_errors:
            raise TxConnectionError(f"Invalid parameters: {', '.join(validation_errors)}")

        method = getattr(self, action_name)
        return method(**kwargs)
