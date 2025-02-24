import logging
import os
import requests
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv, set_key
from web3 import Web3
from web3.middleware import geth_poa_middleware
from src.constants.abi import ERC20_ABI
from src.connections.base_connection import BaseConnection, Action, ActionParameter
from src.constants.networks import SONIC_NETWORKS

logger = logging.getLogger("connections.sonic_connection")


class SonicConnectionError(Exception):
    """Base exception for Sonic connection errors"""
    pass

class SonicConnection(BaseConnection):
    
    def __init__(self, config: Dict[str, Any]):
        logger.info("Initializing Sonic connection...")
        self._web3 = None
        
        # Get network configuration
        network = config.get("network", "mainnet")
        if network not in SONIC_NETWORKS:
            raise ValueError(f"Invalid network '{network}'. Must be one of: {', '.join(SONIC_NETWORKS.keys())}")
            
        network_config = SONIC_NETWORKS[network]
        self.explorer = network_config["scanner_url"]
        self.rpc_url = network_config["rpc_url"]
        
        super().__init__(config)
        self._initialize_web3()
        self.ERC20_ABI = ERC20_ABI
        self.NATIVE_TOKEN = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"
        self.aggregator_api = "https://aggregator-api.kyberswap.com/sonic/api/v1"

    def _get_explorer_link(self, tx_hash: str) -> str:
        """Generate block explorer link for transaction"""
        return f"{self.explorer}/tx/{tx_hash}"

    def _initialize_web3(self):
        """Initialize Web3 connection"""
        if not self._web3:
            self._web3 = Web3(Web3.HTTPProvider(self.rpc_url))
            self._web3.middleware_onion.inject(geth_poa_middleware, layer=0)
            if not self._web3.is_connected():
                raise SonicConnectionError("Failed to connect to Sonic network")
            
            try:
                chain_id = self._web3.eth.chain_id
                logger.info(f"Connected to network with chain ID: {chain_id}")
            except Exception as e:
                logger.warning(f"Could not get chain ID: {e}")

    @property
    def is_llm_provider(self) -> bool:
        return False

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Sonic configuration from JSON"""
        required = ["network"]
        missing = [field for field in required if field not in config]
        if missing:
            raise ValueError(f"Missing config fields: {', '.join(missing)}")
        
        if config["network"] not in SONIC_NETWORKS:
            raise ValueError(f"Invalid network '{config['network']}'. Must be one of: {', '.join(SONIC_NETWORKS.keys())}")
            
        return config

    def get_token_by_ticker(self, ticker: str) -> Optional[str]:
        """Get token address by ticker symbol"""
        try:
            if ticker.lower() in ["s", "S"]:
                return "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"
            
            if ticker.lower() in ["usdc", "usdce", "usd"]:
                return "0x29219dd400f2bf60e5a23d13be72b486d4038894"
                
            response = requests.get(
                f"https://api.dexscreener.com/latest/dex/search?q={ticker}"
            )
            response.raise_for_status()

            data = response.json()
            if not data.get('pairs'):
                return None

            sonic_pairs = [
                pair for pair in data["pairs"] if pair.get("chainId") == "sonic"
            ]
            sonic_pairs.sort(key=lambda x: x.get("fdv", 0), reverse=True)

            sonic_pairs = [
                pair
                for pair in sonic_pairs
                if pair.get("baseToken", {}).get("symbol", "").lower() == ticker.lower()
            ]

            if sonic_pairs:
                return sonic_pairs[0].get("baseToken", {}).get("address")
            return None

        except Exception as error:
            logger.error(f"Error fetching token address: {str(error)}")
            return None

    def register_actions(self) -> None:
        self.actions = {
            "get-token-by-ticker": Action(
                name="get-token-by-ticker",
                parameters=[
                    ActionParameter("ticker", True, str, "Token ticker symbol to look up")
                ],
                description="Get token address by ticker symbol"
            ),
            "get-balance": Action(
                name="get-balance",
                parameters=[
                    ActionParameter("address", True, str, "Address to check balance for"),
                    ActionParameter("token_address", False, str, "Optional token address")
                ],
                description="Get $S or token balance"
            ),
            "transfer": Action(
                name="transfer",
                parameters=[
                    ActionParameter("to_address", True, str, "Recipient address"),
                    ActionParameter("amount", True, float, "Amount to transfer"),
                    ActionParameter("token_address", False, str, "Optional token address")
                ],
                description="Send $S or tokens"
            )
        }

    def configure(self) -> bool:
        logger.info("\n🔷 SONIC CHAIN SETUP")
        if self.is_configured():
            logger.info("Sonic connection is already configured")
            response = input("Do you want to reconfigure? (y/n): ")
            if response.lower() != 'y':
                return True

        try:
            if not self._web3.is_connected():
                raise SonicConnectionError("Failed to connect to Sonic network")

            logger.info(f"\n✅ Successfully connected")
            return True

        except Exception as e:
            logger.error(f"Configuration failed: {e}")
            return False

    def is_configured(self, verbose: bool = False) -> bool:
        try:
            if not self._web3.is_connected():
                if verbose:
                    logger.error("Not connected to Sonic network")
                return False
            return True

        except Exception as e:
            if verbose:
                logger.error(f"Configuration check failed: {e}")
            return False

    def get_balance(self, address: str, token_address: Optional[str] = None) -> float:
        """Get balance for an address or the configured wallet"""
        try:
            if token_address:
                contract = self._web3.eth.contract(
                    address=Web3.to_checksum_address(token_address),
                    abi=self.ERC20_ABI
                )
                balance = contract.functions.balanceOf(address).call()
                decimals = contract.functions.decimals().call()
                return balance / (10 ** decimals)
            else:
                balance = self._web3.eth.get_balance(address)
                return self._web3.from_wei(balance, 'ether')

        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            raise

    def transfer(self, from_address: str, to_address: str, amount: float, token_address: Optional[str] = None) -> Dict:
        """Transfer $S or tokens to an address"""
        try:
            chain_id = self._web3.eth.chain_id
            
            if token_address:
                contract = self._web3.eth.contract(
                    address=Web3.to_checksum_address(token_address),
                    abi=self.ERC20_ABI
                )
                decimals = contract.functions.decimals().call()
                amount_raw = int(amount * (10 ** decimals))
                
                tx = {
                    'to': token_address,
                    'data': contract.encodeABI(fn_name="transfer", args=[Web3.to_checksum_address(to_address), amount_raw]),
                    'from': from_address,
                    'gasPrice': self._web3.eth.gas_price,
                    'chainId': chain_id
                }
            else:
                tx = {
                    'to': Web3.to_checksum_address(to_address),
                    'value': self._web3.to_wei(amount, 'ether'),
                    'gas': 21000,
                    'gasPrice': self._web3.eth.gas_price,
                    'chainId': chain_id
                }

            return tx

        except Exception as e:
            logger.error(f"Transfer failed: {e}")
            raise
    
    def wrap_sonic(self, sender: str, amount: float) -> Dict:
        """S to wS"""
        try:
            chain_id = self._web3.eth.chain_id
            contract = self._web3.eth.contract(
                address=Web3.to_checksum_address("0x039e2fB66102314Ce7b64Ce5Ce3E5183bc94aD38"),
                abi=self.ERC20_ABI
            )
            amount_raw = int(amount * (10 ** 18))

            tx = {
                'to': "0x039e2fB66102314Ce7b64Ce5Ce3E5183bc94aD38",
                'from': sender,
                'gasPrice': self._web3.eth.gas_price,
                'chainId': chain_id,
                'value': amount_raw
            }
            
            tx["gas"] = self._web3.eth.estimate_gas(tx)

            return tx
        except Exception as e:
            logger.error(f"Failed to wrap Sonic: {e}")

    def _get_swap_route(self, token_in: str, token_out: str, amount_in: float) -> Dict:
        """Get the best swap route from Kyberswap API"""
        try:
            # Handle native token address
            
            # Convert amount to raw value
            if token_in.lower() == self.NATIVE_TOKEN.lower():
                amount_raw = self._web3.to_wei(amount_in, 'ether')
            else:
                token_contract = self._web3.eth.contract(
                    address=Web3.to_checksum_address(token_in),
                    abi=self.ERC20_ABI
                )
                decimals = token_contract.functions.decimals().call()
                amount_raw = int(amount_in * (10 ** decimals))
            
            # Set up API request
            url = f"{self.aggregator_api}/routes"
            headers = {"x-client-id": "ZerePyBot"}
            params = {
                "tokenIn": token_in,
                "tokenOut": token_out,
                "amountIn": str(amount_raw),
                "gasInclude": "true"
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data.get("code") != 0:
                raise SonicConnectionError(f"API error: {data.get('message')}")
                
            return data["data"]
                
        except Exception as e:
            logger.error(f"Failed to get swap route: {e}")
            raise

    def _get_encoded_swap_data(self, sender: str, route_summary: Dict, slippage: float = 0.5) -> str:
        """Get encoded swap data from Kyberswap API"""
        try:
            url = f"{self.aggregator_api}/route/build"
            headers = {"x-client-id": "zerepy"}
            
            payload = {
                "routeSummary": route_summary,
                "sender": sender,
                "recipient": sender,
                "slippageTolerance": int(slippage * 100),  # Convert to bps
                "deadline": int(time.time() + 1200),  # 20 minutes
                "source": "ZerePyBot"
            }
            
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            if data.get("code") != 0:
                raise SonicConnectionError(f"API error: {data.get('message')}")
                
            return data["data"]["data"]
                
        except Exception as e:
            logger.error(f"Failed to encode swap data: {e}")
            raise
    
    def _handle_token_approval(self, sender: str, token_address: str, spender_address: str, amount: int) -> bool:
        """Handle token approval for spender"""
        try:
            token_contract = self._web3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=self.ERC20_ABI
            )
            
            # Check current allowance
            current_allowance = token_contract.functions.allowance(
                sender,
                spender_address
            ).call()
            
            if current_allowance < amount:
                raise ValueError("Insufficient allowance")
            return True
                
        except Exception as e:
            logger.error(f"Approval failed: {e}")
            raise

    def swap(self, sender: str, token_in: str, token_out: str, amount: float, slippage: float = 0.5) -> Dict:
        """Execute a token swap using the KyberSwap router"""
        try:
            # Check token balance before proceeding
            logger.info(slippage)
            current_balance = self.get_balance(
                address=sender,
                token_address=None if token_in.lower() == self.NATIVE_TOKEN.lower() else token_in
            )
            
            if current_balance < amount:
                raise ValueError(f"Insufficient balance. Required: {amount}, Available: {current_balance}")
                
            # Get optimal swap route
            route_data = self._get_swap_route(token_in, token_out, amount)
            
            # Get encoded swap data
            encoded_data = self._get_encoded_swap_data(sender, route_data["routeSummary"], slippage)
            
            # Get router address from route data
            router_address = route_data["routerAddress"]
            
            # Prepare transaction
            tx = {
                'from': sender,
                'to': Web3.to_checksum_address(router_address),
                'data': encoded_data,
                'nonce': self._web3.eth.get_transaction_count(sender),
                'gasPrice': self._web3.eth.gas_price,
                'chainId': self._web3.eth.chain_id,
                'value': self._web3.to_wei(amount, 'ether') if token_in.lower() == self.NATIVE_TOKEN.lower() else 0
            }
            
            # Estimate gas
            try:
                tx['gas'] = self._web3.eth.estimate_gas(tx)
            except Exception as e:
                if ("Return amount is not enough" in str(e)):
                    raise SonicConnectionError("Insufficient output amount. Please try with higher slippage")
                logger.warning(f"Gas estimation failed: {e}, using default gas limit")
                tx['gas'] = 500000  # Default gas limit
            
            return tx
                
        except Exception as e:
            logger.error(f"Swap failed: {e}")
            raise
    def estimate_gas(self, tx: Dict) -> Dict:
        """Estimate gas for a transaction"""
        try:
            estimated_gas = self._web3.eth.estimate_gas(tx)
            tx['gas'] = int(estimated_gas * 1.1)
            tx["gasPrice"] = self._web3.eth.gas_price
            return tx
        except Exception as e:
            logger.error(f"Gas estimation failed: {e}")
            raise SonicConnectionError(f"Failed to estimate gas: {e}")
    def perform_action(self, action_name: str, kwargs) -> Any:
        """Execute a Sonic action with validation"""
        if action_name not in self.actions:
            raise KeyError(f"Unknown action: {action_name}")

        load_dotenv()
        
        if not self.is_configured(verbose=True):
            raise SonicConnectionError("Sonic is not properly configured")

        action = self.actions[action_name]
        errors = action.validate_params(kwargs)
        if errors:
            raise ValueError(f"Invalid parameters: {', '.join(errors)}")

        method_name = action_name.replace('-', '_')
        method = getattr(self, method_name)
        return method(**kwargs)