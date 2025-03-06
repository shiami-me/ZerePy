from typing import Dict, List, Optional, Union, Any, TypedDict


class PrivyWallet(TypedDict):
    """Representation of a Privy wallet"""
    id: str
    address: str
    chain_type: str


class PrivyCreateWalletResponse(TypedDict):
    """Response when creating a new Privy wallet"""
    id: str
    address: str
    chain_type: str


class PrivyWalletsResponse(TypedDict):
    """Response when listing Privy wallets"""
    wallets: List[PrivyWallet]


class PrivyTransaction(TypedDict, total=False):
    """Full transaction object for signing"""
    to: str
    value: Union[int, str]
    chain_id: int
    type: Optional[int]
    gas_limit: Optional[str]
    nonce: int
    max_fee_per_gas: Optional[str]
    max_priority_fee_per_gas: Optional[str]


class PrivySimpleTransaction(TypedDict):
    """Simple transaction object for sending"""
    to: str
    value: Union[int, str]
