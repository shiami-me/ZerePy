from typing import Dict, Any
from langchain.tools import BaseTool
import requests

class GetTokenPriceTool(BaseTool):
    name: str = "get_token_price"
    description: str = """
    get_token_price: Get token price
    Input should be a token symbol (e.g. "S", "ANON")
    Returns the token's current price.
    Ex - what's the price of S?. Input - { token: "S" }
    Give a price report of EGGS token. Input - { token: "EGGS" }
    """

    def _run(self, token: str) -> Dict[str, Any]:
        try:
            if token.lower() in ["s", "sonic"]:
                response = requests.get("https://api.dexscreener.com/token-pairs/v1/sonic/0x039e2fB66102314Ce7b64Ce5Ce3E5183bc94aD38")
                response.raise_for_status()
                data = response.json()
                for pair in data:
                    if pair["baseToken"]["address"] == "0x039e2fB66102314Ce7b64Ce5Ce3E5183bc94aD38":
                        return {"price": f"${pair["priceUsd"]}"}
            else: 
                response = requests.get(f"https://api.dexscreener.com/latest/dex/search?q={token}/USDC")
                response.raise_for_status()
                data = response.json()
                for pair in data["pairs"]:
                    if pair["chainId"] == "sonic" and (pair["baseToken"]["symbol"].lower() == token.lower() or pair["baseToken"]["name"].lower() == token.lower()):
                        return {"price": f"${pair["priceUsd"]}"}
            return {"error": f"Token {token} not found. Please not that we currently support only Sonic chain."}
        except Exception as e:
            return {"error": f"Failed to get token price: {str(e)}"}
