import os
import time
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from functools import wraps
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from technical_analyzer import TechnicalAnalyzer
import traceback

load_dotenv()

def rate_limit(calls_per_minute=60):
    min_interval = 60.0 / float(calls_per_minute)
    last_time_called = {}

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            elapsed = current_time - last_time_called.get(func.__name__, 0)
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            result = func(*args, **kwargs)
            last_time_called[func.__name__] = time.time()
            return result
        return wrapper
    return decorator

class CoinGeckoAPI:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.api_key = os.getenv("COINGECKO_API_KEY")
        self.headers = {
            "x-cg-demo-api-key": self.api_key,
            "accept": "application/json"
        }

    @rate_limit(calls_per_minute=60)
    def get_simple_price(self, ids: List[str], vs_currencies: str = 'usd') -> Dict:
        try:
            endpoint = f"{self.base_url}/simple/price"
            params = {'ids': ','.join(ids), 'vs_currencies': vs_currencies}
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching prices from CoinGecko: {e}")
            return {}

    @rate_limit(calls_per_minute=60)
    def get_coin_market_chart(self, id: str, days: int = 30, vs_currency: str = 'usd') -> Dict:
        try:
            endpoint = f"{self.base_url}/coins/{id}/market_chart"
            params = {'vs_currency': vs_currency, 'days': days, 'interval': 'daily'}
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching historical data from CoinGecko: {e}")
            return {}

@dataclass
class PortfolioState:
    messages: List[dict]
    portfolio: Dict[str, float]
    analysis_results: Optional[Dict[str, Any]] = None
    current_agent: str = "portfolio_agent"

class PortfolioDataService:
    def __init__(self):
        self.cg = CoinGeckoAPI()
        self.cache = {}
        self.cache_duration = 300  

    @rate_limit(calls_per_minute=60)
    def get_crypto_prices(self, symbols: List[str]) -> Dict[str, float]:
        try:
            cache_key = f"prices_{','.join(sorted(symbols))}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if time.time() - cached_data['timestamp'] < self.cache_duration:
                    return cached_data['data']
            ids = [self._convert_symbol_to_id(symbol) for symbol in symbols]
            prices = self.cg.get_simple_price(ids=ids, vs_currencies='usd')
            result = {symbol: prices.get(self._convert_symbol_to_id(symbol), {}).get('usd', 0)
                      for symbol in symbols}
            self.cache[cache_key] = {'data': result, 'timestamp': time.time()}
            return result
        except Exception as e:
            print(f"Error getting crypto prices: {e}")
            return {}

    @rate_limit(calls_per_minute=60)
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        try:
            ticker = yf.Ticker(f"{symbol}-USD")
            data = ticker.history(period=f"{days}d")
            return data['Close']
        except Exception as e:
            print(f"Error getting historical data: {e}")
            return pd.DataFrame()

    def _convert_symbol_to_id(self, symbol: str) -> str:
        symbol_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'SOL': 'solana',
            'AVAX': 'avalanche-2',
            'USDC': 'usd-coin'
        }
        return symbol_map.get(symbol.upper(), symbol.lower())

class PortfolioAnalyzer:
    def __init__(self):
        self.data_service = PortfolioDataService()
        self.technical_analyzer = TechnicalAnalyzer()

    def analyze_portfolio(self, portfolio: Dict[str, float]) -> Dict[str, Any]:
        try:
            symbols = list(portfolio.keys())
            prices = self.data_service.get_crypto_prices(symbols)
            historical_data = pd.DataFrame()
            for symbol in symbols:
                historical_data[symbol] = self.data_service.get_historical_data(symbol)
            total_value = sum(portfolio[symbol] * prices.get(symbol, 0) for symbol in symbols)
            weights = {symbol: portfolio[symbol] * prices.get(symbol, 0) / total_value for symbol in symbols}
            returns = historical_data.pct_change().dropna()
            portfolio_return = sum(weights[col] * returns[col] for col in returns.columns)
            portfolio_risk = portfolio_return.std() * np.sqrt(252)  # Annualized
            technical_metrics = {}
            for symbol in symbols:
                hist_data = self.data_service.get_historical_data(symbol)
                if not hist_data.empty:
                    technical_metrics[symbol] = self.technical_analyzer.calculate_indicators(hist_data)
            result = {
                'total_value': total_value,
                'weights': weights,
                'portfolio_risk': portfolio_risk,
                'sharpe_ratio': (portfolio_return.mean() * 252 - 0.02) / portfolio_risk,
                'current_prices': prices,
                'technical_analysis': technical_metrics
            }
            return convert_to_native_types(result)
        except Exception as e:
            print(f"Error analyzing portfolio: {e}")
            return {}

def convert_to_native_types(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                        np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    return obj

@tool
def analyze_portfolio(portfolio: Dict[str, float]) -> Dict[str, Any]:
    """Analyze a crypto portfolio and return metrics including risk, returns, and current allocation."""
    analyzer = PortfolioAnalyzer()
    return analyzer.analyze_portfolio(portfolio)

@tool
def get_crypto_prices(symbols: List[str]) -> Dict[str, float]:
    """Get current prices for crypto assets."""
    data_service = PortfolioDataService()
    return data_service.get_crypto_prices(symbols)

@tool
def get_historical_data(symbol: str, days: int = 30) -> Dict[str, float]:
    """Get historical price data for a crypto asset."""
    data_service = PortfolioDataService()
    return data_service.get_historical_data(symbol, days).to_dict()


def portfolio_agent(portfolio: Dict[str, float]) -> None:
    """
    Unified function that performs portfolio analysis and then invokes
    the LLM to generate comprehensive rebalancing advice.
    """
    try:
        # Instantiate the LLM using the tuple-based message format.
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            streaming=True
        )

        # Perform portfolio analysis.
        analysis_results = PortfolioAnalyzer().analyze_portfolio(portfolio)

        # Build context for the LLM.
        context = f"""Current portfolio analysis:
                    Total Value: ${analysis_results['total_value']:,.2f}
                    Portfolio Weights: {analysis_results['weights']}
                    Risk Level: {analysis_results['portfolio_risk']:.2%}
                    Sharpe Ratio: {analysis_results['sharpe_ratio']:.2f}
                    Current Prices: {analysis_results['current_prices']}

                    Technical Analysis Summary:
                    {analysis_results.get('technical_analysis', {})}"""

        human_prompt = f"""Based on the portfolio data provided:
                    {context}

                    Provide a single, comprehensive response that includes:
                    1. Current portfolio assessment
                    2. Specific rebalancing recommendations with target percentages
                    3. Risk assessment and mitigation strategies
                    4. Clear action items

                    Be direct and conclusive."""
        
        
        messages = [
            ("system", "You are a crypto portfolio advisor."),
            ("human", human_prompt)
        ]

        # Invoke the model.
        response = model.invoke(messages)

        print("\nPortfolio Analysis Results:")
        print(response.content)
        
    except Exception as e:
        print("Error in portfolio_agent:")
        print(traceback.format_exc())

if __name__ == "__main__":
    test_portfolio = {
        'BTC': 0.5,
        'ETH': 5.0,
        'SOL': 100.0,
        'USDC': 5000.0
    }
    
    print("Generating portfolio advice...")
    portfolio_agent(test_portfolio)
