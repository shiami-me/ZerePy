import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Dict, Optional
import logging
from requests.exceptions import RequestException
import backoff
import os

class TechnicalDataCollector:
    def __init__(self, config):
        self.config = config
        self.base_url = "https://api.coingecko.com/api/v3"
        self.auth_method = config['technical']['coingecko'].get('auth_method', 'header')
        self.api_key = config['technical']['coingecko'].get('api_key')
        
        # Initialize headers based on auth method
        self.headers = {"accept": "application/json"}
        if self.auth_method == "header":
            self.headers["x-cg-demo-api-key"] = self.api_key
        
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self.cache_duration = timedelta(hours=1)

    def get_coin_id(self, symbol: str) -> Optional[str]:
        """Get CoinGecko coin ID from symbol"""
        # Hard-coded mappings for major cryptocurrencies
        MAJOR_COINS = {
            'btc': 'bitcoin',
            'eth': 'ethereum',
            'usdt': 'tether',
            'bnb': 'binancecoin',
            'xrp': 'ripple',
            'sol': 'solana',
            'ada': 'cardano',
            'doge': 'dogecoin',
            'dot': 'polkadot',
            'usdc': 'usd-coin',
            'link': 'chainlink',
            'matic': 'matic-network',
            'uni': 'uniswap',
            'ltc': 'litecoin'
        }
        
        # Check if it's a major coin first
        symbol = symbol.lower()
        if symbol in MAJOR_COINS:
            return MAJOR_COINS[symbol]
            
        cache_key = f"coin_list_{symbol}"
        
        if cache_key in self.cache:
            cached_data, cache_time = self.cache[cache_key]
            if datetime.now() - cache_time < self.cache_duration:
                return cached_data
                
        try:
            url = f"{self.base_url}/coins/list"
            params = {}
            if self.auth_method == "query":
                params["x_cg_demo_api_key"] = self.api_key

            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            coin_list = response.json()
            
            # Try to find an exact match first
            for coin in coin_list:
                if coin['symbol'].lower() == symbol.lower() and coin['id'].lower() == symbol.lower():
                    self.cache[cache_key] = (coin['id'], datetime.now())
                    return coin['id']
                    
            # If no exact match, take the first match
            for coin in coin_list:
                if coin['symbol'].lower() == symbol.lower():
                    self.cache[cache_key] = (coin['id'], datetime.now())
                    return coin['id']
                    
            self.logger.warning(f"No coin found for symbol: {symbol}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting coin ID for {symbol}: {str(e)}")
            raise

    @backoff.on_exception(
        backoff.expo,
        (RequestException, Exception),
        max_tries=3,
        max_time=30
    )
    def fetch_data(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch historical data from CoinGecko"""
        try:
            coin_id = self.get_coin_id(symbol)
            if not coin_id:
                return None

            cache_key = f"market_data_{coin_id}_{days}"
            if cache_key in self.cache:
                cached_data, cache_time = self.cache[cache_key]
                if datetime.now() - cache_time < self.cache_duration:
                    return cached_data

            # Get market data
            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {
                "vs_currency": "usd",
                "days": str(days),
                "interval": "daily"
            }
            if self.auth_method == "query":
                params["x_cg_demo_api_key"] = self.api_key

            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            market_data = response.json()

            df = self._process_market_data(market_data)
            self.cache[cache_key] = (df, datetime.now())
            
            return df

        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def get_historical_data(self, symbol: str, date: str) -> Optional[Dict]:
        """Get historical data for a specific date"""
        try:
            coin_id = self.get_coin_id(symbol)
            if not coin_id:
                self.logger.error(f"Could not find coin ID for symbol: {symbol}")
                return None

            url = f"{self.base_url}/coins/{coin_id}/history"
            params = {
                "date": date,  # Format: dd-mm-yyyy
                "localization": "false"
            }
            
            # Add API key to params if using query auth method
            if self.auth_method == "query":
                params["x_cg_demo_api_key"] = self.api_key

            self.logger.info(f"Fetching historical data for {coin_id} on {date}")
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 429:
                self.logger.warning("Rate limit reached. Waiting before retry...")
                time.sleep(60)  # Wait for 60 seconds before retry
                response = requests.get(url, headers=self.headers, params=params)
            
            response.raise_for_status()
            data = response.json()
            
            if not data:
                self.logger.warning(f"No data returned for {symbol} on {date}")
                return None
                
            return data

        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}")
            self.logger.error(f"URL: {url}")
            self.logger.error(f"Params: {params}")
            self.logger.error(f"Response: {response.text if 'response' in locals() else 'No response'}")
            self.logger.error(f"Exception: {str(e)}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            coin_id = self.get_coin_id(symbol)
            if not coin_id:
                return None

            url = f"{self.base_url}/simple/price"
            params = {
                "ids": coin_id,
                "vs_currencies": "usd"
            }
            
            if self.auth_method == "query":
                params["x_cg_demo_api_key"] = self.api_key

            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            return data[coin_id]['usd']

        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None

    def _process_market_data(self, market_data: Dict) -> pd.DataFrame:
        """Process market data into DataFrame"""
        try:
            timestamps = [datetime.fromtimestamp(price[0]/1000) for price in market_data['prices']]
            
            df = pd.DataFrame(index=timestamps)
            df['close'] = [price[1] for price in market_data['prices']]
            df['market_cap'] = [cap[1] for cap in market_data['market_caps']]
            df['volume'] = [vol[1] for vol in market_data['total_volumes']]
            
            return self._add_technical_indicators(df)

        except Exception as e:
            self.logger.error(f"Error processing market data: {str(e)}")
            raise
            
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame"""
        try:
            # Simple Moving Averages
            for period in self.config['technical']['indicators']['sma']:
                df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()

            # Exponential Moving Averages
            for period in self.config['technical']['indicators']['ema']:
                df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

            # Moving Average Convergence Divergence
            macd_config = self.config['technical']['indicators']['macd']
            df['EMA_fast'] = df['close'].ewm(span=macd_config['fast'], adjust=False).mean()
            df['EMA_slow'] = df['close'].ewm(span=macd_config['slow'], adjust=False).mean()
            df['MACD'] = df['EMA_fast'] - df['EMA_slow']
            df['MACD_Signal'] = df['MACD'].ewm(span=macd_config['signal'], adjust=False).mean()

            # Relative Strength Index
            rsi_period = self.config['technical']['indicators']['rsi']['period']
            delta = df['close'].diff()
            
            # Separate gain and loss
            gain = delta.copy()
            gain[gain < 0] = 0
            loss = -delta.copy()
            loss[loss < 0] = 0
            
            # Calculate average gain and loss
            avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
            avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
            
            # Calculate RS and RSI with handling for division by zero
            rs = avg_gain / avg_loss
            rs = rs.replace([np.inf, -np.inf], np.nan)
            
            df['RSI'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            bb_config = self.config['technical']['indicators']['bollinger']
            df['BB_middle'] = df['close'].rolling(window=bb_config['period']).mean()
            bb_std = df['close'].rolling(window=bb_config['period']).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std * bb_config['std_dev'])
            df['BB_lower'] = df['BB_middle'] - (bb_std * bb_config['std_dev'])

            # Handle NaN values
            df['RSI'] = df['RSI'].fillna(50)  # Fill RSI NaN with neutral value
            df['MACD'] = df['MACD'].fillna(0)
            df['MACD_Signal'] = df['MACD_Signal'].fillna(0)
            
            # Forward fill then backward fill remaining NaN values
            df = df.ffill().bfill()

            return df

        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            raise
def main():
    # Example configuration
    config = {
        'technical': {
            'coingecko': {
                'api_key': 'CG-bTKRKZS66KK19873LJPqx331',
                'auth_method': 'header'  # or 'query'
            },
            'indicators': {
                'sma': [20, 50],
                'ema': [12, 26],
                'macd': {
                    'fast': 12,
                    'slow': 26,
                    'signal': 9
                },
                'rsi': {
                    'period': 14
                },
                'bollinger': {
                    'period': 20,
                    'std_dev': 2
                }
            }
        }
    }

    collector = TechnicalDataCollector(config)
    
    # Example usage
    symbol = 'eth'
    
    # Get historical data
    historical_data = collector.get_historical_data(symbol, '01-05-2024')
    if historical_data:
        print("\nHistorical Data:")
        print(historical_data)
    
    # Get current price
    current_price = collector.get_current_price(symbol)
    if current_price:
        print(f"\nCurrent Price: ${current_price:,.2f}")
    
    # Get technical data
    df = collector.fetch_data(symbol, days=30)
    if df is not None:
        print("\nTechnical Data Summary:")
        print(f"Latest close price: ${df['close'].iloc[-1]:.2f}")
        print(f"Latest RSI: {df['RSI'].iloc[-1]:.2f}")
        print(f"Latest MACD: {df['MACD'].iloc[-1]:.2f}")
        print(f"24h Volume: ${df['volume'].iloc[-1]:,.2f}")
        
        # Save to CSV for analysis
        df.to_csv(f"{symbol}_technical_data.csv")
    else:
        print(f"Failed to fetch data for {symbol}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()