from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

class TechnicalAnalyzer:
    def __init__(self):
        self.indicators = {}
        self.cached_analysis = {}
        self.cache_duration = 300  # 5 minutes

    def calculate_indicators(self, data: pd.Series) -> Dict[str, float]:
        """Calculate technical indicators for a given price series"""
        if data.empty:
            return {}

        # Convert series to dataframe with OHLCV structure
        df = pd.DataFrame({
            'close': data,
            'high': data,
            'low': data,
            'volume': data * 0  # placeholder since we don't have volume data
        })

        # Calculate RSI
        rsi = RSIIndicator(close=df['close'], window=14)
        current_rsi = rsi.rsi().iloc[-1]

        # Calculate MACD
        macd = MACD(close=df['close'])
        current_macd = macd.macd().iloc[-1]
        current_signal = macd.macd_signal().iloc[-1]

        # Calculate Bollinger Bands
        bb = BollingerBands(close=df['close'])
        bb_upper = bb.bollinger_hband().iloc[-1]
        bb_lower = bb.bollinger_lband().iloc[-1]
        
        # Calculate price trends
        sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
        sma_50 = df['close'].rolling(window=50).mean().iloc[-1]
        
        # Calculate volatility
        volatility = df['close'].pct_change().std() * np.sqrt(252)  # Annualized

        return {
            'rsi': round(current_rsi, 2),
            'macd': round(current_macd, 4),
            'macd_signal': round(current_signal, 4),
            'bb_upper': round(bb_upper, 2),
            'bb_lower': round(bb_lower, 2),
            'sma_20': round(sma_20, 2),
            'sma_50': round(sma_50, 2),
            'volatility': round(volatility * 100, 2),  # as percentage
            'current_price': round(df['close'].iloc[-1], 2),
            'trend': 'bullish' if sma_20 > sma_50 else 'bearish',
            'overbought': current_rsi > 70,
            'oversold': current_rsi < 30
        }
