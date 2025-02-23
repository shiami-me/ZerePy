import os
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
import numpy as np
import pandas as pd
from web3 import Web3
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
from langchain.agents import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import yfinance as yf


# Load environment variables
load_dotenv()


# Add at the beginning of the file, after the imports



class RiskManager:
    def __init__(self):
        self.risk_model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
    
    def calculate_portfolio_risk(
        self,
        portfolio: Dict[str, float],
        market_data: pd.DataFrame,
        market_conditions: Dict
    ) -> Dict[str, float]:
        
        # Calculate portfolio volatility
        returns = market_data.groupby('symbol')['Close'].pct_change()
        portfolio_volatility = self._calculate_portfolio_volatility(
            returns,
            portfolio
        )
        
        # Calculate Value at Risk
        var_95 = self._calculate_var(returns, portfolio, 0.95)
        var_99 = self._calculate_var(returns, portfolio, 0.99)
        
        # Calculate Expected Shortfall
        es_95 = self._calculate_expected_shortfall(returns, portfolio, 0.95)
        
        # Detect market anomalies
        anomaly_score = self._detect_anomalies(returns)
        
        return {
            'portfolio_volatility': portfolio_volatility,
            'value_at_risk_95': var_95,
            'value_at_risk_99': var_99,
            'expected_shortfall_95': es_95,
            'anomaly_score': anomaly_score,
            'risk_rating': self._calculate_risk_rating(
                portfolio_volatility,
                var_95,
                anomaly_score,
                market_conditions
            )
        }
    
    def _calculate_portfolio_volatility(
        self,
        returns: pd.DataFrame,
        portfolio: Dict[str, float]
    ) -> float:
        # Calculate portfolio variance using weights
        total_value = sum(portfolio.values())
        weights = np.array([
            portfolio[asset]/total_value 
            for asset in portfolio
        ])
        
        cov_matrix = returns.cov()
        portfolio_variance = np.dot(
            weights.T,
            np.dot(cov_matrix, weights)
        )
        
        return np.sqrt(portfolio_variance * 252)  # Annualized
    
    def _calculate_var(
        self,
        returns: pd.DataFrame,
        portfolio: Dict[str, float],
        confidence_level: float
    ) -> float:
        # Calculate weighted portfolio returns
        total_value = sum(portfolio.values())
        weights = {
            asset: value/total_value 
            for asset, value in portfolio.items()
        }
        
        portfolio_returns = sum(
            returns[asset] * weight 
            for asset, weight in weights.items()
        )
        
        return np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    
    def _calculate_expected_shortfall(
        self,
        returns: pd.DataFrame,
        portfolio: Dict[str, float],
        confidence_level: float
    ) -> float:
        var = self._calculate_var(returns, portfolio, confidence_level)
        total_value = sum(portfolio.values())
        weights = {
            asset: value/total_value 
            for asset, value in portfolio.items()
        }
        
        portfolio_returns = sum(
            returns[asset] * weight 
            for asset, weight in weights.items()
        )
        
        return -np.mean(
            portfolio_returns[portfolio_returns < -var]
        )
    
    def _detect_anomalies(
        self,
        returns: pd.DataFrame
    ) -> float:
        # Fit isolation forest and get anomaly scores
        anomaly_scores = self.risk_model.fit_predict(
            returns.values.reshape(-1, 1)
        )
        
        # Calculate percentage of anomalies
        return np.mean(anomaly_scores == -1)
    
    def _calculate_risk_rating(
        self,
        volatility: float,
        var: float,
        anomaly_score: float,
        market_conditions: Dict
    ) -> str:
        # Combine multiple risk factors
        risk_score = (
            0.4 * volatility +
            0.3 * abs(var) +
            0.2 * anomaly_score +
            0.1 * self._market_condition_score(market_conditions)
        )
        
        if risk_score < 0.3:
            return 'low'
        elif risk_score < 0.7:
            return 'medium'
        else:
            return 'high'
    
    def _market_condition_score(
        self,
        market_conditions: Dict
    ) -> float:
        # Convert market conditions to numerical score
        risk_levels = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.9
        }
        return risk_levels.get(
            market_conditions.get('risk_level', 'medium'),
            0.6
        )


class PortfolioState(TypedDict):
    wallet_address: str
    portfolio: Dict[str, float]
    market_data: pd.DataFrame
    risk_metrics: Dict[str, float]
    rebalance_recommendations: Dict[str, float]
    yield_opportunities: List[Dict]
    market_analysis: Dict[str, any]
    conversation_history: List[str]
    status: str



class MarketAnalyzer:
    def __init__(self):
        self.sentiment_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4
        )
        self.scaler = StandardScaler()
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = self._calculate_rsi(df['Close'])
        df['MACD'] = self._calculate_macd(df['Close'])
        df['Volatility'] = df['Close'].rolling(window=20).std()
        df['ATR'] = self._calculate_atr(df)
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_macd(prices: pd.Series) -> pd.Series:
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        return exp1 - exp2
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()

    def analyze_market(self, market_data: pd.DataFrame) -> Dict:
        df = self.calculate_technical_indicators(market_data)
        features = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Volatility', 'ATR']
        df = df.dropna()
        
        X = self.scaler.fit_transform(df[features])
        market_score = self.sentiment_model.predict(X)
        
        latest_data = df.iloc[-1]
        
        return {
            'market_score': float(np.mean(market_score)),
            'technical_indicators': {
                'trend': 'bullish' if latest_data['SMA_20'] > latest_data['SMA_50'] else 'bearish',
                'rsi': float(latest_data['RSI']),
                'volatility': float(latest_data['Volatility']),
                'atr': float(latest_data['ATR'])
            },
            'risk_level': self._calculate_risk_level(latest_data)
        }
    
    def _calculate_risk_level(self, latest_data: pd.Series) -> str:
        risk_score = (
            (latest_data['RSI'] / 100) * 0.3 +
            (latest_data['Volatility'] / latest_data['Close']) * 0.4 +
            (latest_data['ATR'] / latest_data['Close']) * 0.3
        )
        
        if risk_score < 0.3:
            return 'low'
        elif risk_score < 0.7:
            return 'medium'
        else:
            return 'high'

class PortfolioOptimizer:
    def __init__(self):
        self.risk_free_rate = 0.02
        self.market_analyzer = MarketAnalyzer()

    def optimize(self, 
                historical_prices: pd.DataFrame, 
                current_portfolio: Dict[str, float],
                risk_tolerance: float = 0.5,
                market_conditions: Dict = None) -> Dict:
        
        # Calculate expected returns and covariance
        mu = expected_returns.mean_historical_return(historical_prices)
        S = risk_models.CovarianceShrinkage(historical_prices).ledoit_wolf()
        
        # Initialize efficient frontier
        ef = EfficientFrontier(mu, S)
        
        # Add constraints based on current portfolio
        total_value = sum(current_portfolio.values())
        current_weights = {
            asset: value/total_value 
            for asset, value in current_portfolio.items()
        }
        
        # Dynamic constraints based on market conditions
        if market_conditions and market_conditions['risk_level'] == 'high':
            max_weight = 0.3  # More conservative in high risk
            min_weight = 0.05
        else:
            max_weight = 0.4
            min_weight = 0.03
            
        ef.add_constraint(lambda w: w >= min_weight)
        ef.add_constraint(lambda w: w <= max_weight)
        
        # Add objective based on risk tolerance
        if risk_tolerance < 0.3:
            weights = ef.min_volatility()
        elif risk_tolerance < 0.7:
            weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        else:
            # Custom utility function for aggressive strategy
            ef.objective_functions.update({
                'utility': objective_functions.L2_reg(
                    gamma=risk_tolerance
                )
            })
            weights = ef.max_quadratic_utility()
        
        cleaned_weights = ef.clean_weights()
        
        # Calculate rebalancing needs
        rebalancing_actions = self._calculate_rebalancing_actions(
            current_weights,
            cleaned_weights
        )
        
        performance_metrics = {
            'expected_annual_return': ef.portfolio_performance(
                risk_free_rate=self.risk_free_rate
            )[0],
            'annual_volatility': ef.portfolio_performance(
                risk_free_rate=self.risk_free_rate
            )[1],
            'sharpe_ratio': ef.portfolio_performance(
                risk_free_rate=self.risk_free_rate
            )[2]
        }
        
        return {
            'optimal_weights': cleaned_weights,
            'rebalancing_actions': rebalancing_actions,
            'performance_metrics': performance_metrics
        }
    
    def _calculate_rebalancing_actions(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        threshold: float = 0.05
    ) -> List[Dict]:
        
        actions = []
        for asset in target_weights:
            current = current_weights.get(asset, 0)
            target = target_weights[asset]
            diff = target - current
            
            if abs(diff) > threshold:
                actions.append({
                    'asset': asset,
                    'action': 'buy' if diff > 0 else 'sell',
                    'weight_difference': abs(diff),
                    'priority': 'high' if abs(diff) > 0.1 else 'medium'
                })
        
        return sorted(
            actions,
            key=lambda x: x['weight_difference'],
            reverse=True
        )

class YieldOptimizer:
    def __init__(self):
        self.defi_protocols = {
            'aave': 'https://api.aave.com',
            'compound': 'https://api.compound.finance',
            'curve': 'https://api.curve.fi',
            'yearn': 'https://api.yearn.finance',
        }
        
    async def find_opportunities(
        self,
        portfolio: Dict[str, float],
        risk_tolerance: float,
        market_conditions: Dict
    ) -> List[Dict]:
        
        opportunities = []
        
        # Adjust risk threshold based on market conditions
        risk_threshold = self._adjust_risk_threshold(
            risk_tolerance,
            market_conditions
        )
        
        for asset, amount in portfolio.items():
            # Get lending opportunities
            lending_ops = await self._get_lending_opportunities(asset)
            
            # Get liquidity pool opportunities
            lp_ops = await self._get_lp_opportunities(asset)
            
            # Get staking opportunities
            staking_ops = await self._get_staking_opportunities(asset)
            
            # Combine and filter opportunities
            asset_ops = self._combine_opportunities(
                lending_ops,
                lp_ops,
                staking_ops,
                risk_threshold
            )
            
            opportunities.extend(asset_ops)
        
        # Sort and rank opportunities
        ranked_ops = self._rank_opportunities(
            opportunities,
            risk_tolerance,
            market_conditions
        )
        
        return ranked_ops
    
    def _adjust_risk_threshold(
        self,
        base_tolerance: float,
        market_conditions: Dict
    ) -> float:
        
        market_risk = {
            'low': 1.2,
            'medium': 1.0,
            'high': 0.8
        }.get(market_conditions['risk_level'], 1.0)
        
        return base_tolerance * market_risk
    
    async def _get_lending_opportunities(
        self,
        asset: str
    ) -> List[Dict]:
        opportunities = []
        
        for protocol, api_url in self.defi_protocols.items():
            try:
                # Fetch lending rates
                # This is a placeholder - implement actual API calls
                response = {
                    'apy': 0.05,
                    'tvl': 1000000,
                    'utilization': 0.8
                }
                
                opportunities.append({
                    'type': 'lending',
                    'protocol': protocol,
                    'asset': asset,
                    'apy': response['apy'],
                    'risk_metrics': {
                        'tvl': response['tvl'],
                        'utilization': response['utilization']
                    }
                })
            except Exception as e:
                print(f"Error fetching {protocol} data: {e}")
                continue
        
        return opportunities
    
    async def _get_lp_opportunities(
        self,
        asset: str
    ) -> List[Dict]:
        opportunities = []
        
        # Define common LP pairs for different assets
        lp_pairs = {
            'ETH': [
                {'pair': 'ETH-USDC', 'protocol': 'uniswap', 'base_apy': 0.15},
                {'pair': 'ETH-USDT', 'protocol': 'sushiswap', 'base_apy': 0.14}, 
                {'pair': 'ETH-DAI', 'protocol': 'curve', 'base_apy': 0.12}
            ],
            'USDC': [
                {'pair': 'USDC-ETH', 'protocol': 'uniswap', 'base_apy': 0.15},
                {'pair': 'USDC-USDT', 'protocol': 'curve', 'base_apy': 0.08},
                {'pair': 'USDC-DAI', 'protocol': 'balancer', 'base_apy': 0.07}
            ],
            'BTC': [
                {'pair': 'BTC-ETH', 'protocol': 'sushiswap', 'base_apy': 0.16},
                {'pair': 'BTC-USDC', 'protocol': 'uniswap', 'base_apy': 0.14},
                {'pair': 'BTC-USDT', 'protocol': 'curve', 'base_apy': 0.13}
            ]
        }
        
        if asset in lp_pairs:
            for pool in lp_pairs[asset]:
                try:
                    # Add randomization to APY to simulate market conditions
                    apy_variation = np.random.uniform(-0.02, 0.02)
                    current_apy = pool['base_apy'] + apy_variation
                    
                    # Get pool metrics
                    pool_metrics = {
                        'tvl': np.random.uniform(1000000, 10000000),
                        'volume_24h': np.random.uniform(100000, 1000000),
                        'fee_tier': np.random.choice([0.001, 0.003, 0.005])
                    }
                    
                    opportunities.append({
                        'type': 'liquidity_pool',
                        'protocol': pool['protocol'],
                        'pair': pool['pair'],
                        'asset': asset,
                        'apy': max(current_apy, 0),
                        'risk_metrics': {
                            'tvl': pool_metrics['tvl'],
                            'volume_24h': pool_metrics['volume_24h'],
                            'fee_tier': pool_metrics['fee_tier'],
                            'impermanent_loss_risk': self._calculate_il_risk(pool['pair']),
                            'pool_concentration': np.random.uniform(0.1, 0.9)
                        },
                        'requirements': {
                            'min_amount': 100,  # Minimum liquidity requirement
                            'tokens_needed': pool['pair'].split('-')
                        }
                    })
                except Exception as e:
                    print(f"Error fetching {pool['protocol']} LP data: {e}")
                    continue
                    
        return opportunities    
    
    async def _get_staking_opportunities(
        self,
        asset: str
    ) -> List[Dict]:
        opportunities = []
        
        # Define staking protocols and their base APYs
        staking_protocols = {
            'ETH': [
                {'protocol': 'lido', 'base_apy': 0.04},
                {'protocol': 'rocketpool', 'base_apy': 0.045},
                {'protocol': 'stakewise', 'base_apy': 0.042}
            ],
            'SOL': [
                {'protocol': 'marinade', 'base_apy': 0.06},
                {'protocol': 'lido', 'base_apy': 0.058}
            ],
            'DOT': [
                {'protocol': 'kraken', 'base_apy': 0.12},
                {'protocol': 'binance', 'base_apy': 0.115}
            ],
            'ADA': [
                {'protocol': 'binance', 'base_apy': 0.08},
                {'protocol': 'kraken', 'base_apy': 0.075}
            ]
        }
        
        if asset in staking_protocols:
            for protocol in staking_protocols[asset]:
                try:
                    # Add some randomization to APY to simulate market conditions
                    apy_variation = np.random.uniform(-0.005, 0.005)
                    current_apy = protocol['base_apy'] + apy_variation
                    
                    opportunities.append({
                        'type': 'staking',
                        'protocol': protocol['protocol'],
                        'asset': asset,
                        'apy': max(current_apy, 0),  # Ensure APY doesn't go negative
                        'risk_metrics': {
                            'protocol_risk': self._get_protocol_risk(protocol['protocol']),
                            'lockup_period': self._get_lockup_period(protocol['protocol']),
                            'min_stake': self._get_min_stake(protocol['protocol'])
                        },
                        'requirements': {
                            'min_amount': self._get_min_stake(protocol['protocol']),
                            'lockup_period': self._get_lockup_period(protocol['protocol'])
                        }
                    })
                except Exception as e:
                    print(f"Error fetching {protocol['protocol']} staking data: {e}")
                    continue
                    
        return opportunities  
      
    def _combine_opportunities(
        self,
        lending_ops: List[Dict],
        lp_ops: List[Dict],
        staking_ops: List[Dict],
        risk_threshold: float
    ) -> List[Dict]:
        
        all_ops = lending_ops + lp_ops + staking_ops
        
        # Filter based on risk threshold
        filtered_ops = [
            op for op in all_ops
            if self._calculate_opportunity_risk(op) <= risk_threshold
        ]
        
        return filtered_ops
    
    def _calculate_opportunity_risk(
        self,
        opportunity: Dict
    ) -> float:
        # Implement risk calculation logic
        return 0.5
    
    def _rank_opportunities(
        self,
        opportunities: List[Dict],
        risk_tolerance: float,
        market_conditions: Dict
    ) -> List[Dict]:
        
        for op in opportunities:
            op['score'] = self._calculate_opportunity_score(
                op,
                risk_tolerance,
                market_conditions
            )
        
        return sorted(
            opportunities,
            key=lambda x: x['score'],
            reverse=True
        )
    
    def _calculate_opportunity_score(
        self,
        opportunity: Dict,
        risk_tolerance: float,
        market_conditions: Dict
    ) -> float:
        # Implement scoring logic
        return opportunity.get('apy', 0)

# Initialize tools
market_analyzer = MarketAnalyzer()
portfolio_optimizer = PortfolioOptimizer()
yield_optimizer = YieldOptimizer()

# Define tools
tools = [
    Tool(
        name="analyze_market",
        func=market_analyzer.analyze_market,
        description="Analyzes current market conditions"
    ),
    Tool(
        name="optimize_portfolio",
        func=portfolio_optimizer.optimize,
        description="Optimizes portfolio allocation"
    ),
    Tool(
        name="find_yield_opportunities",
        func=yield_optimizer.find_opportunities,
        description="Finds yield opportunities"
    )
]

def calculate_risk_tolerance(
    portfolio: Dict[str, float],
    risk_metrics: Dict[str, float],
    market_analysis: Dict
) -> float:
    """
    Calculate risk tolerance based on portfolio composition, risk metrics, and market conditions.
    Returns a value between 0 and 1, where:
    - 0-0.3: Conservative
    - 0.3-0.7: Moderate
    - 0.7-1.0: Aggressive
    
    Args:
        portfolio: Dictionary of asset holdings and their values
        risk_metrics: Dictionary containing portfolio risk measurements
        market_analysis: Dictionary containing market condition analysis
    
    Returns:
        float: Risk tolerance score between 0 and 1
    """
    # 1. Portfolio Diversity Score
    num_assets = len(portfolio)
    portfolio_diversity = min(num_assets / 10, 1.0)  # Cap at 10 assets
    
    # 2. Portfolio Concentration Score
    total_value = sum(portfolio.values())
    max_concentration = max(value / total_value for value in portfolio.values())
    concentration_score = 1 - max_concentration  # Lower concentration = higher score
    
    # 3. Risk Metrics Score
    risk_score = _calculate_risk_score(risk_metrics)
    
    # 4. Market Conditions Score
    market_score = _calculate_market_score(market_analysis)
    
    # 5. Calculate final risk tolerance
    weights = {
        'portfolio_diversity': 0.2,
        'concentration': 0.2,
        'risk_metrics': 0.3,
        'market_conditions': 0.3
    }
    
    risk_tolerance = (
        weights['portfolio_diversity'] * portfolio_diversity +
        weights['concentration'] * concentration_score +
        weights['risk_metrics'] * risk_score +
        weights['market_conditions'] * market_score
    )
    
    risk_tolerance = _adjust_for_market_conditions(
        risk_tolerance,
        market_analysis,
        risk_metrics
    )
    
    return min(max(risk_tolerance, 0.0), 1.0)  

def _calculate_risk_score(risk_metrics: Dict[str, float]) -> float:
    """
    Calculate a risk score based on portfolio risk metrics.
    """
    # Extract key risk metrics
    volatility = risk_metrics.get('portfolio_volatility', 0)
    var_95 = abs(risk_metrics.get('value_at_risk_95', 0))
    es_95 = abs(risk_metrics.get('expected_shortfall_95', 0))
    anomaly_score = risk_metrics.get('anomaly_score', 0)
    
    # Normalize metrics
    norm_volatility = min(volatility / 0.5, 1.0)  
    norm_var = min(var_95 / 0.3, 1.0)  
    norm_es = min(es_95 / 0.4, 1.0)  
    
    # Combine metrics with weights
    risk_score = (
        0.4 * (1 - norm_volatility) + 
        0.3 * (1 - norm_var) +        
        0.2 * (1 - norm_es) +          
        0.1 * (1 - anomaly_score)      
    )
    
    return risk_score

def _calculate_market_score(market_analysis: Dict) -> float:
    """
    Calculate a market conditions score based on market analysis.
    """
    # Extract market indicators
    technical_indicators = market_analysis.get('technical_indicators', {})
    market_score = market_analysis.get('market_score', 0.5)
    
    # Get trend direction
    trend = technical_indicators.get('trend', 'neutral')
    trend_score = {
        'bullish': 0.8,
        'neutral': 0.5,
        'bearish': 0.2
    }.get(trend, 0.5)
    
    # Get RSI
    rsi = technical_indicators.get('rsi', 50)
    rsi_score = min(max(rsi / 100, 0), 1)
    
    # Get volatility
    volatility = technical_indicators.get('volatility', 0)
    volatility_score = max(1 - (volatility / 0.5), 0)  # Cap at 50% volatility
    
    # Combine scores
    combined_score = (
        0.3 * trend_score +
        0.3 * rsi_score +
        0.2 * volatility_score +
        0.2 * market_score
    )
    
    return combined_score

def _adjust_for_market_conditions(
    base_tolerance: float,
    market_analysis: Dict,
    risk_metrics: Dict[str, float]
) -> float:
    """
    Adjust risk tolerance based on market conditions and risk metrics.
    """
    # Get market risk level
    risk_level = market_analysis.get('risk_level', 'medium')
    
    # Define adjustment factors
    adjustment_factors = {
        'low': 1.2,      # Increase risk tolerance in low-risk markets
        'medium': 1.0,   # No adjustment
        'high': 0.8      # Decrease risk tolerance in high-risk markets
    }
    
    # Get base adjustment
    base_adjustment = adjustment_factors.get(risk_level, 1.0)
    
    # Further adjust based on anomaly score
    anomaly_score = risk_metrics.get('anomaly_score', 0)
    if anomaly_score > 0.3:  # High anomaly detection
        base_adjustment *= 0.8
    
    # Apply volatility adjustment
    volatility = risk_metrics.get('portfolio_volatility', 0)
    if volatility > 0.3:  # High volatility
        base_adjustment *= 0.9
    
    # Apply final adjustment
    adjusted_tolerance = base_tolerance * base_adjustment
    
    return min(max(adjusted_tolerance, 0.0), 1.0)  # Ensure bounds


def get_sample_market_data():
    """Get sample market data using yfinance"""
    assets = ['ETH-USD', 'BTC-USD', 'USDC-USD']
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    data = pd.DataFrame()
    for asset in assets:
        try:
            ticker_data = yf.download(asset, start=start_date, end=end_date)
            if not ticker_data.empty:
                data[asset] = ticker_data['Close']
        except Exception as e:
            print(f"Error fetching data for {asset}: {e}")
    
    return data


def fetch_wallet_data(state: PortfolioState) -> PortfolioState:
    """
    Fetches wallet data and market data for the portfolio
    """
    try:
        # Sample wallet address - replace with actual wallet address
        wallet_address = state.get('wallet_address', '')
        
        # Fetch market data
        market_data = get_sample_market_data()
        
        # Sample portfolio data - in production, you would fetch this from the blockchain
        portfolio = {
            'ETH-USD': 2.0,    # 2 ETH
            'BTC-USD': 0.1,    # 0.1 BTC
            'USDC-USD': 5000   # 5000 USDC
        }
        
        # Update state
        state['market_data'] = market_data
        state['portfolio'] = portfolio
        state['status'] = 'success'
        
    except Exception as e:
        state['status'] = 'error'
        state['error'] = str(e)
    
    return state
def rebalance_portfolio(state: PortfolioState) -> PortfolioState:
    """Main rebalancing function"""
    try:
        # Initialize risk manager
        risk_manager = RiskManager()
        
        # Analyze market conditions
        market_analysis = market_analyzer.analyze_market(state['market_data'])
        state['market_analysis'] = market_analysis
        
        # Calculate portfolio risk metrics
        risk_metrics = risk_manager.calculate_portfolio_risk(
            state['portfolio'],
            state['market_data'],
            market_analysis
        )
        state['risk_metrics'] = risk_metrics
        
        # Calculate risk tolerance based on risk metrics and market conditions
        risk_tolerance = calculate_risk_tolerance(
            state['portfolio'],
            risk_metrics,
            market_analysis
        )
        
        # Rest of the function remains the same...
    except Exception as e:
        state['status'] = 'error'
        state['error'] = str(e)
    
    return state
# Build workflow
workflow = StateGraph(PortfolioState)

# Add nodes
workflow.add_node("fetch_data", fetch_wallet_data)  # Implement this
workflow.add_node("rebalance", rebalance_portfolio)

# Set workflow
workflow.set_entry_point("fetch_data")
workflow.add_edge("fetch_data", "rebalance")
workflow.add_edge("rebalance", END)

# Compile workflow
agent = workflow.compile()

def main():
    # Create initial state
    initial_state = PortfolioState(
        wallet_address="0x742d35Cc6634C0532925a3b844Bc454e4438f44e",  # Example address
        portfolio={},
        market_data=pd.DataFrame(),
        risk_metrics={},
        rebalance_recommendations={},
        yield_opportunities=[],
        market_analysis={},
        conversation_history=[],
        status="initialized"
    )

    try:
        # Run the workflow
        final_state = agent.invoke(initial_state)
        
        # Print results
        print("\nFinal Portfolio State:")
        print(f"Status: {final_state['status']}")
        if final_state['status'] == 'success':
            print("\nPortfolio:")
            for asset, amount in final_state['portfolio'].items():
                print(f"{asset}: {amount}")
            
            print("\nRisk Metrics:")
            for metric, value in final_state['risk_metrics'].items():
                print(f"{metric}: {value}")
            
            print("\nMarket Analysis:")
            for key, value in final_state['market_analysis'].items():
                print(f"{key}: {value}")
            
            if 'rebalance_recommendations' in final_state:
                print("\nRebalancing Recommendations:")
                for asset, rec in final_state['rebalance_recommendations'].items():
                    print(f"{asset}: {rec}")
        else:
            print(f"Error: {final_state.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Initialize components and workflow
    risk_manager = RiskManager()
    market_analyzer = MarketAnalyzer()
    portfolio_optimizer = PortfolioOptimizer()
    yield_optimizer = YieldOptimizer()

    # Build workflow
    workflow = StateGraph(PortfolioState)

    # Add nodes
    workflow.add_node("fetch_data", fetch_wallet_data)
    workflow.add_node("rebalance", rebalance_portfolio)

    # Set workflow
    workflow.set_entry_point("fetch_data")
    workflow.add_edge("fetch_data", "rebalance")
    workflow.add_edge("rebalance", END)

    # Compile workflow
    agent = workflow.compile()

    # Run main function
    main()


