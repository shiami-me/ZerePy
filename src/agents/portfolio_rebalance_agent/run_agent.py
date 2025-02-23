import os
from typing import TypedDict, Dict, List, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
import numpy as np
import pandas as pd
from web3 import Web3
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
from langchain.agents import Tool
from datetime import datetime, timedelta
import yfinance as yf

# Load environment variables
load_dotenv()

# Constants
RISK_FREE_RATE = 0.02
MAX_ASSETS = 10
VOLATILITY_THRESHOLD = 0.5
VAR_THRESHOLD = 0.3
ES_THRESHOLD = 0.4
ANOMALY_THRESHOLD = 0.3
HIGH_ANOMALY_SCORE = 0.3
HIGH_VOLATILITY = 0.3

class RiskManager:
    """Class to manage portfolio risk calculations."""
    
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
        """Calculate various risk metrics for the given portfolio."""
        try:
            # Calculate returns directly from the market data
            returns = market_data.pct_change()
            
            portfolio_volatility = self._calculate_portfolio_volatility(returns, portfolio)
            var_95 = self._calculate_var(returns, portfolio, 0.95)
            var_99 = self._calculate_var(returns, portfolio, 0.99)
            es_95 = self._calculate_expected_shortfall(returns, portfolio, 0.95)
            anomaly_score = self._detect_anomalies(returns)
            
            return {
                'portfolio_volatility': portfolio_volatility,
                'value_at_risk_95': var_95,
                'value_at_risk_99': var_99,
                'expected_shortfall_95': es_95,
                'anomaly_score': anomaly_score,
                'risk_rating': self._calculate_risk_rating(portfolio_volatility, var_95, anomaly_score, market_conditions)
            }
        except Exception as e:
            print(f"Error calculating portfolio risk: {e}")
            return {}

            
    def _calculate_portfolio_volatility(self, returns: pd.DataFrame, portfolio: Dict[str, float]) -> float:
        total_value = sum(portfolio.values())
        weights = np.array([portfolio[asset] / total_value for asset in portfolio])
        cov_matrix = returns.cov()
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return np.sqrt(portfolio_variance * 252)  # Annualized
    
    def _calculate_var(self, returns: pd.DataFrame, portfolio: Dict[str, float], confidence_level: float) -> float:
        total_value = sum(portfolio.values())
        weights = {asset: value / total_value for asset, value in portfolio.items()}
        portfolio_returns = sum(returns[asset] * weight for asset, weight in weights.items())
        return np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    
    def _calculate_expected_shortfall(self, returns: pd.DataFrame, portfolio: Dict[str, float], confidence_level: float) -> float:
        var = self._calculate_var(returns, portfolio, confidence_level)
        total_value = sum(portfolio.values())
        weights = {asset: value / total_value for asset, value in portfolio.items()}
        portfolio_returns = sum(returns[asset] * weight for asset, weight in weights.items())
        return -np.mean(portfolio_returns[portfolio_returns < -var])
    
    def _detect_anomalies(self, returns: pd.DataFrame) -> float:
        anomaly_scores = self.risk_model.fit_predict(returns.values.reshape(-1, 1))
        return np.mean(anomaly_scores == -1)
    
    def _calculate_risk_rating(self, volatility: float, var: float, anomaly_score: float, market_conditions: Dict) -> str:
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
    
    def _market_condition_score(self, market_conditions: Dict) -> float:
        risk_levels = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
        return risk_levels.get(market_conditions.get('risk_level', 'medium'), 0.6)

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
        result = df.copy()
        
        for column in df.columns:
            result[f'SMA_20_{column}'] = df[column].rolling(window=20).mean()
            result[f'SMA_50_{column}'] = df[column].rolling(window=50).mean()
            result[f'RSI_{column}'] = self._calculate_rsi(df[column])
            result[f'MACD_{column}'] = self._calculate_macd(df[column])
            result[f'Volatility_{column}'] = df[column].rolling(window=20).std()
        
        return result

    
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
        if market_data.empty:
            raise ValueError("Market data is empty")
            
        df = self.calculate_technical_indicators(market_data)
        
        # Get the latest data
        latest_data = df.iloc[-1]
        
        # Initialize technical indicators
        technical_indicators = {
            'trend': {},
            'rsi': {},
            'volatility': {},
        }
        
        market_score = 0
        
        for asset in market_data.columns:
            try:
                # Calculate individual asset metrics
                sma_20 = df[f'SMA_20_{asset}'].iloc[-1]
                sma_50 = df[f'SMA_50_{asset}'].iloc[-1]
                rsi = df[f'RSI_{asset}'].iloc[-1]
                volatility = df[f'Volatility_{asset}'].iloc[-1]
                
                # Store technical indicators
                technical_indicators['trend'][asset] = 'bullish' if sma_20 > sma_50 else 'bearish'
                technical_indicators['rsi'][asset] = float(rsi)
                technical_indicators['volatility'][asset] = float(volatility)
                
                # Update market score
                market_score += (1 if sma_20 > sma_50 else -1)
                
            except Exception as e:
                print(f"Error analyzing {asset}: {e}")
                continue
        
        # Normalize market score to 0-1
        market_score = (market_score / len(market_data.columns) + 1) / 2
        
        return {
            'market_score': float(market_score),
            'technical_indicators': technical_indicators,
            'risk_level': 'medium'  # Default to medium if can't calculate
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
        self.risk_free_rate = RISK_FREE_RATE
        self.market_analyzer = MarketAnalyzer()

    def optimize(self, 
                historical_prices: pd.DataFrame, 
                current_portfolio: Dict[str, float],
                risk_tolerance: float = 0.5,
                market_conditions: Dict = None) -> Dict:
        
        mu = expected_returns.mean_historical_return(historical_prices)
        S = risk_models.CovarianceShrinkage(historical_prices).ledoit_wolf()
        
        ef = EfficientFrontier(mu, S)
        
        total_value = sum(current_portfolio.values())
        current_weights = {asset: value / total_value for asset, value in current_portfolio.items()}
        
        if market_conditions and market_conditions['risk_level'] == 'high':
            max_weight = 0.3
            min_weight = 0.05
        else:
            max_weight = 0.4
            min_weight = 0.03
            
        ef.add_constraint(lambda w: w >= min_weight)
        ef.add_constraint(lambda w: w <= max_weight)
        
        if risk_tolerance < 0.3:
            weights = ef.min_volatility()
        elif risk_tolerance < 0.7:
            weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        else:
            ef.objective_functions.update({
                'utility': objective_functions.L2_reg(gamma=risk_tolerance)
            })
            weights = ef.max_quadratic_utility()
        
        cleaned_weights = ef.clean_weights()
        
        rebalancing_actions = self._calculate_rebalancing_actions(current_weights, cleaned_weights)
        
        performance_metrics = {
            'expected_annual_return': ef.portfolio_performance(risk_free_rate=self.risk_free_rate)[0],
            'annual_volatility': ef.portfolio_performance(risk_free_rate=self.risk_free_rate)[1],
            'sharpe_ratio': ef.portfolio_performance(risk_free_rate=self.risk_free_rate)[2]
        }
        
        return {
            'optimal_weights': cleaned_weights,
            'rebalancing_actions': rebalancing_actions,
            'performance_metrics': performance_metrics
        }
    
    def _calculate_rebalancing_actions(self, current_weights: Dict[str, float], target_weights: Dict[str, float], threshold: float = 0.05) -> List[Dict]:
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
        
        return sorted(actions, key=lambda x: x['weight_difference'], reverse=True)

class YieldOptimizer:
    def __init__(self):
        self.defi_protocols = {
            'aave': 'https://api.aave.com',
            'compound': 'https://api.compound.finance',
            'curve': 'https://api.curve.fi',
            'yearn': 'https://api.yearn.finance',
        }
        
    async def find_opportunities(self, portfolio: Dict[str, float], risk_tolerance: float, market_conditions: Dict) -> List[Dict]:
        opportunities = []
        risk_threshold = self._adjust_risk_threshold(risk_tolerance, market_conditions)
        
        for asset, amount in portfolio.items():
            lending_ops = await self._get_lending_opportunities(asset)
            lp_ops = await self._get_lp_opportunities(asset)
            staking_ops = await self._get_staking_opportunities(asset)
            asset_ops = self._combine_opportunities(lending_ops, lp_ops, staking_ops, risk_threshold)
            opportunities.extend(asset_ops)
        
        ranked_ops = self._rank_opportunities(opportunities, risk_tolerance, market_conditions)
        return ranked_ops
    
    def _adjust_risk_threshold(self, base_tolerance: float, market_conditions: Dict) -> float:
        market_risk = {'low': 1.2, 'medium': 1.0, 'high': 0.8}.get(market_conditions['risk_level'], 1.0)
        return base_tolerance * market_risk
    
    async def _get_lending_opportunities(self, asset: str) -> List[Dict]:
        opportunities = []
        for protocol, api_url in self.defi_protocols.items():
            try:
                response = {'apy': 0.05, 'tvl': 1000000, 'utilization': 0.8}  # Placeholder
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
    
    async def _get_lp_opportunities(self, asset: str) -> List[Dict]:
        opportunities = []
        lp_pairs = {
            'ETH': [{'pair': 'ETH-USDC', 'protocol': 'uniswap', 'base_apy': 0.15}, {'pair': 'ETH-USDT', 'protocol': 'sushiswap', 'base_apy': 0.14}, {'pair': 'ETH-DAI', 'protocol': 'curve', 'base_apy': 0.12}],
            'USDC': [{'pair': 'USDC-ETH', 'protocol': 'uniswap', 'base_apy': 0.15}, {'pair': 'USDC-USDT', 'protocol': 'curve', 'base_apy': 0.08}, {'pair': 'USDC-DAI', 'protocol': 'balancer', 'base_apy': 0.07}],
            'BTC': [{'pair': 'BTC-ETH', 'protocol': 'sushiswap', 'base_apy': 0.16}, {'pair': 'BTC-USDC', 'protocol': 'uniswap', 'base_apy': 0.14}, {'pair': 'BTC-USDT', 'protocol': 'curve', 'base_apy': 0.13}]
        }
        
        if asset in lp_pairs:
            for pool in lp_pairs[asset]:
                try:
                    apy_variation = np.random.uniform(-0.02, 0.02)
                    current_apy = pool['base_apy'] + apy_variation
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
                            'min_amount': 100,
                            'tokens_needed': pool['pair'].split('-')
                        }
                    })
                except Exception as e:
                    print(f"Error fetching {pool['protocol']} LP data: {e}")
                    continue
                    
        return opportunities    
    
    async def _get_staking_opportunities(self, asset: str) -> List[Dict]:
        opportunities = []
        staking_protocols = {
            'ETH': [{'protocol': 'lido', 'base_apy': 0.04}, {'protocol': 'rocketpool', 'base_apy': 0.045}, {'protocol': 'stakewise', 'base_apy': 0.042}],
            'SOL': [{'protocol': 'marinade', 'base_apy': 0.06}, {'protocol': 'lido', 'base_apy': 0.058}],
            'DOT': [{'protocol': 'kraken', 'base_apy': 0.12}, {'protocol': 'binance', 'base_apy': 0.115}],
            'ADA': [{'protocol': 'binance', 'base_apy': 0.08}, {'protocol': 'kraken', 'base_apy': 0.075}]
        }
        
        if asset in staking_protocols:
            for protocol in staking_protocols[asset]:
                try:
                    apy_variation = np.random.uniform(-0.005, 0.005)
                    current_apy = protocol['base_apy'] + apy_variation
                    opportunities.append({
                        'type': 'staking',
                        'protocol': protocol['protocol'],
                        'asset': asset,
                        'apy': max(current_apy, 0),
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
    
    def _combine_opportunities(self, lending_ops: List[Dict], lp_ops: List[Dict], staking_ops: List[Dict], risk_threshold: float) -> List[Dict]:
        all_ops = lending_ops + lp_ops + staking_ops
        filtered_ops = [op for op in all_ops if self._calculate_opportunity_risk(op) <= risk_threshold]
        return filtered_ops
    
    def _calculate_opportunity_risk(self, opportunity: Dict) -> float:
        return 0.5  # Placeholder for actual risk calculation
    
    def _rank_opportunities(self, opportunities: List[Dict], risk_tolerance: float, market_conditions: Dict) -> List[Dict]:
        for op in opportunities:
            op['score'] = self._calculate_opportunity_score(op, risk_tolerance, market_conditions)
        return sorted(opportunities, key=lambda x: x['score'], reverse=True)
    
    def _calculate_opportunity_score(self, opportunity: Dict, risk_tolerance: float, market_conditions: Dict) -> float:
        return opportunity.get('apy', 0)

def calculate_risk_tolerance(portfolio: Dict[str, float], risk_metrics: Dict[str, float], market_analysis: Dict) -> float:
    num_assets = len(portfolio)
    portfolio_diversity = min(num_assets / MAX_ASSETS, 1.0)
    total_value = sum(portfolio.values())
    max_concentration = max(value / total_value for value in portfolio.values())
    concentration_score = 1 - max_concentration
    risk_score = _calculate_risk_score(risk_metrics)
    market_score = _calculate_market_score(market_analysis)
    
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
    
    risk_tolerance = _adjust_for_market_conditions(risk_tolerance, market_analysis, risk_metrics)
    return min(max(risk_tolerance, 0.0), 1.0)  

def _calculate_risk_score(risk_metrics: Dict[str, float]) -> float:
    volatility = risk_metrics.get('portfolio_volatility', 0)
    var_95 = abs(risk_metrics.get('value_at_risk_95', 0))
    es_95 = abs(risk_metrics.get('expected_shortfall_95', 0))
    anomaly_score = risk_metrics.get('anomaly_score', 0)
    
    norm_volatility = min(volatility / VOLATILITY_THRESHOLD, 1.0)  
    norm_var = min(var_95 / VAR_THRESHOLD, 1.0)  
    norm_es = min(es_95 / ES_THRESHOLD, 1.0)  
    
    risk_score = (
        0.4 * (1 - norm_volatility) + 
        0.3 * (1 - norm_var) +        
        0.2 * (1 - norm_es) +          
        0.1 * (1 - anomaly_score)      
    )
    
    return risk_score

def _calculate_market_score(market_analysis: Dict) -> float:
    technical_indicators = market_analysis.get('technical_indicators', {})
    market_score = market_analysis.get('market_score', 0.5)
    trend = technical_indicators.get('trend', 'neutral')
    trend_score = {'bullish': 0.8, 'neutral': 0.5, 'bearish': 0.2}.get(trend, 0.5)
    rsi = technical_indicators.get('rsi', 50)
    rsi_score = min(max(rsi / 100, 0), 1)
    volatility = technical_indicators.get('volatility', 0)
    volatility_score = max(1 - (volatility / VOLATILITY_THRESHOLD), 0)
    
    combined_score = (
        0.3 * trend_score +
        0.3 * rsi_score +
        0.2 * volatility_score +
        0.2 * market_score
    )
    
    return combined_score

def _adjust_for_market_conditions(base_tolerance: float, market_analysis: Dict, risk_metrics: Dict[str, float]) -> float:
    risk_level = market_analysis.get('risk_level', 'medium')
    adjustment_factors = {'low': 1.2, 'medium': 1.0, 'high': 0.8}
    base_adjustment = adjustment_factors.get(risk_level, 1.0)
    
    anomaly_score = risk_metrics.get('anomaly_score', 0)
    if anomaly_score > HIGH_ANOMALY_SCORE:
        base_adjustment *= 0.8
    
    volatility = risk_metrics.get('portfolio_volatility', 0)
    if volatility > HIGH_VOLATILITY:
        base_adjustment *= 0.9
    
    adjusted_tolerance = base_tolerance * base_adjustment
    return min(max(adjusted_tolerance, 0.0), 1.0)

def get_sample_market_data() -> pd.DataFrame:
    assets = ['ETH-USD', 'BTC-USD', 'USDC-USD']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    try:
        # Create an empty DataFrame with date index
        all_data = pd.DataFrame()
        
        for asset in assets:
            # Download data for each asset
            df = yf.download(asset, start=start_date, end=end_date)
            if not df.empty:
                # If this is the first asset, use its index
                if all_data.empty:
                    all_data = pd.DataFrame(index=df.index)
                # Add the Close price as a column
                all_data[asset] = df['Close']
        
        if all_data.empty:
            print("Warning: No data was downloaded")
            return pd.DataFrame()
            
        # Forward fill any missing values
        all_data = all_data.fillna(method='ffill')
        
        print("Market data shape:", all_data.shape)  # Debug print
        print("Market data columns:", all_data.columns.tolist())  # Debug print
        print("First few rows of market data:\n", all_data.head())  # Debug print
        
        return all_data
        
    except Exception as e:
        print(f"Error in get_sample_market_data: {str(e)}")
        return pd.DataFrame()


def fetch_wallet_data(state: PortfolioState) -> PortfolioState:
    try:
        wallet_address = state.get('wallet_address', '')
        print(f"Fetching data for wallet: {wallet_address}")  # Debug print
        
        market_data = get_sample_market_data()
        
        if market_data.empty:
            raise ValueError("No market data was retrieved")
            
        portfolio = {
            'ETH-USD': 2.0,
            'BTC-USD': 0.1,
            'USDC-USD': 5000
        }
        
        # Verify that we have data for all portfolio assets
        missing_assets = [asset for asset in portfolio.keys() if asset not in market_data.columns]
        if missing_assets:
            raise ValueError(f"Missing market data for assets: {missing_assets}")
        
        print("Portfolio assets:", list(portfolio.keys()))  # Debug print
        print("Market data assets:", list(market_data.columns))  # Debug print
        
        state['market_data'] = market_data
        state['portfolio'] = portfolio
        state['status'] = 'success'
        
    except Exception as e:
        print(f"Error in fetch_wallet_data: {str(e)}")
        state['status'] = 'error'
        state['error'] = f"Error fetching wallet data: {str(e)}"
    
    return state

def validate_market_data(market_data: pd.DataFrame) -> bool:
    """Validate that the market data DataFrame is properly structured."""
    if market_data is None or market_data.empty:
        print("Market data is None or empty")
        return False
        
    required_assets = ['ETH-USD', 'BTC-USD', 'USDC-USD']
    missing_assets = [asset for asset in required_assets if asset not in market_data.columns]
    
    if missing_assets:
        print(f"Missing required assets: {missing_assets}")
        return False
        
    if not isinstance(market_data.index, pd.DatetimeIndex):
        print("Market data index is not DatetimeIndex")
        return False
        
    return True


def rebalance_portfolio(state: PortfolioState) -> PortfolioState:
    try:
        if not validate_market_data(state.get('market_data')):
            raise ValueError("Invalid or missing market data")
            
        risk_manager = RiskManager()
        market_analyzer = MarketAnalyzer()
        
        market_analysis = market_analyzer.analyze_market(state['market_data'])
        state['market_analysis'] = market_analysis
        
        risk_metrics = risk_manager.calculate_portfolio_risk(
            state['portfolio'], 
            state['market_data'], 
            market_analysis
        )
        state['risk_metrics'] = risk_metrics
        
        state['status'] = 'success'
        
    except Exception as e:
        print(f"Error in rebalance_portfolio: {str(e)}")
        state['status'] = 'error'
        state['error'] = f"Error during portfolio rebalancing: {str(e)}"
    
    return state

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

def main():
    initial_state = PortfolioState(
        wallet_address="0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
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
        print("Starting portfolio rebalancing process...")
        
        final_state = agent.invoke(initial_state)
        
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
        else:
            print(f"Error: {final_state.get('error', 'Unknown error')}")
            print("State keys available:", list(final_state.keys()))  # Add debugging
            
    except Exception as e:
        print(f"An error occurred in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
