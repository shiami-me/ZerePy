from datetime import datetime, timedelta
import pandas as pd
from typing import Dict

# Import our agent 
from portfolio_agent import agent

def run_portfolio_rebalancing():
    # Sample initial state
    initial_state = {
        "wallet_address": "0xYourWalletAddress",
        "portfolio": {
            "ETH": 2.5,
            "USDC": 5000.0,
            "WBTC": 0.15
        },
        "market_data": pd.DataFrame(),  # Will be populated by the agent
        "risk_metrics": {},
        "rebalance_recommendations": {},
        "yield_opportunities": [],
        "market_analysis": {},
        "conversation_history": [],
        "status": "started"
    }

    try:
        # Run the agent
        result = agent.invoke(initial_state)

        # Check if rebalancing was successful
        if result['status'] == 'completed':
            print("\n=== Portfolio Rebalancing Results ===")
            
            # Print market analysis
            print("\nMarket Analysis:")
            print(f"Market Condition: {result['market_analysis']['risk_level']}")
            print(f"Market Score: {result['market_analysis']['market_score']:.2f}")
            
            # Print risk metrics
            print("\nRisk Metrics:")
            for metric, value in result['risk_metrics'].items():
                print(f"{metric}: {value:.4f}")
            
            # Print rebalancing recommendations
            print("\nRebalancing Recommendations:")
            for asset, action in result['rebalance_recommendations']['rebalancing_actions'].items():
                print(f"{asset}: {action['action']} - Priority: {action['priority']}")
            
            # Print yield opportunities
            print("\nTop Yield Opportunities:")
            for opp in result['yield_opportunities'][:3]:  # Top 3 opportunities
                print(f"Protocol: {opp['protocol']}")
                print(f"Type: {opp['type']}")
                print(f"APY: {opp['apy']:.2%}")
                print("---")
                
        else:
            print(f"Error during rebalancing: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error running portfolio rebalancing: {e}")

if __name__ == "__main__":
    run_portfolio_rebalancing()