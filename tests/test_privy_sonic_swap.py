import sys
import os
import argparse
import json
import logging
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.actions.sonic_actions import get_token_by_ticker, get_sonic_balance, send_sonic, swap_sonic, get_swap_summary
from src.tools.privy.sonic_tools import SonicSwapToolPrivy
from src.connections.privy_connection import PrivyConnection
from src.connections.sonic_connection import SonicConnection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_privy_sonic_swap")

class MockAgent:
    """Mock agent class for testing tools"""
    
    def __init__(self):
        """Initialize the mock agent with necessary connections"""
        self.connection_manager = MockConnectionManager()
        
    def get_tool(self, tool_name):
        """Mock method to get a tool by name"""
        return None

class MockConnectionManager:
    """Mock connection manager for testing"""
    
    def __init__(self):
        """Initialize with real connections for testing"""
        self.connections = {}
        
        # Load environment variables
        load_dotenv()
        
        # Initialize Privy connection
        self.connections["privy"] = PrivyConnection({
            "app_id": os.getenv("PRIVY_APP_ID"),
            "app_secret": os.getenv("PRIVY_APP_SECRET")
        })
        
        # Initialize Sonic connection
        self.connections["sonic"] = SonicConnection({
            "network": "anvil"
        })

class MockLLM:
    """Mock LLM for testing"""
    pass

def test_sonic_swap(from_token, to_token, amount, sender, slippage=0.5):
    """Test Privy Sonic swap functionality"""
    try:
        # Create mock agent and LLM
        agent = MockAgent()
        llm = MockLLM()
        
        # Create Sonic swap tool
        sonic_swap_tool = SonicSwapToolPrivy(agent, llm)
        
        # Execute swap
        result = sonic_swap_tool._run(
            from_token=from_token,
            to_token=to_token,
            amount=float(amount),
            sender=sender,
            slippage=float(slippage)
        )
        
        logger.info(result)
        
        # Log final status based on result
        if "success" in result and result["success"]:
            logger.info("✅ Test completed successfully!")
        else:
            logger.error("❌ Test failed!")
        
    except Exception as e:
        logger.error(f"Error during test: {str(e)}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Privy Sonic swap functionality")
    parser.add_argument("--from", "-f", dest="from_token", required=True, help="Token to swap from (e.g., S)")
    parser.add_argument("--to", "-t", dest="to_token", required=True, help="Token to swap to (e.g., USDC)")
    parser.add_argument("--amount", "-a", required=True, help="Amount to swap")
    parser.add_argument("--sender", "-s", required=True, help="Sender wallet address")
    parser.add_argument("--slippage", "-sl", default=0.5, help="Slippage tolerance in percentage (default: 0.5)")
    
    args = parser.parse_args()
    
    test_sonic_swap(
        from_token=args.from_token,
        to_token=args.to_token,
        amount=args.amount,
        sender=args.sender,
        slippage=args.slippage
    )
