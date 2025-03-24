import sys
import os
import argparse
import logging
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.actions.sonic_actions import get_token_by_ticker, get_sonic_balance, send_sonic, swap_sonic, get_swap_summary

from src.tools.privy.silo_tools import SiloLoopingStrategyToolPrivy
from src.connections.privy_connection import PrivyConnection
from src.connections.silo_connection import SiloConnection
from src.connections.sonic_connection import SonicConnection
from src.connections.pendle_connection import PendleConnection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_privy_silo_looping")

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
        
        # Initialize Silo connection
        self.connections["silo"] = SiloConnection({})
        self.connections["sonic"] = SonicConnection({
            "network": "anvil"
        })
        self.connections["pendle"] = PendleConnection({})

class MockLLM:
    """Mock LLM for testing"""
    pass

def test_silo_looping(token_pair, initial_amount, sender, loops=3, borrow_percentage=95, id = 0, deposit_token="S"):
    """Test Privy Silo looping strategy functionality"""
    try:
        # Create mock agent and LLM
        agent = MockAgent()
        llm = MockLLM()
        
        # Create Silo looping tool
        silo_looping_tool = SiloLoopingStrategyToolPrivy(agent, llm)
        
        # Execute looping strategy
        result = silo_looping_tool._run(
            token=token_pair,
            initial_amount=float(initial_amount),
            sender=sender,
            loops=int(loops),
            borrow_percentage=float(borrow_percentage),
            id=int(id),
            deposit_token=deposit_token
        )
        
        # Log the result
        logger.info(result)
        
        # Final status
        import json
        result_dict = json.loads(result)
        if "status" in result_dict and result_dict["status"] == "success":
            logger.info("✅ Test completed successfully!")
            logger.info(f"Completed {result_dict.get('completed_loops', 0)} loops")
            logger.info(f"Final leverage: {result_dict.get('leverage', '1.00x')}")
            logger.info(f"Total deposited: {result_dict.get('total_deposited', initial_amount)}")
            logger.info(f"Total borrowed: {result_dict.get('total_borrowed', 0)}")
        else:
            logger.error(f"❌ Test failed! Error: {result_dict.get('error', 'Unknown error')}")
        
    except Exception as e:
        logger.error(f"Error during test: {str(e)}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Privy Silo looping strategy functionality")
    parser.add_argument("--token-pair", "-t", required=True, help="Comma-separated token pair (e.g., 'USDC,S')")
    parser.add_argument("--amount", "-a", required=True, help="Initial amount to deposit")
    parser.add_argument("--sender", "-s", required=True, help="Sender wallet address")
    parser.add_argument("--loops", "-l", default=3, help="Maximum number of loops to execute (default: 3)")
    parser.add_argument("--borrow-percentage", "-b", default=95, 
                        help="Percentage of max borrow to use in each loop (default: 95)")
    parser.add_argument("--id", "-i", default=0, help="Makret ID for the test run")
    parser.add_argument("--deposit-token", "-d", default="S", help="Token to deposit (default: S)")
    
    args = parser.parse_args()
    
    test_silo_looping(
        token_pair=args.token_pair,
        initial_amount=args.amount,
        sender=args.sender,
        loops=args.loops,
        borrow_percentage=args.borrow_percentage,
        id=args.id,
        deposit_token=args.deposit_token
    )
