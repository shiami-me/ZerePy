import sys
import os
import argparse
import json
import logging
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.actions.sonic_actions import wrap_sonic
from src.tools.privy.sonic_tools import SonicWrapToolPrivy
from src.connections.privy_connection import PrivyConnection
from src.connections.sonic_connection import SonicConnection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_privy_sonic_wrap")

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

def test_sonic_wrap(amount, sender):
    """Test Privy Sonic wrap functionality"""
    try:
        # Create mock agent and LLM
        agent = MockAgent()
        
        # Create Sonic wrap tool
        sonic_wrap_tool = SonicWrapToolPrivy(agent)
        
        # Execute wrap
        result = sonic_wrap_tool._run(
            amount=float(amount),
            sender=sender
        )
        
        # Log final status based on result
        if isinstance(result, dict) and result.get("status") == "success":
            logger.info("✅ Test completed successfully!")
            logger.info(f"Transaction hash: {result.get('tx')}")
        else:
            logger.error("❌ Test failed!")
        
    except Exception as e:
        logger.error(f"Error during test: {str(e)}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Privy Sonic wrap functionality")
    parser.add_argument("--amount", "-a", required=True, help="Amount of S tokens to wrap")
    parser.add_argument("--sender", "-s", required=True, help="Sender wallet address")
    
    args = parser.parse_args()
    
    test_sonic_wrap(
        amount=args.amount,
        sender=args.sender
    )
