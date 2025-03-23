import sys
import os
import argparse
import logging
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.privy.silo_tools import SiloClaimRewardsToolPrivy
from src.connections.privy_connection import PrivyConnection
from src.connections.silo_connection import SiloConnection
from src.connections.sonic_connection import SonicConnection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_privy_silo_claim")

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

class MockLLM:
    """Mock LLM for testing"""
    pass

def test_silo_claim_rewards(sender):
    """Test Privy Silo claim rewards functionality"""
    try:
        # Create mock agent and LLM
        agent = MockAgent()
        llm = MockLLM()
        
        # Create Silo claim rewards tool
        silo_claim_tool = SiloClaimRewardsToolPrivy(agent, llm)
        
        # Execute claim
        result = silo_claim_tool._run(sender=sender)
        
        logger.info(result)
        
        # Final status
        if "success" in result and result["success"]:
            logger.info("✅ Test completed successfully!")
        else:
            logger.error("❌ Test failed!")
        
    except Exception as e:
        logger.error(f"Error during test: {str(e)}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Privy Silo claim rewards functionality")
    parser.add_argument("--sender", "-s", required=True, help="Sender wallet address")
    
    args = parser.parse_args()
    
    test_silo_claim_rewards(
        sender=args.sender
    )
