import sys
import os
import argparse
import logging
import json
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tools.privy.silo_tools import SiloExitStrategyToolPrivy
from src.connections.privy_connection import PrivyConnection
from src.connections.silo_connection import SiloConnection
from src.connections.sonic_connection import SonicConnection
from src.actions.sonic_actions import get_token_by_ticker, get_sonic_balance, send_sonic, swap_sonic, get_swap_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_privy_silo_exit_strategy")


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


def test_silo_exit_strategy(strategy_result, sender, swap_slippage=0.5, id=0):
    """Test Privy Silo strategy exit functionality"""
    try:
        # Create mock agent and LLM
        agent = MockAgent()
        llm = MockLLM()

        # Check token balances before
        strategy_data = json.loads(strategy_result) if isinstance(
            strategy_result, str) else strategy_result
        token_pair = strategy_data.get("token_pair", "").split('/')
        exit_strategy_tool = SiloExitStrategyToolPrivy(agent, llm)

        # Execute strategy exit
        result = exit_strategy_tool._run(
            strategy_result=strategy_result,
            sender=sender,
            swap_slippage=float(swap_slippage),
            id=int(id)
        )

        # Log the result
        logger.info(result)

        # Final status
        result_dict = json.loads(result) if isinstance(result, str) else result
        if "status" in result_dict and result_dict["status"] == "success":
            logger.info("✅ Exit strategy completed successfully!")
        else:
            logger.error(
                f"❌ Exit strategy failed! Error: {result_dict.get('error', 'Unknown error')}")

    except Exception as e:
        logger.error(f"Error during test: {str(e)}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Privy Silo exit strategy functionality")
    parser.add_argument("--sender", "-s", required=True,
                        help="Sender wallet address")
    parser.add_argument("--slippage", "-sl", default=0.5,
                        help="Slippage tolerance for swaps (default: 0.5)")
    parser.add_argument("--id", "-i", default=0,
                        help="Market ID to use for testing (default: 0)")

    args = parser.parse_args()

    # Read strategy result from file
    try:
        strategy_result = json.dumps({
            "status": "success",
            "type": "silo_loop_strategy",
            "deposit_token": {
                "original": "S",
                "silo": "wS"
            },
            "token_pair": "S/stS",
            "initial_amount": 1.0,
            "completed_loops": 2,
            "total_deposited": 2.08520592,
            "total_borrowed": 1.07356307,
            "leverage": "2.09x",
            "initial_deposit_tx": "0xededa2c3eb701880a534088b5a71b4168b61f8815e9cf0a759f1e99df1a7312e",
            "loop_details": [
                {
                    "loop": 1,
                    "max_borrow": 0.6903116,
                    "borrow_amount": 0.62128044,
                    "swap_amount_out": 0.62816092,
                    "deposit_amount": 0.62816092,
                    "borrow_tx": "0x22c1ac4cefca371e91f8749779f2d636eed985376e476614f752b0063c710b46",
                    "deposit_tx": "0xab07c552f12e4f629669efae9eefd423bd807d64c3564b3d5cc4ca9c29fd8fbf"
                },
                {
                    "loop": 2,
                    "max_borrow": 0.50253626,
                    "borrow_amount": 0.45228263,
                    "swap_amount_out": 0.457045,
                    "deposit_amount": 0.457045,
                    "borrow_tx": "0x69f81e86e379279752fc6cbdac66ddcae28b4be0ef441a44ec982f867791d57e",
                    "deposit_tx": "0xededa2c3eb701880a534088b5a71b4168b61f8815e9cf0a759f1e99df1a7312e"
                }
            ],
            "token_conversion": {
                "type": "wrap",
                "from_token": "S",
                "to_token": "wS",
                "amount": 1.0
            }
        })

        test_silo_exit_strategy(
            strategy_result=strategy_result,
            sender=args.sender,
            swap_slippage=args.slippage,
            id=args.id
        )
    except Exception as e:
        logger.error(f"Failed to read strategy file: {str(e)}")
