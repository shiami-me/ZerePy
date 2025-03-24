
import sys
import os
import argparse
import logging
import json
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.actions.sonic_actions import get_token_by_ticker, get_sonic_balance, send_sonic, swap_sonic, get_swap_summary
from src.connections.pendle_connection import PendleConnection
from src.connections.sonic_connection import SonicConnection
from src.connections.silo_connection import SiloConnection
from src.connections.privy_connection import PrivyConnection
from src.tools.privy.silo_tools import SiloExitStrategyToolPrivy
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
        self.connections["pendle"] = PendleConnection({})


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
                "silo": "PT-wstkscUSD (29 May)",
                "is_pt_token": True,
                "pt_market": "wstkscUSD"
            },
            "token_pair": "PT-wstkscUSD (29 May)/frxUSD",
            "initial_amount": 1.0,
            "completed_loops": 2,
            "total_deposited": 1.967248,
            "total_borrowed": 0.94995864,
            "leverage": "1.97x",
            "initial_deposit_tx": "0x24dc8d7c47c97cec4ba14e01af1ef748747d1ff567df8d3f89f6b3cfcbfcaea3",
            "loop_details": [
                {
                    "loop": 1,
                    "max_borrow": 0.56611102,
                    "borrow_amount": 0.50949992,
                    "swap_amount_out": 0.518777,
                    "deposit_amount": 0.518777,
                    "borrow_tx": "0x2eb04183ef8022fc5d6c407f314785916c5028740a209fad8c9ced5ec4cded11",
                    "deposit_tx": "0xc259c88e57f88e54f3a6928b3177444cb8e1444442f025c8873db1cfc992af55"
                },
                {
                    "loop": 2,
                    "max_borrow": 0.48939858,
                    "borrow_amount": 0.44045872,
                    "swap_amount_out": 0.448471,
                    "deposit_amount": 0.448471,
                    "borrow_tx": "0x736d60eed5861b6475e232a6b43c9d26a08019a2264407317a6cebe447a0febe",
                    "deposit_tx": "0x24dc8d7c47c97cec4ba14e01af1ef748747d1ff567df8d3f89f6b3cfcbfcaea3"
                }
            ],
            "token_conversion": {
                "type": "swap",
                "method": "pendle",
                "from_token": "S",
                "to_token": "PT-wstkscUSD (29 May)",
                "original_amount": 1.0,
                "converted_amount": 0.610753
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
