
import sys
import os
import argparse
import logging
import json
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.actions.sonic_actions import get_token_by_ticker, get_sonic_balance, send_sonic, swap_sonic, get_swap_summary
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


class MockLLM:
    """Mock LLM for testing"""
    pass


def test_silo_exit_strategy(strategy_result, sender, swap_slippage=0.5):
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
            swap_slippage=float(swap_slippage)
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

    args = parser.parse_args()

    # Read strategy result from file
    try:
        strategy_result = json.dumps({
            "status": "success",
            "type": "silo_loop_strategy",
            "token_pair": "USDC/S",
            "initial_amount": 1.0,
            "completed_loops": 3,
            "total_deposited": 2.625995,
            "total_borrowed": 3.04498311,
            "leverage": "2.63x",
            "initial_deposit_tx": "0xaff2fe86af991377730ba0794fff15324a3a221f59ba32ab1acd5568d894513d",
            "loop_details": [
                {
                    "loop": 1,
                    "max_borrow": 1.41486825,
                    "borrow_amount": 1.27338142,
                    "swap_amount_out": 0.683634,
                    "deposit_amount": 0.683634,
                    "borrow_tx": "0x7f599ced18a21912e1f60e6ddd8a018ebbcfbf191c1fd440ec142ff80770aac1",
                    "deposit_tx": "0x6251fe2a520c0267aef49f1a8575440f4f054e170a1ecd7512a06e538769c297"
                },
                {
                    "loop": 2,
                    "max_borrow": 1.10874116,
                    "borrow_amount": 0.99786705,
                    "swap_amount_out": 0.529258,
                    "deposit_amount": 0.529258,
                    "borrow_tx": "0xcd890b8947616e175ed8cc6b89a57577ac0cd7d9fb26d9a7c262ab3e5729475d",
                    "deposit_tx": "0xe2c160b4cfe356be92371ad0848a779aa107c863ad70b1e019da4734b1cd0fa9"
                },
                {
                    "loop": 3,
                    "max_borrow": 0.85970514,
                    "borrow_amount": 0.77373463,
                    "swap_amount_out": 0.413103,
                    "deposit_amount": 0.413103,
                    "borrow_tx": "0x9df8307e9614efece1684e061d22dedfc6a9c759787368e48d97f4a8cefea65f",
                    "deposit_tx": "0xaff2fe86af991377730ba0794fff15324a3a221f59ba32ab1acd5568d894513d"
                }
            ]
        }




        )

        test_silo_exit_strategy(
            strategy_result=strategy_result,
            sender=args.sender,
            swap_slippage=args.slippage
        )
    except Exception as e:
        logger.error(f"Failed to read strategy file: {str(e)}")
