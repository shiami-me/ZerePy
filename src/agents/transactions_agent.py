from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph import MessagesState, END
from src.tools.privy_tools import PrivySendTransactionTool
from src.constants.networks import EVM_NETWORKS
from ..utils.vector_store_utils import VectorStoreUtils
from ..utils.create_agent import Agent
import json
import logging

logger = logging.getLogger("agents.transactions_agent")


class State(MessagesState):
    next: str


class TransactionsAgent:
    """Agent for handling blockchain transactions using Privy."""

    def __init__(self, llm, name: str, prompt: str, next: str, agent):
        self._name = name
        self._agent = agent
        self._next = next

        # Initialize tools for transaction management
        self.privy_tools = [PrivySendTransactionTool(agent=self._agent)]

        # Create the agent with the tools
        self.transactions_agent = Agent(
            tools=self.privy_tools,
            vector_store=VectorStoreUtils(tools=self.privy_tools),
            llm=llm,
            prompt=self._enhance_prompt(prompt)
        )._create_conversation_graph()

    def _enhance_prompt(self, prompt: str) -> str:
        """Enhance the prompt with network information."""
        # Add network information to the prompt
        network_info = "\n\nAvailable networks:\n"
        for network_name, network_data in EVM_NETWORKS.items():
            network_info += f"- {network_name}: Chain ID {network_data['chain_id']}\n"

        # Add additional transaction guidance
        transaction_guidance = """
When handling transactions, always follow these steps:
1. For contract interactions, ensure that the data field contains properly formatted calldata
2. For ETH transfers, ensure that the value is specified in wei (1 ETH = 10^18 wei)
3. Always verify transaction parameters before signing or sending
4. For transactions requiring a nonce, use the correct sequence number

Remember that transactions on different networks require different chain IDs. Use the correct
chain ID for the network you're interacting with.
"""
        return prompt + network_info + transaction_guidance

    def node(self, state: State):
        """Process the current state and return the next command."""
        try:
            result = self.transactions_agent.invoke(state)

            # Check if the output contains transaction data from tools
            response_content = result["messages"][-1].content

            # Log transaction details for monitoring
            if "transaction" in response_content.lower():
                logger.info(
                    f"Transaction processed: {response_content[:100]}...")

            # Return the result to the user
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=response_content,
                            name=self._name
                        )
                    ]
                },
                goto="shiami"

            )

        except Exception as e:
            error_msg = f"Transaction agent error: {str(e)}"
            logger.error(error_msg)
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=f"I encountered an issue while processing your transaction: {str(e)}. Please check the transaction details and try again.",
                            name=self._name
                        )
                    ]
                },
                goto="shiami",
            )

    def validate_transaction(self, tx_data: dict) -> tuple:
        """
        Validate transaction data before processing.

        Returns:
            tuple: (is_valid, error_message)
        """
        required_fields = ['to', 'value']

        # Check required fields
        for field in required_fields:
            if field not in tx_data:
                return False, f"Missing required field: {field}"

        # Validate address format
        if not tx_data['to'].startswith('0x') or len(tx_data['to']) != 42:
            return False, f"Invalid address format: {tx_data['to']}"

        # Validate value is numeric
        try:
            int(tx_data['value'])
        except (ValueError, TypeError):
            return False, f"Value must be a number: {tx_data['value']}"

        return True, ""
