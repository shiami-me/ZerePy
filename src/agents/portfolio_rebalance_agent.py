from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from typing import Dict, Any, List
from ..utils.vector_store_utils import VectorStoreUtils
from ..utils.create_agent import Agent
from ..tools.portfolio_rebalance_tools import (
    analyze_portfolio,
    get_crypto_prices,
    get_historical_data,
    PortfolioAnalyzer
)
from ..tools.technical_analyzer import TechnicalAnalyzer

class State(MessagesState):
    next: str

class PortfolioRebalanceAgent:
    """Agent for handling portfolio analysis and rebalancing recommendations."""
    
    def __init__(self, llm, name: str, prompt: str, next: str):
        self._name = name
        self.portfolio_tools = [
            analyze_portfolio,
            get_crypto_prices,
            get_historical_data
        ]
        self.portfolio_agent = Agent(
            tools=self.portfolio_tools,
            vector_store=VectorStoreUtils(tools=self.portfolio_tools),
            llm=llm,
            prompt=prompt
        )._create_conversation_graph()
        self.next = next
        self.analyzer = PortfolioAnalyzer()

    def _format_analysis_results(self, analysis: Dict[str, Any]) -> str:
        """Format portfolio analysis results into a readable string."""
        return f"""Portfolio Analysis:
- Total Value: ${analysis['total_value']:,.2f}
- Current Allocation: {analysis['weights']}
- Risk Level: {analysis['portfolio_risk']:.2%}
- Sharpe Ratio: {analysis['sharpe_ratio']:.2f}
- Technical Indicators: {analysis.get('technical_analysis', {})}"""

    def node(self, state: State):
        """Process the current state and return the next command."""
        try:
            # Extract portfolio from state messages if available
            portfolio = self._extract_portfolio_from_messages(state.messages)
            
            # Perform portfolio analysis
            analysis_results = self.analyzer.analyze_portfolio(portfolio)
            
            # Format analysis results
            formatted_analysis = self._format_analysis_results(analysis_results)
            
            # Get agent's recommendation
            result = self.portfolio_agent.invoke({
                "messages": state.messages + [
                    HumanMessage(content=f"Based on this analysis: {formatted_analysis}\n"
                               "Please provide specific rebalancing recommendations.")
                ]
            })

            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=result["messages"][-1].content,
                            name=self._name
                        )
                    ]
                },
                goto="shiami"
            )

        except Exception as e:
            error_msg = f"Portfolio rebalancing agent error: {str(e)}"
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=error_msg,
                            name=self._name
                        )
                    ]
                },
                goto="shiami"
            )

    def _extract_portfolio_from_messages(self, messages: List[dict]) -> Dict[str, float]:
        """Extract portfolio information from messages."""
        # Default test portfolio if none found in messages
        default_portfolio = {
            'BTC': 1.0,
            'ETH': 10.0,
            'SOL': 50.0,
            'USDC': 1000.0
        }
        
        # TODO: Implement logic to extract portfolio information and format it.
        
        return default_portfolio