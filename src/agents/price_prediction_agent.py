from typing import Dict, Any, List
import json
import asyncio
from datetime import datetime, timezone
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ..agent import ZerePyAgent
from ..actions.price_prediction_workflow import PricePredictionWorkflow

class PricePredictionAgent(ZerePyAgent, BaseTool):
    """Agent specialized in cryptocurrency price prediction"""
    
    name: str = "price_prediction_agent"
    description: str = "Predict cryptocurrency prices using technical and sentiment analysis"
    
    class Input(BaseModel):
        coin_ids: List[str] = Field(description="List of cryptocurrency IDs to analyze")
        detailed: bool = Field(default=False, description="Whether to return detailed analysis")
    
    def _run(self, coin_ids: List[str], detailed: bool = False) -> Dict[str, Any]:
        """Synchronous method for running the tool"""
        async def run_analysis():
            return await self.perform_price_analysis(coin_ids)
        
        return asyncio.run(run_analysis())
    
    async def _arun(self, coin_ids: List[str], detailed: bool = False) -> Dict[str, Any]:
        """Asynchronous method for running the tool"""
        return await self.perform_price_analysis(coin_ids)
    
    def __init__(self, agent_config_path: str):
        ZerePyAgent.__init__(self, agent_config_path)
        BaseTool.__init__(self)
        self.workflow = PricePredictionWorkflow()
        
    async def analyze_price(self, coin_id: str) -> Dict[str, Any]:
        """Analyze price for a given cryptocurrency"""
        try:
            result = await self.workflow.execute_workflow(coin_id)
            
            # Format the prediction results
            prediction = result["price_prediction"]
            current_price = prediction["current_price"]
            predicted_price = prediction["prediction"]
            percent_change = ((predicted_price - current_price) / current_price) * 100
            
            analysis = {
                "coin_id": coin_id,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "percent_change": percent_change,
                "sentiment_score": prediction["sentiment_score"],
                "confidence": self._calculate_confidence(result),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "analysis_details": result["workflow_analysis"]
            }
            
            return analysis
        except Exception as e:
            self.logger.error(f"Error analyzing price for {coin_id}: {str(e)}")
            raise

    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate confidence score based on technical and sentiment indicators"""
        try:
            # Extract relevant scores
            sentiment_score = abs(result["price_prediction"]["sentiment_score"])
            price_volatility = self._calculate_volatility(result)
            
            # Combine scores with weights
            confidence = (
                0.6 * (1 - price_volatility) +  # Lower volatility = higher confidence
                0.4 * sentiment_score           # Stronger sentiment = higher confidence
            )
            
            return min(max(confidence, 0.0), 1.0)  # Ensure between 0 and 1
        except Exception as e:
            self.logger.warning(f"Error calculating confidence: {str(e)}")
            return 0.5  # Default confidence

    def _calculate_volatility(self, result: Dict) -> float:
        """Calculate price volatility from historical data"""
        try:
            technical_data = result.get("workflow_analysis", {}).get("technical_data", {})
            if isinstance(technical_data, str):
                technical_data = json.loads(technical_data)
            
            prices = technical_data.get("price", [])
            if not prices:
                return 0.5
            
            # Calculate standard deviation of price changes
            price_changes = [
                (prices[i] - prices[i-1]) / prices[i-1]
                for i in range(1, len(prices))
            ]
            
            volatility = abs(sum(price_changes) / len(price_changes))
            return min(max(volatility, 0.0), 1.0)  # Normalize between 0 and 1
        except Exception as e:
            self.logger.warning(f"Error calculating volatility: {str(e)}")
            return 0.5  # Default volatility

    async def perform_price_analysis(self, coin_ids: List[str]) -> List[Dict]:
        """Analyze multiple cryptocurrencies concurrently"""
        tasks = [self.analyze_price(coin_id) for coin_id in coin_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        analyses = []
        for coin_id, result in zip(coin_ids, results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to analyze {coin_id}: {str(result)}")
                continue
            analyses.append(result)
        
        return analyses

    async def run_analysis_loop(self):
        """Main analysis loop"""
        while True:
            try:
                # Get configured coins from agent config
                coins = self.config.get("coins", ["bitcoin", "ethereum"])
                
                # Perform analysis
                analyses = await self.perform_price_analysis(coins)
                
                # Log results
                for analysis in analyses:
                    self.logger.info(
                        f"Price prediction for {analysis['coin_id']}: "
                        f"Current: ${analysis['current_price']:.2f}, "
                        f"Predicted: ${analysis['predicted_price']:.2f} "
                        f"({analysis['percent_change']:+.2f}%), "
                        f"Confidence: {analysis['confidence']:.2f}"
                    )
                
                # Store results if needed
                # TODO: Implement result storage
                
                # Wait for next analysis cycle
                await asyncio.sleep(self.loop_delay)
                
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying