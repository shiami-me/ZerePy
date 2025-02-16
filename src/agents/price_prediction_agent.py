from typing import Type, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.types import Command
import pandas as pd
import yaml
import logging

# Import premade components

from price_prediction.technical_collector import TechnicalDataCollector
from price_prediction.sentiment_analyzer import SentimentAnalyzer
from price_prediction.price_predictor import PricePredictor
from price_prediction.reddit_scraper import RedditScraper
from price_prediction.twitter_data import TwitterScraper

import os
from typing import Dict, Any

@staticmethod
def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    try:
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Handle environment variables
        config['technical']['coingecko']['api_key'] = os.getenv('COINGECKO_API_KEY')
        if not config['technical']['coingecko']['api_key']:
            raise ValueError("Missing COINGECKO_API_KEY environment variable")
        
        # Validate required fields
        required_fields = [
            ('technical', 'coingecko', 'api_key', 'header'),
            ('model_params', 'xgboost'),
            ('model_params', 'lightgbm'),
            ('weights', 'short'),
            ('weights', 'long')
        ]
        
        for fields in required_fields:
            current = config
            for field in fields:
                if field not in current:
                    raise KeyError(f"Missing required field: {'.'.join(fields)}")
                current = current[field]
        
        return config
        
    except FileNotFoundError:
        raise FileNotFoundError("config.yaml not found in current directory")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config.yaml: {str(e)}")


class State(MessagesState):
    def __init__(self):
        super().__init__()
        self.data: Dict[str, Any] = {}
        self.next: str = "collect_data"

# Tool schemas

class TechnicalAnalysisInput(BaseModel):
    symbol: str = Field(description="Cryptocurrency symbol (e.g., 'btc', 'eth')")
    days: int = Field(description="Number of days of historical data", default=120)

class SentimentAnalysisInput(BaseModel):
    symbol: str = Field(description="Cryptocurrency symbol")
    timeframe: str = Field(description="Analysis timeframe ('short' or 'long')")

class PricePredictionInput(BaseModel):
    symbol: str = Field(description="Cryptocurrency symbol")
    timeframe: str = Field(description="Prediction timeframe ('short' or 'long')")

# Tools
class TechnicalAnalysisTool(BaseTool):
    name: str = "technical_analysis"
    description: str = "Collects technical analysis data for a cryptocurrency"
    args_schema: Type[BaseModel] = TechnicalAnalysisInput
    
    def __init__(self, config: Dict):
        super().__init__()
        self.collector = TechnicalDataCollector(config['technical'])
        
    def _run(self, symbol: str, days: int, 
             run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict:
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("Invalid symbol")
        if not isinstance(days, int) or days <= 0:
            raise ValueError("Days must be a positive integer")
        
        try:
            df = self.collector.fetch_data(symbol, days)
            if df is None or df.empty:
                return {"error": "Failed to fetch technical data"}
            
            latest_data = df.iloc[-1].to_dict()
            return {
                "price": latest_data["close"],
                "rsi": latest_data["RSI"],
                "macd": latest_data["MACD"],
                "bb_upper": latest_data["BB_upper"],
                "bb_lower": latest_data["BB_lower"],
                "dataframe": df
            }
        except Exception as e:
            return {"error": f"Technical analysis failed: {str(e)}"}

class SentimentAnalysisTool(BaseTool):
    name: str = "sentiment_analysis"
    description: str = "Analyzes sentiment from social media for a cryptocurrency"
    args_schema: Type[BaseModel] = SentimentAnalysisInput
    
    def __init__(self, config: Dict):
        super().__init__()
        self.reddit_scraper = RedditScraper(config)
        self.twitter_scraper = TwitterScraper(config["twitter"]["api_key"])
        self.sentiment_analyzer = SentimentAnalyzer()
        self.valid_timeframes = ['short', 'long']
    
    async def _arun(self, symbol: str, timeframe: str,
                    run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict:
        if timeframe not in self.valid_timeframes:
            raise ValueError(f"Invalid timeframe. Must be one of {self.valid_timeframes}")
            
        try:
            # Get Reddit data
            reddit_df = await self._get_reddit_data(symbol)
            reddit_sentiment = self.sentiment_analyzer.analyze_dataframe(reddit_df)
            
            # Get Twitter data
            twitter_sentiment = await self._get_twitter_data(symbol)
            
            # Combine sentiment scores
            return self._combine_sentiments(reddit_sentiment, twitter_sentiment)
            
        except Exception as e:
            return {"error": f"Sentiment analysis failed: {str(e)}"}
            
    async def _get_reddit_data(self, symbol: str) -> pd.DataFrame:
        try:
            return self.reddit_scraper.fetch_posts(symbol)
        except Exception as e:
            logging.error(f"Reddit scraping failed: {e}")
            return pd.DataFrame()
            
    async def _get_twitter_data(self, symbol: str) -> Dict:
        try:
            twitter_posts = await self.twitter_scraper.get_tweets(symbol)
            if twitter_posts:
                twitter_df = pd.DataFrame([post.dict() for post in twitter_posts.posts])
                return self.sentiment_analyzer.analyze_dataframe(twitter_df)
            return {"sentiment_score": 0}
        except Exception as e:
            logging.error(f"Twitter scraping failed: {e}")
            return {"sentiment_score": 0}

class PricePredictionTool(BaseTool):
    name: str = "price_prediction"
    description: str = "Predicts cryptocurrency price based on technical and sentiment analysis"
    args_schema: Type[BaseModel] = PricePredictionInput
    
    def __init__(self, config: Dict):
        super().__init__()
        self.predictor = PricePredictor(config)
        self.valid_timeframes = ['short', 'long']
    
    def _validate_data(self, technical_data: Dict, sentiment_data: Dict) -> None:
        required_technical = ['price', 'rsi', 'macd', 'dataframe']
        required_sentiment = ['overall_score']
        
        if not all(k in technical_data for k in required_technical):
            raise ValueError(f"Missing required technical fields: {required_technical}")
        if not all(k in sentiment_data for k in required_sentiment):
            raise ValueError(f"Missing required sentiment fields: {required_sentiment}")
    
    def _run(self, symbol: str, timeframe: str,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict:
        try:
            if timeframe not in self.valid_timeframes:
                raise ValueError(f"Invalid timeframe. Must be one of {self.valid_timeframes}")
            
            technical_data = self.get_technical_data(symbol)
            sentiment_data = self.get_sentiment_data(symbol)
            self._validate_data(technical_data, sentiment_data)
            
            features = self.predictor.prepare_features(
                technical_data["dataframe"],
                sentiment_data["overall_score"]
            )
            
            prediction = self.predictor.predict(features, timeframe)
            confidence = self.predictor.get_confidence_score(features, technical_data["dataframe"])
            
            return {
                "predicted_price": prediction,
                "confidence": confidence,
                "timeframe": timeframe,
                "current_price": technical_data["price"]
            }
        except Exception as e:
            return {"error": f"Price prediction failed: {str(e)}"}

class CryptoPredictionAgent:
    def __init__(self, llm, config: Dict):
        self.config = config
        self.tools = [
            TechnicalAnalysisTool(config),
            SentimentAnalysisTool(config),
            PricePredictionTool(config)
        ]
        
        self.agent = create_react_agent(
            llm,
            self.tools,
            """You are a cryptocurrency price prediction agent..."""
        )
        self.workflow = self.create_workflow().compile()
    
    def create_workflow(self) -> StateGraph:
        workflow = StateGraph(State)
        
        # Define nodes
        workflow.add_node("collect_data", self.collect_data)
        workflow.add_node("analyze_sentiment", self.analyze_sentiment)
        workflow.add_node("predict_price", self.predict_price)
        
        # Define edges
        workflow.add_edge("collect_data", "analyze_sentiment")
        workflow.add_edge("analyze_sentiment", "predict_price")
        
        # Set entry and exit
        workflow.set_entry_point("collect_data")
        workflow.set_exit_point("predict_price")
        
        return workflow
    
    def collect_data(self, state: State) -> Command:
        result = self.agent.invoke({
            "messages": state.messages,
            "tool": "technical_analysis"
        })
        
        state.data["technical"] = result["output"]
        return Command(
            update={"messages": [HumanMessage(content=result["messages"][-1].content)]},
            goto="analyze_sentiment"
        )
    
    async def analyze_sentiment(self, state: State) -> Command:
        result = await self.agent.ainvoke({
            "messages": state.messages,
            "tool": "sentiment_analysis"
        })
        
        state.data["sentiment"] = result["output"]
        return Command(
            update={"messages": [HumanMessage(content=result["messages"][-1].content)]},
            goto="predict_price"
        )
    
    def predict_price(self, state: State) -> Command:
        result = self.agent.invoke({
            "messages": state.messages,
            "tool": "price_prediction",
            "technical_data": state.data["technical"],
            "sentiment_data": state.data["sentiment"]
        })
        
        return Command(
            update={"messages": [HumanMessage(content=result["messages"][-1].content)]},
            goto="shiami"
        )
