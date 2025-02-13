from typing import Dict, List
import json
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.base import RunnableSerializable
from langchain import LangChain
from langchain.graphs import Graph
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .price_prediction_actions import PricePredictionTools
from .social_media_collector import SocialMediaCollector

class PricePredictionWorkflow:
    def __init__(self):
        self.tools = PricePredictionTools()
        self.social_collector = SocialMediaCollector()
        self.setup_workflow()

    def setup_workflow(self):
        # Define the workflow nodes
        data_collection = (
            ChatPromptTemplate.from_template(
                "Collect historical price data and social media sentiment for {coin_id}. Format the request properly."
            )
            | StrOutputParser()
        )

        technical_analysis = (
            ChatPromptTemplate.from_template(
                "Analyze the technical indicators and patterns in the historical data:\n{technical_data}"
            )
            | StrOutputParser()
        )

        sentiment_analysis = (
            ChatPromptTemplate.from_template(
                "Analyze the social media sentiment from recent posts:\n{sentiment_data}"
            )
            | StrOutputParser()
        )

        price_prediction = (
            ChatPromptTemplate.from_template(
                "Based on technical analysis {technical_analysis} and sentiment analysis {sentiment_analysis}, "
                "predict the price movement for {coin_id}. Consider both short-term and medium-term outlook."
            )
            | StrOutputParser()
        )

        # Create the workflow graph
        self.workflow = Graph()
        self.workflow.add_node("data_collection", data_collection)
        self.workflow.add_node("technical_analysis", technical_analysis)
        self.workflow.add_node("sentiment_analysis", sentiment_analysis)
        self.workflow.add_node("price_prediction", price_prediction)

        # Define edges
        self.workflow.add_edge("data_collection", "technical_analysis")
        self.workflow.add_edge("data_collection", "sentiment_analysis")
        self.workflow.add_edge("technical_analysis", "price_prediction")
        self.workflow.add_edge("sentiment_analysis", "price_prediction")

    async def execute_workflow(self, coin_id: str) -> Dict:
        """Execute the price prediction workflow"""
        # Get historical data and social media data concurrently
        historical_data = self.tools.get_historical_data(coin_id)
        social_posts = await self.social_collector.collect_social_data(coin_id)
        
        # Process data through the workflow
        workflow_input = {
            "coin_id": coin_id,
            "technical_data": json.dumps(historical_data.to_dict()),
            "sentiment_data": json.dumps(social_posts)
        }

        result = await self.workflow.aexecute(workflow_input)
        
        # Get final prediction using the tools
        prediction = self.tools.predict_price(
            coin_id=coin_id,
            social_texts=result.get("sentiment_data", [])
        )
        
        return {
            "workflow_analysis": result,
            "price_prediction": prediction
        }

    def get_chain(self) -> RunnableSerializable:
        """Get the workflow chain for use in the agent framework"""
        return self.workflow