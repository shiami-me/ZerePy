from pydantic import BaseModel, SecretStr
from typing import List
import pandas as pd
from browser_use import Agent, Controller
from langchain_google_genai import ChatGoogleGenerativeAI
import os

class Post(BaseModel):
    post_title: str
    post_url: str
    num_comments: int
    hours_since_post: int

class Posts(BaseModel):
    posts: List[Post]

class TwitterScraper:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.controller = Controller(output_model=Posts)
        self.model = ChatGoogleGenerativeAI(
            model='gemini-1.5-flash',
            api_key=os.getenv('GEMINI_API_KEY')
        )

    def _create_task(self, crypto_name: str) -> str:
        return f"""Follow these steps to extract information about {crypto_name} from x.com:
        1. Search for "#{crypto_name}" in the search bar
        2. Switch to "Latest" tweets tab
        3. For each of the top 10 most recent tweets:`
            a. Extract the username
            b. Extract the timestamp
            c. Extract the tweet content
            d. Extract the number of likes
            e. Extract the number of retweets
        4. Make sure to:
         
               - Only collect tweets that are actually about {crypto_name}
            - Ignore retweets
            - Ignore advertisements
            - Collect exactly 10 tweets if possible
            - Format the data according to the specified model"""

    async def get_tweets(self, symbol: str) -> Posts:
        initial_actions = [
            {'open_tab': {'url': 'https://x.com'}}
        ]
        
        task = self._create_task(symbol)
        agent = Agent(
            task=task,
            llm=self.model,
            controller=self.controller,
            use_vision=True
        )

        history = await agent.run(max_steps=25)
        
        if history.final_result():
            return Posts.model_validate_json(history.final_result())
        return None