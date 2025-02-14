import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import logging
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SocialMediaCollector(BaseTool):
    name: str = "social_media_collector"
    description: str = "Collect social media posts about cryptocurrencies from Twitter and Reddit"
    
    class Input(BaseModel):
        query: str = Field(description="Cryptocurrency or topic to search")
        max_posts: int = Field(default=10, description="Maximum number of posts to retrieve")
        subreddits: Optional[List[str]] = Field(default=None, description="Optional list of subreddits to search")

    def _run(self, query: str, max_posts: int = 10, subreddits: Optional[List[str]] = None) -> Dict:
        """Synchronous method for running the tool"""
        async def collect_data():
            twitter_posts = await self.get_twitter_posts(query, max_posts)
            reddit_posts = await self.get_reddit_posts(subreddits or [], query, max_posts) if subreddits else []
            return {
                'twitter_posts': twitter_posts,
                'reddit_posts': reddit_posts
            }
        
        return asyncio.run(collect_data())

    async def _arun(self, query: str, max_posts: int = 10, subreddits: Optional[List[str]] = None) -> Dict:
        """Asynchronous method for running the tool"""
        twitter_posts = await self.get_twitter_posts(query, max_posts)
        reddit_posts = await self.get_reddit_posts(subreddits or [], query, max_posts) if subreddits else []
        return {
            'twitter_posts': twitter_posts,
            'reddit_posts': reddit_posts
        }

    def __init__(self):
        super().__init__()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        try:
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    return await response.text()
                logger.warning(f"Failed to fetch {url}: Status {response.status}")
                return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    async def get_twitter_posts(self, query: str, max_posts: int = 10) -> List[Dict]:
        """Fetch recent Twitter posts about a cryptocurrency using web scraping"""
        url = f"https://nitter.net/search?f=tweets&q={query}"
        posts = []
        
        async with aiohttp.ClientSession() as session:
            html = await self.fetch_url(session, url)
            if not html:
                return posts

            soup = BeautifulSoup(html, 'html.parser')
            tweet_containers = soup.find_all('div', class_='tweet-content')
            
            for tweet in tweet_containers[:max_posts]:
                if tweet.text:
                    posts.append({
                        'source': 'twitter',
                        'content': tweet.text.strip(),
                        'url': url
                    })
        
        return posts

    async def get_reddit_posts(self, subreddits: List[str], query: str, max_posts: int = 10) -> List[Dict]:
        """Fetch recent Reddit posts about a cryptocurrency"""
        posts = []
        tasks = []
        
        async with aiohttp.ClientSession() as session:
            for subreddit in subreddits:
                url = f"https://old.reddit.com/r/{subreddit}/search.json?q={query}&restrict_sr=on&sort=new"
                tasks.append(self.fetch_url(session, url))
            
            responses = await asyncio.gather(*tasks)
            
            for response in responses:
                if response:
                    try:
                        data = response.json()
                        for post in data.get('data', {}).get('children', [])[:max_posts]:
                            post_data = post.get('data', {})
                            if post_data.get('selftext') or post_data.get('title'):
                                posts.append({
                                    'source': 'reddit',
                                    'content': f"{post_data.get('title', '')} {post_data.get('selftext', '')}",
                                    'url': f"https://reddit.com{post_data.get('permalink', '')}"
                                })
                    except Exception as e:
                        logger.error(f"Error parsing Reddit response: {str(e)}")
                        continue
        
        return posts[:max_posts]

    async def collect_social_data(self, coin_id: str) -> List[Dict]:
        """Collect social media data from multiple sources"""
        # Define relevant subreddits for crypto
        crypto_subreddits = ['cryptocurrency', 'cryptomarkets', f'{coin_id}', 'cryptotrading']
        
        # Collect data from both platforms concurrently
        twitter_task = self.get_twitter_posts(f"#{coin_id} crypto")
        reddit_task = self.get_reddit_posts(crypto_subreddits, coin_id)
        
        twitter_posts, reddit_posts = await asyncio.gather(twitter_task, reddit_task)
        
        # Combine and return all social media posts
        return twitter_posts + reddit_posts