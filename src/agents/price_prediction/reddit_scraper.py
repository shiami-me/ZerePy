import praw
import pandas as pd
from datetime import datetime
import os
from typing import Dict

class RedditScraper:
    def __init__(self, config):
        # Extract subreddits from config, with fallback to default
        self.config = config
        
        # Try multiple possible config paths for subreddits
        subreddit_paths = [
            ['sentiment', 'reddit', 'subreddits'],
            ['reddit', 'subreddits']
        ]
        
        # Default subreddits if no config is found
        default_subreddits = ['CryptoCurrency', 'Bitcoin', 'Ethereum']
        
        # Find the first valid subreddit list
        self.subreddits = default_subreddits
        for path in subreddit_paths:
            try:
                current_dict = config
                for key in path:
                    current_dict = current_dict[key]
                self.subreddits = current_dict
                break
            except (KeyError, TypeError):
                continue
        
        # Extract max posts, with fallback to default
        max_posts_paths = [
            ['sentiment', 'reddit', 'max_posts'],
            ['reddit', 'max_posts']
        ]
        
        self.max_posts = 50  # Default max posts
        for path in max_posts_paths:
            try:
                current_dict = config
                for key in path:
                    current_dict = current_dict[key]
                self.max_posts = current_dict
                break
            except (KeyError, TypeError):
                continue
        
        # Initialize Reddit client
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent='Crypto Sentiment Analysis Bot'
        )

    def fetch_posts(self, coin: str) -> pd.DataFrame:
        """
        Fetch latest posts about a specific cryptocurrency from configured subreddits
        
        Args:
            coin (str): name of the cryptocurrency (e.g., 'bitcoin', 'Ethereum')
        
        Returns:
            pd.DataFrame: DataFrame containing post data
        """
        all_posts = []
        search_terms = coin
        
        for subreddit_name in self.subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search for posts containing the coin name/symbol
                for search_term in search_terms:
                    posts = subreddit.search(
                        query=search_term,
                        sort='new',
                        time_filter='week',
                        limit=10
                    )
                    
                    for post in posts:
                        post_data = {
                            'subreddit': subreddit_name,
                            'title': post.title,
                            'text': post.selftext,
                            'score': post.score,
                            'num_comments': post.num_comments,
                            'created_utc': datetime.fromtimestamp(post.created_utc),
                            'upvote_ratio': post.upvote_ratio,
                            'url': post.url,
                            'author': str(post.author),
                            'is_original_content': post.is_original_content,
                            'search_term': search_term
                        }
                        all_posts.append(post_data)
                        
            except Exception as e:
                print(f"Error fetching posts from r/{subreddit_name}: {str(e)}")
                continue

        if not all_posts:
            print(f"No posts found for {coin}")
            return pd.DataFrame()

        # Convert to DataFrame and sort by date
        df = pd.DataFrame(all_posts)
        df = df.sort_values('created_utc', ascending=False)
        
        # Remove duplicates based on title
        df = df.drop_duplicates(subset=['title'])
        
        # Get the 10 most recent posts
        df = df.head(10)
        
        return df
    
    def analyze_sentiment(self, df: pd.DataFrame) -> Dict:
        """
        Calculate basic sentiment metrics for the posts
        
        Args:
            df (pd.DataFrame): DataFrame containing post data
        
        Returns:
            Dict: Dictionary containing sentiment metrics
        """
        if df.empty:
            return {
                'total_posts': 0,
                'avg_score': 0,
                'avg_comments': 0,
                'avg_upvote_ratio': 0
            }

        sentiment_metrics = {
            'total_posts': len(df),
            'avg_score': df['score'].mean(),
            'avg_comments': df['num_comments'].mean(),
            'avg_upvote_ratio': df['upvote_ratio'].mean()
        }

        return sentiment_metrics
