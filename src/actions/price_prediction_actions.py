import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
from bs4 import BeautifulSoup
from langchain.tools import BaseTool

class RateLimiter:
    def __init__(self, calls_per_minute):
        self.calls_per_minute = calls_per_minute
        self.calls = []
        
    def wait(self):
        now = time.time()
        minute_ago = now - 60
        
        # Remove calls older than 1 minute
        self.calls = [call for call in self.calls if call > minute_ago]
        
        if len(self.calls) >= self.calls_per_minute:
            # Wait until oldest call is more than 1 minute old
            sleep_time = 60 - (now - self.calls[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.calls.append(now)

class PricePredictionTools(BaseTool):
    def __init__(self):
        self.coingecko_limiter = RateLimiter(30)  # CoinGecko free tier: 30 calls/minute
        
    def get_historical_data(self, coin_id):
        """Fetch historical price data from CoinGecko"""
        self.coingecko_limiter.wait()
        
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": "365",
            "interval": "daily"
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.drop("timestamp", axis=1)
            return df
        else:
            raise RuntimeError(f"Failed to fetch data: {response.status_code}")

    def process_social_media_text(self, text):
        """Clean and process social media text"""
        soup = BeautifulSoup(text, "html.parser")
        clean_text = soup.get_text()
        # Remove URLs
        clean_text = ' '.join(word for word in clean_text.split() 
                            if not word.startswith(('http', 'https', 'www')))
        return clean_text

    def get_sentiment_score(self, text):
        """Get sentiment score using TextBlob"""
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def prepare_technical_features(self, df):
        """Prepare technical indicators for prediction"""
        df["MA7"] = df["price"].rolling(window=7).mean()
        df["MA30"] = df["price"].rolling(window=30).mean()
        df["RSI"] = self.calculate_rsi(df["price"])
        df["price_change"] = df["price"].pct_change()
        
        # Fill NaN values
        df = df.fillna(method="bfill")
        return df

    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def predict_price(self, coin_id, social_texts):
        """Main prediction function combining technical and sentiment analysis"""
        # Get historical data
        df = self.get_historical_data(coin_id)
        
        # Prepare technical features
        df = self.prepare_technical_features(df)
        
        # Process sentiment from social media
        sentiment_scores = []
        for text in social_texts:
            clean_text = self.process_social_media_text(text)
            sentiment_scores.append(self.get_sentiment_score(clean_text))
        
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        
        # Prepare data for ML model
        features = ["MA7", "MA30", "RSI", "price_change"]
        X = df[features].values
        y = df["price"].values
        
        # Scale the features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        train_size = int(len(X_scaled) * 0.8)
        
        X_train = X_scaled[:train_size]
        y_train = y[:train_size]
        
        model.fit(X_train, y_train)
        
        # Make prediction
        last_features = X_scaled[-1].reshape(1, -1)
        technical_prediction = model.predict(last_features)[0]
        
        # Combine technical and sentiment predictions
        sentiment_weight = 0.2
        technical_weight = 0.8
        
        # Adjust prediction based on sentiment
        sentiment_adjustment = avg_sentiment * 0.05  # 5% maximum adjustment
        final_prediction = (technical_prediction * technical_weight * (1 + sentiment_adjustment) + 
                          technical_prediction * sentiment_weight)
        
        return {
            "prediction": final_prediction,
            "technical_score": technical_prediction,
            "sentiment_score": avg_sentiment,
            "current_price": df["price"].iloc[-1]
        }