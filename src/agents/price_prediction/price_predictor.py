import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

class PricePredictor:
    def __init__(self, config):
        self.config = config
        self.technical_model = xgb.XGBRegressor(**config['model_params']['xgboost'])
        self.sentiment_model = lgb.LGBMRegressor(**config['model_params']['lightgbm'])
        self.scaler = StandardScaler()
        
    def prepare_features(self, technical_df, sentiment_score):
        # Prepare technical features
        tech_features = technical_df[['close', 'BB_upper', 'BB_lower', 'volume', 'SMA_50', 'EMA_26', 'RSI', 'MACD']].copy()
        
        # Add sentiment features
        tech_features['sentiment'] = sentiment_score
        
        return tech_features
        
    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2)
        
        # Train technical model
        self.technical_model.fit(X_train, y_train)
        tech_score = self.technical_model.score(X_val, y_val)
        
        # Train sentiment model
        self.sentiment_model.fit(X_train, y_train)
        sent_score = self.sentiment_model.score(X_val, y_val)
        
        return (tech_score + sent_score) / 2
    
    def predict(self, X, timeframe='short'):
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from both models
        tech_pred = self.technical_model.predict(X_scaled)
        sent_pred = self.sentiment_model.predict(X_scaled)
        
        # Apply timeframe-specific weights
        weights = self.config['weights'][timeframe]
        final_pred = (tech_pred * weights['technical'] + 
                     sent_pred * weights['sentiment'])
        
        return float(final_pred[0])
