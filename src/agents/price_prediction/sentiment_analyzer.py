from transformers import pipeline
import pandas as pd

class SentimentAnalyzer:
    def __init__(self):
        # Using FinBERT for financial sentiment analysis
        self.analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert"
        )
    
    def analyze_text(self, text):
        try:
            result = self.analyzer(text)[0]
            # Convert sentiment labels to numerical scores
            sentiment_map = {
                'positive': 1.0,
                'negative': -1.0,
                'neutral': 0.0
            }
            return sentiment_map[result['label']] * result['score']
        except:
            return 0.0
    
    def analyze_dataframe(self, df):
        df['sentiment_score'] = df['text'].apply(self.analyze_text)
        return df
