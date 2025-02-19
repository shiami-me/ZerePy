from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import ta
from datetime import datetime, timedelta
from pydantic import Field, BaseModel
from langchain.tools import BaseTool
import requests
import os
import json

class CoinGeckoDataCollector(BaseTool):
    name: str = "crypto_data_collector"
    description: str = "Collects cryptocurrency data from CoinGecko API."
    api_key: Optional[str] = Field(default=None)
    base_url: str = Field(default="https://api.coingecko.com/api/v3")

    def _prepare_headers(self) -> Dict[str, str]:
        return {"x-cg-demo-api-key": self.api_key} if self.api_key else {}

    def get_historical_data(self, coin_id: str, days: int = 365) -> pd.DataFrame:
        try:
            response = requests.get(
                f"{self.base_url}/coins/{coin_id}/market_chart",
                params={"vs_currency": "usd", "days": str(days), "interval": "daily"},
                headers=self._prepare_headers()
            )
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['volume'] = [v[1] for v in data['total_volumes']]
            df['market_cap'] = [m[1] for m in data['market_caps']]
            df['high'] = df['close']  # CoinGecko provides daily close prices only
            df['low'] = df['close']
            df['open'] = df['close']
            
            df = df.drop('timestamp', axis=1)
            df.set_index('date', inplace=True)
            
            return df
        except Exception as e:
            raise Exception(f"Failed to fetch historical data: {str(e)}")

    def _run(self, coin_id: str, days: int = 365) -> pd.DataFrame:
        return self.get_historical_data(coin_id, days)

    async def _arun(self, coin_id: str) -> Dict[str, Any]:
        raise NotImplementedError("Async version not implemented")

class CryptoDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    @staticmethod
    def create_sequences(data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length, 0])
        return np.array(X), np.array(y)

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        hidden_dim = hidden_size * 2  # *2 for bidirectional
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :]), None

class CryptoPricePredictionTool(BaseTool):
    name: str = "crypto_price_predictor"
    description: str = "Predicts cryptocurrency prices using LSTM model."
    data_collector: CoinGeckoDataCollector = Field(...)
    sequence_length: int = Field(default=60)
    model: Optional[Any] = Field(default=None)
    scaler: Any = Field(default_factory=MinMaxScaler)
    device: Any = Field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def __init__(self, data_collector: CoinGeckoDataCollector, **kwargs):
        super().__init__(
            data_collector=data_collector,
            model=None,
            scaler=MinMaxScaler(),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            **kwargs
        )

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['SMA_7'] = ta.trend.sma_indicator(df['close'], window=7)
        df['SMA_30'] = ta.trend.sma_indicator(df['close'], window=30)
        df['RSI'] = ta.momentum.rsi(df['close'])
        df['MACD'] = ta.trend.macd_diff(df['close'])
        bb = ta.volatility.BollingerBands(df['close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['price_momentum'] = df['close'].pct_change()
        df['volume_momentum'] = df['volume'].pct_change()
        
        df.bfill(inplace=True)
        df.ffill(inplace=True)
        return df

    def train_model(self, coin_id: str, training_days: int = 365) -> Dict[str, Any]:
        try:
            df = self.data_collector.get_historical_data(coin_id, days=training_days)
            df = self.prepare_features(df)
            
            features = ['close', 'volume', 'SMA_7', 'SMA_30', 'RSI', 'MACD', 
                       'BB_upper', 'BB_lower', 'price_momentum', 'volume_momentum']
            
            data = self.scaler.fit_transform(df[features].values)
            X, y = CryptoDataset.create_sequences(data, self.sequence_length)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            train_loader = DataLoader(CryptoDataset(X_train, y_train), batch_size=32, shuffle=True)
            val_loader = DataLoader(CryptoDataset(X_val, y_val), batch_size=32, shuffle=False)

            self.model = LSTMModel(len(features)).to(self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            best_val_loss = float('inf')
            for epoch in range(50):
                self.model.train()
                train_loss = self._train_epoch(train_loader, optimizer, criterion)
                val_loss = self._validate_epoch(val_loader, criterion)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), f'best_model_{coin_id}.pth')
                
                if epoch % 5 == 0:
                    print(f"Epoch [{epoch+1}/50], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            self.model.load_state_dict(torch.load(f'best_model_{coin_id}.pth'))
            return {"status": "success", "validation_loss": best_val_loss}
            
        except Exception as e:
            return {"error": f"Training failed: {str(e)}"}

    def _train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            optimizer.zero_grad()
            outputs, _ = self.model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _validate_epoch(self, loader: DataLoader, criterion: nn.Module) -> float:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs, _ = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                total_loss += loss.item()
        return total_loss / len(loader)

    def predict_price(self, coin_id: str, days_to_predict: int = 7) -> Dict[str, Any]:
        try:
            if self.model is None:
                training_result = self.train_model(coin_id)
                if "error" in training_result:
                    return training_result

            df = self.data_collector.get_historical_data(coin_id, days=60)
            df = self.prepare_features(df)
            
            features = ['close', 'volume', 'SMA_7', 'SMA_30', 'RSI', 'MACD', 
                       'BB_upper', 'BB_lower', 'price_momentum', 'volume_momentum']
            
            recent_data = self.scaler.transform(df[features].values)
            
            predictions = []
            last_sequence = torch.FloatTensor(recent_data[-self.sequence_length:]).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                for _ in range(days_to_predict):
                    pred, _ = self.model(last_sequence.unsqueeze(0))
                    predictions.append(pred.item())
                    
                    new_row = last_sequence[-1].clone()
                    new_row[0] = pred.item()
                    last_sequence = torch.cat((last_sequence[1:], new_row.unsqueeze(0)))

            pred_prices = self.scaler.inverse_transform(
                np.hstack([np.array(predictions).reshape(-1, 1), 
                          np.zeros((len(predictions), len(features)-1))])
            )[:, 0]
            
            current_date = datetime.now()
            prediction_results = [
                {
                    "date": (current_date + timedelta(days=i+1)).strftime('%Y-%m-%d'),
                    "predicted_price": float(pred_prices[i]),
                    "confidence_score": 1.0 / (i + 1)
                }
                for i in range(days_to_predict)
            ]

            return {
                "status": "success",
                "coin_id": coin_id,
                "current_price": float(df['close'].iloc[-1]),
                "predictions": prediction_results
            }

        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

    def _run(self, coin_id: str, days_to_predict: int = 7) -> Dict[str, Any]:
        return self.predict_price(coin_id, days_to_predict)

    async def _arun(self, coin_id: str) -> Dict[str, Any]:
        raise NotImplementedError("Async version not implemented")

def test_prediction():
    try:
        # Initialize with proper field declarations
        data_collector = CoinGeckoDataCollector(api_key=os.getenv("COINGECKO_API_KEY"))
        predictor = CryptoPricePredictionTool(data_collector=data_collector)
        
        for coin in ["bitcoin", "ethereum"]:
            print(f"\nMaking predictions for {coin}...")
            result = predictor._run(coin, days_to_predict=7)
            print(f"\n{coin.capitalize()} Predictions:")
            print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction()