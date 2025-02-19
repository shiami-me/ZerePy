from typing import Dict, Any, Optional, Tuple
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
from langchain.tools import BaseTool
from pydantic import Field
import requests
import os

class LiveCoinWatchDataCollector(BaseTool):
    name: str = "crypto_data_collector"
    description: str = "Collects cryptocurrency data from LiveCoinWatch API."
    api_key: Optional[str] = Field(default=None)
    base_url: str = Field(default="https://api.livecoinwatch.com")

    def _prepare_headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise ValueError("LiveCoinWatch API key is required")
        return {
            "x-api-key": self.api_key,
            "content-type": "application/json"
        }

    def get_historical_data(self, coin_code: str, days: int = 365) -> pd.DataFrame:
        try:
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = end_time - (days * 24 * 60 * 60 * 1000)  # Convert days to milliseconds

            payload = {
                "currency": "USD",
                "code": coin_code,
                "start": start_time,
                "end": end_time
            }

            response = requests.post(
                f"{self.base_url}/coins/single/history",
                headers=self._prepare_headers(),
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            # Convert history data to DataFrame
            df = pd.DataFrame(data['history'])
            df['date'] = pd.to_datetime(df['date'], unit='ms')
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'rate': 'close',
                'volume': 'volume',
                'cap': 'market_cap'
            })
            
            # Add required columns
            df['high'] = df['close']  # LiveCoinWatch provides only rate
            df['low'] = df['close']
            df['open'] = df['close']
            
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            return df
        except Exception as e:
            raise Exception(f"Failed to fetch historical data: {str(e)}")

    def _run(self, coin_code: str, days: int = 365) -> pd.DataFrame:
        return self.get_historical_data(coin_code, days)

    async def _arun(self, coin_code: str) -> Dict[str, Any]:
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
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        hidden_dim = hidden_size * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :]), None


class CryptoPricePredictionTool(BaseTool):
    name: str = "crypto_price_predictor"
    description: str = """Predicts cryptocurrency prices using historical data and machine learning models.
    Input should include:
    - coin_code: The cryptocurrency code (e.g., 'BTC', 'ETH', "S" etc)
    
    Example: Predict ETH prices
    Input: {"coin_code": "ETH"}
    
    This tool will return price predictions with confidence scores.
    """
    
    def __init__(self):
        super().__init__()
        self._data_collector = LiveCoinWatchDataCollector(
            api_key=os.getenv("LIVECOINWATCH_API_KEY")
        )
        self._sequence_length = 60
        self._model = None
        self._scalers = {}  # Initialize scalers dictionary
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def train_model(self, coin_code: str, training_days: int = 365) -> Dict[str, Any]:
        try:
            df = self._data_collector.get_historical_data(coin_code, days=training_days)
            df = self.prepare_features(df)
            
            features = ['close', 'volume', 'SMA_7', 'SMA_30', 'RSI', 'MACD', 
                       'BB_upper', 'BB_lower', 'price_momentum', 'volume_momentum']
            
            # Initialize scalers for each feature
            self._scalers = {feature: MinMaxScaler() for feature in features}
            scaled_data = np.column_stack([
                self._scalers[feat].fit_transform(df[feat].values.reshape(-1, 1))
                for feat in features
            ])
            
            X, y = CryptoDataset.create_sequences(scaled_data, self._sequence_length)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            train_loader = DataLoader(CryptoDataset(X_train, y_train), batch_size=32, shuffle=True)
            val_loader = DataLoader(CryptoDataset(X_val, y_val), batch_size=32, shuffle=False)

            self._model = LSTMModel(len(features)).to(self._device)
            optimizer = torch.optim.AdamW(self._model.parameters(), lr=0.001, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            criterion = nn.HuberLoss()
            
            best_val_loss = float('inf')
            patience = 15
            no_improve_count = 0
            
            for epoch in range(100):
                self._model.train()
                train_loss = self._train_epoch(train_loader, optimizer, criterion)
                val_loss = self._validate_epoch(val_loader, criterion)
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self._model.state_dict(), f'best_model_{coin_code}.pth')
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                if no_improve_count >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                    
                if epoch % 5 == 0:
                    print(f"Epoch [{epoch+1}/100], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            self._model.load_state_dict(torch.load(f'best_model_{coin_code}.pth'))
            return {"status": "success", "validation_loss": best_val_loss}
                
        except Exception as e:
            return {"error": f"Training failed: {str(e)}"}


    def _train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(self._device), batch_y.to(self._device)
            optimizer.zero_grad()
            outputs, _ = self._model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _validate_epoch(self, loader: DataLoader, criterion: nn.Module) -> float:
        self._model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self._device), batch_y.to(self._device)
                outputs, _ = self._model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                total_loss += loss.item()
        return total_loss / len(loader)

    def predict_price(self, coin_code: str, days_to_predict: int = 7) -> Dict[str, Any]:
        try:
            if self._model is None:
                training_result = self.train_model(coin_code)
                if "error" in training_result:
                    return training_result

            df = self._data_collector.get_historical_data(coin_code, days=60)
            df = self.prepare_features(df)
            
            features = ['close', 'volume', 'SMA_7', 'SMA_30', 'RSI', 'MACD', 
                       'BB_upper', 'BB_lower', 'price_momentum', 'volume_momentum']
            
            # Scale features using the trained scalers
            scaled_data = np.column_stack([
                self._scalers[feat].transform(df[feat].values.reshape(-1, 1))
                for feat in features
            ])
            
            initial_sequence = torch.FloatTensor(scaled_data[-self._sequence_length:]).to(self._device)
            
            self._model.eval()
            predictions = []
            current_sequence = initial_sequence.clone()
            
            with torch.no_grad():
                for _ in range(days_to_predict):
                    output, _ = self._model(current_sequence.unsqueeze(0))
                    pred = output.squeeze().item()
                    predictions.append(pred)
                    
                    new_row = current_sequence[-1].clone()
                    new_row[0] = pred
                    current_sequence = torch.cat((current_sequence[1:], new_row.unsqueeze(0)))

            pred_prices = self._scalers['close'].inverse_transform(
                np.array(predictions).reshape(-1, 1)
            ).flatten()
            
            current_date = datetime.now()
            prediction_results = [
                {
                    "date": (current_date + timedelta(days=i+1)).strftime('%Y-%m-%d'),
                    "predicted_price": float(pred_prices[i]),
                    "confidence_score": np.exp(-i * 0.2)
                }
                for i in range(days_to_predict)
            ]

            return {
                "status": "success",
                "coin_code": coin_code,
                "current_price": float(df['close'].iloc[-1]),
                "predictions": prediction_results
            }

        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}


    def _run(self, coin_code: str) -> Dict[str, Any]:
        return self.predict_price(coin_code, 1)
