import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def load_data(ticker: str, start: str, end: str, interval: str):
    data = yf.download(ticker, start=start, end=end, interval=interval)

    if data is None or data.empty:
        return ValueError(f"No data has been found for {ticker}")

    data.columns = data.columns.get_level_values(0)

    data = data.dropna()
    return data

def create_sequences(values: np.ndarray, seq_length: int):
    """
    Stack sequences (X) with target value (y)
    """
    X = []
    y = []
    for i in range(len(values) - seq_length):
        X.append(values[i:i+seq_length])
        y.append(values[i+seq_length])
    return np.array(X), np.array(y)

def process_data(data: pd.DataFrame, seq_length: int, train_split: float = 0.8):
    closePrices = data["Close"].values.reshape(-1, 1)

    # Min/max normalization of data
    # Use all data (training and testing) to scale data 
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaledPrices = scaler.fit_transform(closePrices)

    # trainData is the historical data up to the train_split
    # testData is the rest but need to start with the first seq_length to make the first prediction
    train_idx = int(len(scaledPrices) * train_split)
    trainData = scaledPrices[:train_idx]
    testData = scaledPrices[train_idx-seq_length:]

    X_train, y_train = create_sequences(trainData, seq_length)
    X_test, y_test = create_sequences(testData, seq_length)

    return X_train, y_train, X_test, y_test, scaler

if __name__ == "__main__":
    myData = load_data("AAPL", "2024-01-01", "2026-01-01", "1d")
    processed = process_data(myData, 60)

    print(processed)