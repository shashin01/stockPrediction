import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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