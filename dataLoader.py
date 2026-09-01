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

if __name__=='__main__':
    print(load_data("AAPL", "2024-01-01", "2026-01-01", "1d"))