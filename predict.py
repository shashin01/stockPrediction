import joblib
import numpy as np
import pandas as pd
from dataLoader import load_data
from pathlib import Path
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

BASE_DIR = Path(__file__).resolve().parent
MODELS_PATH = BASE_DIR / "models"

def forecast(ticker: str, days: int):
    ticker_path = MODELS_PATH / ticker
    model_path = ticker_path / "model.keras"
    scaler_path = ticker_path / "scaler.joblib"
    meta_path = ticker_path / "meta.joblib"

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    meta = joblib.load(meta_path)

    seq_length = meta["seq_length"]

    today_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=seq_length * 2)).strftime('%Y-%m-%d')

    data = load_data(ticker, start_date, today_date, "1d")
    close_vals = data["Close"].values[-seq_length-1:].reshape(-1, 1)
    last_known_price = close_vals[-1][0]

    pct_changes = []
    for i in range(len(close_vals) - 1):
        pct_changes.append((close_vals[i + 1] - close_vals[i])/close_vals[i])

    pct_changes_scaled = scaler.transform(pct_changes).flatten().tolist()
    window = pct_changes_scaled.copy()

    predicted_prices = [close_vals[-1][0]]

    # The more days we predict for, the more inaccurate it gets
    # because it is predicting next days based on its own prediction
    for _ in range(days):
        x = np.array(window[-seq_length:]).reshape(1, seq_length, 1)
        next_day_pct_scaled = model.predict(x)[0, 0].item()
        next_day_pct = scaler.inverse_transform([[next_day_pct_scaled]])[0, 0].item()
        next_day_price = predicted_prices[-1] * (1 + next_day_pct)
        predicted_prices.append(next_day_price)
        window.append(next_day_pct_scaled)

    predicted_dates = pd.bdate_range(start=data.index[-1], periods=days+1)[1:]

    print(f"Last known price on {predicted_dates[0].date()}: {last_known_price}")
    for date, price in zip(predicted_dates, predicted_prices):
        print(f"{date.date()}: {price}")

if __name__ == "__main__":
    ticker = input("Ticker: ").upper()
    days = input("Number of days to predict (more days increases inaccuracy): ")
    forecast(ticker, int(days))
