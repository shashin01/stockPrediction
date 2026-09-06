import joblib
import numpy as np
import pandas as pd
from dataLoader import load_data
from pathlib import Path
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent 
MODELS_PATH = BASE_DIR / "models"

def load_artifacts(ticker: str):
    ticker_path = MODELS_PATH / ticker
    model_path = ticker_path / "model.keras"
    scaler_path = ticker_path / "scaler.joblib"
    meta_path = ticker_path / "meta.joblib"

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    meta = joblib.load(meta_path)

    return model, scaler, meta

def forecast(ticker: str, days: int):
    model, scaler, meta = load_artifacts(ticker)
    seq_length = meta["seq_length"]

    data = load_data(ticker, "2024-01-01", "2026-09-01", "1d")
    print(data["Close"].values[-10:])
    close_vals = data["Close"].values[-seq_length:].reshape(-1,1)
    scaled_vals = scaler.transform(close_vals).flatten().tolist()

    window = scaled_vals.copy()
    predictions_scaled = []

    # The more days we predict for, the more inaccurate it gets 
    # because it is predicting next days based on its own prediction
    for _ in range(days):
        x = np.array(window[-seq_length:]).reshape(1, seq_length, 1)
        next = model.predict(x)[0, 0]
        predictions_scaled.append(next)
        window.append(next)

    predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()

    predicted_dates = pd.bdate_range(start=data.index[-1], periods=days+1)[1:]

    last_known_price = float(data["Close"].values[-1])
    print(f"Last known price on {predicted_dates[0].date()}: {last_known_price}")
    for date, price in zip(predicted_dates, predictions):
        print(f"{date.date()}: {price}")

if __name__ == "__main__":
    forecast("AAPL", 10)