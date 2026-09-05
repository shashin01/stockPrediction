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

    data = load_data(ticker, "2025-01-01", "2026-01-01", "1d")
    close_vals = data["Close"].values[-seq_length:].reshape(-1,1)
    scaled_vals = scaler.transform(close_vals).flatten().tolist()

    window = scaled_vals.copy()
    predictions_scaled = []

    for _ in range(days):
        x = np.array(window[-seq_length:]).reshape(1, seq_length, 1)
        next = model.predict(x)[0, 0]
        predictions_scaled.append(next)
        window.append(next)

    predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()

    print(predictions)

if __name__ == "__main__":
    forecast("AAPL", 10)

