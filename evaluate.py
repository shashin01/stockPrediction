import joblib
import numpy as np
from pathlib import Path

from dataLoader import load_data, process_data
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

def evaluate(ticker: str):
    model, scaler, meta = load_artifacts(ticker)
    seq_length = meta["seq_length"]
    start = meta["start"]
    end = meta["end"]
    interval = meta["interval"]

    data = load_data(ticker, start, end, interval)
    _, _, X_test, y_test, scaler = process_data(data, seq_length)

    prediction_scaled = model.predict(X_test)
    predictions = scaler.inverse_transform(prediction_scaled).flatten()
    actuals = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    naive_prediction = scaler.inverse_transform(X_test[:, -1, :]).flatten()

    rmse = float(np.sqrt(np.mean((predictions - actuals) ** 2)))
    naive_rmse = float(np.sqrt(np.mean((naive_prediction - actuals) ** 2)))
    print(f"Evaluated RMSE: {naive_rmse}, {rmse}")

if __name__ == "__main__":
    evaluate("GOOG")