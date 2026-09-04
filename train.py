import dataLoader as dl
from model import create_model
import numpy as np
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent 
MODELS_PATH = BASE_DIR / "models"

LSTM_UNITS = 50
LSTM_DROPOUT = 0.2

def train(ticker: str, start: str, end: str, interval: str, seq_length: int, epochs: int, batch_size: int):
    data = dl.load_data(ticker, start, end, interval)
    print(f"Retrieved {len(data)} data points from {data.index[0].date()} to {data.index[-1].date()}")

    X_train, y_train, X_test, y_test, scaler = dl.process_data(data, seq_length, 0.8)

    model = create_model(seq_length, LSTM_UNITS, LSTM_DROPOUT)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    predictions_scaled = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1,1))
    error_rmse = float(np.sqrt(np.mean((predictions - actual)**2)))
    print(f"RMSE: {error_rmse}")

    ticker_path = MODELS_PATH / ticker
    ticker_path.mkdir(exist_ok=True)

    model_path = ticker_path / "model.keras"
    scaler_path = ticker_path / "scaler.joblib"
    meta_path = ticker_path / "meta.joblib"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump({"seq_length": seq_length, "start": start, "end": end, "interval": interval}, meta_path)

if __name__== "__main__":
    train("AAPL", "2024-01-01", "2026-01-01", "1d", 60, 1, 10)