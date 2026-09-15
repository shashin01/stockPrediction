import joblib
import numpy as np
from pathlib import Path

from dataLoader import load_data, create_sequences
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
MODELS_PATH = BASE_DIR / "models"

def evaluate(ticker: str, start: str, end: str, seq_length: int):
    ticker_path = MODELS_PATH / ticker
    model_path = ticker_path / "model.keras"
    scaler_path = ticker_path / "scaler.joblib"
    meta_path = ticker_path / "meta.joblib"

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    meta = joblib.load(meta_path)

    # We will always evaluate day-by-day for now since this is widest array to use
    data = load_data(ticker, start, end, "1d")
    close_prices = data["Close"].values.reshape(-1, 1)
    scaled_prices = scaler.transform(close_prices)

    X, y = create_sequences(scaled_prices, seq_length)

    prediction_scaled = model.predict(X)
    predictions = scaler.inverse_transform(prediction_scaled).flatten()
    targets = scaler.inverse_transform(y.reshape(-1, 1)).flatten()

    rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    mape = float(np.mean(np.abs(targets - predictions) / targets)*100)

    findings = f"LSTM, {ticker}, evaluated from {start} to {end}, RMSE = {rmse:.4f}, MAPE = {mape: .2f}% \nTraining Configurations: (seq_length: {meta['seq_length']}, start: {meta['start']}, end: {meta['end']}, interval: {meta['interval']}, epochs: {meta['epochs']}, batch_size: {meta['batch_size']})\n\n_____________________\n\n"
    findings_path = BASE_DIR / "findings.txt"

    with open(findings_path, "a") as f:
        f.write(findings)

    print(f"Evaluation of LSTM model for {ticker} is complete.")
    print(f"Evaluated RMSE: {rmse:.4f}, MAPE: {mape: .2f}%")

if __name__ == "__main__":
    ticker = input("Ticker being tested: ")
    start = input("Start Date (YYYY-MM-DD): ")
    end = input("End Date (YYYY-MM-DD): ")
    seq_length = input("Number of past days model looks at to predict (e.g. 60): ")
    evaluate(ticker, start, end, int(seq_length))
