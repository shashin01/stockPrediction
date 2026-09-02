import dataLoader as dl
from model import create_model

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

    prediction = model.predict(X_test)
    print(prediction)

if __name__== "__main__":
    train("AAPL", "2024-01-01", "2026-01-01", "1d", 60, 1, 10)