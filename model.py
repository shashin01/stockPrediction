from tensorflow import keras

def create_model(seq_length: int, units: int, dropout: float):
    model = keras.Sequential()

    model.add(keras.layers.Input(shape=(seq_length, 1)))
    model.add(keras.layers.LSTM(units, return_sequences=True))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.LSTM(units, return_sequences=False))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(25, activation="relu"))
    model.add(keras.layers.Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.MeanSquaredError
    )

    return model

if __name__ == "__main__":
    model = create_model(50, 4, 0.2)
    print(model.summary())
