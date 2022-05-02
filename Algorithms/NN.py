import tensorflow as tf
import matplotlib.pyplot as plt


def modelNN(X_train, y_train, X_test, y_test):
    model = tf.keras.Sequential([
                                tf.keras.layers.Dense(128, activation="relu"),
                                tf.keras.layers.Dropout(0.1),
                                tf.keras.layers.Dense(64, activation="relu"),
                                tf.keras.layers.Dense(64, activation="relu"),
                                tf.keras.layers.Dense(1)
                                ])

    rmse = tf.keras.metrics.RootMeanSquaredError()

    # Compiling our Model
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=[rmse])

    history = model.fit(x=X_train, y=y_train, epochs=40,
                        validation_data=(X_test, y_test))

    plt.figure(figsize=(13, 7))
    plt.plot(history.history['root_mean_squared_error'],
             label='root_mean_squared_error')
    plt.plot(history.history['val_root_mean_squared_error'],
             label='val_root_mean_squared_error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
