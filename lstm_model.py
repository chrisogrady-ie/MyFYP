import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from tensorflow_core.python.keras.optimizer_v2.adam import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error


# --------------------------------------------------
#
# --------------------------------------------------
def predict_lstm(x_train, y_train, x_test, y_test, my_epochs):
    unit = 96
    drop = 0.1
    model = Sequential()
    model.add(LSTM(units=unit, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(drop))
    model.add(LSTM(units=unit, return_sequences=True))
    model.add(Dropout(drop))
    model.add(LSTM(units=unit, return_sequences=True))
    model.add(Dropout(drop))
    model.add(LSTM(units=unit))
    # model.add(LSTM(my_units, activation='relu', input_shape=(x_train.shape[1], 1)))

    model.add(Dense(units=1))
    # opt = Adam(learning_rate=0.001)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=my_epochs)
    predictions = model.predict(x_test)

    model.summary()
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")

    print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}")
    # model.save('./data/models/lstm.model')

    return predictions
