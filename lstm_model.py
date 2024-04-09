import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from tensorflow_core.python.keras.optimizer_v2.adam import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import my_tools


# --------------------------------------------------
#
# --------------------------------------------------
def predict_lstm(x_train, y_train, x_test, y_test,
                 my_epochs, batch_size, dropout):
    my_layers = 2
    model = Sequential()
    model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(dropout))
    for i in range(0, my_layers):
        model.add(LSTM(units=96, return_sequences=True))
        model.add(Dropout(dropout))
    model.add(LSTM(units=96))
    # model.add(LSTM(my_units, activation='relu', input_shape=(x_train.shape[1], 1)))

    model.add(Dense(units=1))
    # opt = Adam(learning_rate=0.001)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
    model.summary()

    model.fit(x_train, y_train, epochs=my_epochs, batch_size=batch_size)
    predictions = model.predict(x_test)

    mse, r2, mae, rmse, mape = my_tools.evaluate_model(y_test, predictions)
    # model.save('./data/models/lstm.model')

    return predictions

