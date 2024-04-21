
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.layers.recurrent import SimpleRNN
import my_tools


# --------------------------------------------------
#
# --------------------------------------------------
def predict_rnn(x_train, y_train, x_test, y_test,
                my_epochs, my_batch_size, dropout):
    model = Sequential()
    model.add(SimpleRNN(units=64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(dropout))
    model.add(SimpleRNN(units=64))
    model.add(Dropout(dropout))
    model.add(Dense(units=1, activation='relu'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
    model.summary()

    model.fit(x_train, y_train, epochs=my_epochs, batch_size=my_batch_size)

    predictions = model.predict(x_test)

    mse, r2, mae, rmse, mape = my_tools.evaluate_model(y_test, predictions)
    model.save('./data/models/RNN/RNN_models/HMC/rnn_hmc.model')

    return predictions

