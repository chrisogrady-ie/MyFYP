
from keras.models import Sequential
from keras.layers import Dropout, Dense
import my_tools


# --------------------------------------------------
#
# --------------------------------------------------
def predict_ann(x_train, y_train, x_test, y_test,
                my_epochs, my_batch_size, dropout):
    model = Sequential()
    # RELU OR SIGMOID
    model.add(Dense(12, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dropout(dropout))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=1, activation='relu'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
    model.summary()
    model.fit(x_train, y_train, epochs=my_epochs, batch_size=my_batch_size)

    predictions = model.predict(x_test)

    mse, r2, mae, rmse, mape = my_tools.evaluate_model(y_test, predictions)
    model.save('./data/models/ANN/ANN_models/HMC/ann_hmc.model')

    return predictions
