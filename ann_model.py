import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from tensorflow_core.python.keras.optimizer_v2.adam import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error


# --------------------------------------------------
#
# --------------------------------------------------
def predict_ann(x_train, y_train, x_test, y_test,
                my_dense_units):
    model = Sequential()
    model.add(Dense(my_dense_units, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dense(units=my_dense_units, activation='relu'))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=100, batch_size=32)

    predictions = model.predict(x_test)

    model.summary()
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}")
    # model.save('./data/models/ann.model')

    return predictions