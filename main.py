
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import ann_model, lstm_model, my_tools


def make_graph(train, test, predicted, stock_abbreviation):
    print(test[:10])
    print(predicted[:10])
    test_appended = np.append(train, test, axis=0)
    predicted_appended = np.append(train, predicted, axis=0)
    # plt.plot(train, color="black", label="Train")
    plt.plot(predicted_appended, color="red", label="Predicted Price")
    plt.plot(test_appended, color="black", label="Trade Price")
    # plt.plot(train_price_dataframe.index, train_price_dataframe['Close'])
    # plt.plot(test_price_dataframe.index, test_price_dataframe['Close'])
    plt.title(stock_abbreviation + " price")
    plt.xlabel('Days')
    plt.ylabel('Price (USD)')
    # plt.legend(['Train', 'Test'])
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(predicted, color="red", label="Predicted Price")
    plt.plot(test, color="black", label="Trade Price")
    plt.title(stock_abbreviation + " price")
    plt.xlabel('Days')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # --------------------------------------------------
    # Turn on and off predictions
    # Total days from dataset
    # Days to remember
    # Network variables
    # --------------------------------------------------
    stock_abbreviation = "GOOGL"

    ann_model_on = False
    rnn_model_on = False
    lstm_model_on = True

    days_total = 5500
    days_memory = 50
    test_train_split_size = 0.8
    my_epochs = 12
    my_dense_units = 1

    # --------------------------------------------------
    # Retrieve the dataset and format it
    # --------------------------------------------------
    df = my_tools.fetch_data(stock_abbreviation, days_total)

    # --------------------------------------------------
    # Scale the data
    # --------------------------------------------------
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df)
    scaled_data = scaler.transform(df)

    # --------------------------------------------------
    # Get train and test subsets
    # Build x to hold number of days and y to hold the next day
    # --------------------------------------------------
    train_price_scaled, test_price_scaled = my_tools.create_train_test(scaled_data, test_train_split_size)

    x_train, y_train = [], []
    x_test, y_test = [], []

    # --------------------------------------------------
    # ANN predictions
    # --------------------------------------------------
    if ann_model_on is True:
        print("=====Running ANN mode=====")
        structure = 'ann'
        x_train, y_train = my_tools.create_structure(train_price_scaled, days_memory, structure)
        x_test, y_test = my_tools.create_structure(test_price_scaled, days_memory, structure)

        ann_predictions = ann_model.predict_ann(x_train, y_train, x_test, y_test, my_dense_units)

        # --------------------------------------------------
        # Reversing scaler
        # --------------------------------------------------
        ann_y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
        ann_y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        lstm_predictions = scaler.inverse_transform(ann_predictions)

        make_graph(ann_y_train, ann_y_test, lstm_predictions)

        print("=====ANN complete======")

    # --------------------------------------------------
    # RNN predictions
    # --------------------------------------------------
    if rnn_model_on is True:
        print("=====Running RNN mode=====")
        structure = 'rnn'

        print("=====RNN complete=====")

    # --------------------------------------------------
    # LSTM predictions
    # --------------------------------------------------
    if lstm_model_on is True:
        print("=====Running LSTM mode=====")
        structure = 'lstm'
        x_train, y_train = my_tools.create_structure(train_price_scaled, days_memory, structure)
        x_test, y_test = my_tools.create_structure(test_price_scaled, days_memory, structure)

        lstm_predictions = lstm_model.predict_lstm(x_train, y_train, x_test, y_test, my_epochs)

        # --------------------------------------------------
        # Reversing scaler
        # --------------------------------------------------
        lstm_y_train = scaler.inverse_transform(y_train).round(decimals=2)
        lstm_y_test = scaler.inverse_transform(y_test).round(decimals=2)
        lstm_prediction_values = scaler.inverse_transform(lstm_predictions).round(decimals=2)
        # for x in range(len(lstm_y_test)):
        #    print(lstm_y_test[x], lstm_prediction_values[x], lstm_y_test[x] - lstm_prediction_values[x])
        make_graph(lstm_y_train, lstm_y_test, lstm_prediction_values, stock_abbreviation)

        print("=====LSTM complete=====")


# --------------------------------------------------
# 1 - Using GPU, start the program
# --------------------------------------------------
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
with tf.device('/gpu:0'):
    main()
