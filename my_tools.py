import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# --------------------------------------------------
# return a dataframe of requested stock in for of (datetime : price)
# --------------------------------------------------
def fetch_data(stock_abbreviation, size):
    df = pd.read_csv('data/stocks/' + stock_abbreviation + '.csv')
    df = df.tail(size)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Open']]
    df.index = df.pop("Date")
    return df


# --------------------------------------------------
# returns a train/test split
# --------------------------------------------------
def create_train_test(prices, split_size):
    # getting size of data set and splitting it as requested
    split = round(prices.size * split_size)

    if isinstance(prices, np.ndarray):
        train_price, test_price = prices[:split, ...], prices[split:, ...]
    elif isinstance(prices, pd.core.frame.DataFrame):
        train_price, test_price = prices.iloc[:split], prices.iloc[split:]
    else:
        print("----------UNKNOWN DATA----------")

    return train_price, test_price


# --------------------------------------------------
# Create a 2d structure for ann to learn from
# Create a 3d structure for lstm to learn from
# Create a 1d structure for prediction - feature
# --------------------------------------------------
def create_structure(inputs, days, structure):
    x_list = []
    y_list = []

    if structure == 'ann':
        for x in range(days, len(inputs)):
            x_list.append(inputs[x - days:x, 0])
            y_list.append(inputs[x, 0])
        print(structure + ' structure complete')

    if structure == 'lstm' or structure == 'rnn':
        for x in range(days, len(inputs)):
            x_list.append(inputs[x - days:x])
            y_list.append(inputs[x])
        print(structure + ' structure complete')

    x_list, y_list = np.array(x_list), np.array(y_list)
    return x_list, y_list


# --------------------------------------------------
# Plots graph with supplied data and abbreviation
# --------------------------------------------------
def make_graph(train, test, predicted, stock_abbreviation, model):
    # print(test[:10])
    # print(predicted[:10])
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
    plt.title(stock_abbreviation + " price prediction with " + model)
    plt.xlabel('Days')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()


# --------------------------------------------------
# calculates mape
# --------------------------------------------------
def mean_absolute_percentage_error(y_test, predictions):
    #y_test, predictions = np.array(y_test), np.array(predictions)
    #print(y_test)
    return np.mean(np.abs((y_test - predictions) / y_test)) * 100


# --------------------------------------------------
# Evaluates the model on the actual and predicted prices
# under r2, mae, rmse and mape
# --------------------------------------------------
def evaluate_model(y_test, predictions):
    print('evaluating.....')
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, predictions)

    dec = 4
    # print('Mean Squared Error: ',  mse)
    print('r2 score (Closer to 1 gives a good prediction):\n',
          r2.round(decimals=dec))
    print('Mean Absolute Error (Average error):\n',
          mae.round(decimals=dec))
    print('Root Mean Squared Error (Show how far predictions fall from actual prices):\n',
          rmse.round(decimals=dec))
    print('Mean Absolute Percentage Error (Percentage of difference from prediction to actual ):\n',
          mape.round(decimals=dec), '%')

    return mse, r2, mae, rmse, mape

