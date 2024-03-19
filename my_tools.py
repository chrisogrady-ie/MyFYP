import pandas as pd
import numpy as np


# --------------------------------------------------
# return a dataframe of requested stock in for of (datetime : price)
# --------------------------------------------------
def fetch_data(stock_abbreviation, size):
    df = pd.read_csv('data/stocks/' + stock_abbreviation + '.csv')
    df = df.tail(size)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Close']]
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
        print('ann structure complete')

    if structure == 'rnn':
        print('rnn structure complete')

    if structure == 'lstm':
        for x in range(days, len(inputs)):
            x_list.append(inputs[x - days:x])
            y_list.append(inputs[x])
        print('lstm structure complete')

    x_list, y_list = np.array(x_list), np.array(y_list)
    return x_list, y_list
