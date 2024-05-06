import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import ann_model, rnn_model, lstm_model, my_tools
import numpy as np


def calc_stock_sharpe_ratio():
    d = 255  # trading days
    rf = 0.023  # risk-free rate
    stock_abbreviation = "INTU"
    df = my_tools.fetch_data(stock_abbreviation, 0)
    change = df['Close'].pct_change().dropna()

    mean = change.mean() * d - rf
    sigma = change.std() * np.sqrt(d)
    print(mean/sigma)


def trading(stock_abbreviation, stocks, initial_investment, short_days, long_days, risk_free_rate):
    my_funds = initial_investment
    my_position = 0
    position_quantity = 0
    my_purchases = {}
    my_sales = {}
    my_transactions = {}
    my_transactions_percent = {}
    current_ticker = {}

    for this_date, r in stocks.iterrows():
        current_price = r['Close']
        current_ticker[this_date] = current_price

        my_short_timeframe = (pd.DataFrame(current_ticker.items(), columns=['Date', 'Price'])).tail(short_days)
        my_short_timeframe.index = my_short_timeframe.pop("Date")

        my_long_timeframe = (pd.DataFrame(current_ticker.items(), columns=['Date', 'Price'])).tail(long_days)
        my_long_timeframe.index = my_long_timeframe.pop("Date")

        short_moving_average = np.mean(my_short_timeframe['Price'].values)
        long_moving_average = np.mean(my_long_timeframe['Price'].values)
        # print(r['Close'], short_moving_average)

        # ==============================================================================================================
        # trading logic start
        # ==============================================================================================================

        buy_signal = False
        sell_signal = False
        if position_quantity > 0:
            if short_moving_average < long_moving_average:
                sell_signal = True
            elif my_position * 0.90 > current_price:
                sell_signal = True
            elif my_position * 1.10 < current_price:
                sell_signal = True
            #elif short_moving_average < current_price:
            #    sell_signal = True

            if sell_signal:
                sale_value = current_price * position_quantity
                purchase_value = my_position * position_quantity

                my_funds += sale_value
                my_funds = round(my_funds, 2)

                my_transactions[this_date] = sale_value - purchase_value
                my_transactions_percent[this_date] = ((sale_value - purchase_value)/purchase_value)*100
                my_sales[this_date] = current_price

                position_quantity = 0
                my_position = 0

        else:
            #if short_moving_average > current_price:
            #    buy_signal = True
            #elif long_moving_average > current_price:
            #    buy_signal = True
            if short_moving_average > long_moving_average:
                buy_signal = True

            if buy_signal:
                position_quantity = my_funds // current_price
                if position_quantity > 0:
                    current_transaction = position_quantity * current_price
                    my_funds = my_funds - current_transaction
                    my_purchases[this_date] = current_price
                    my_position = current_price

        # ==============================================================================================================
        # trading logic end
        # ==============================================================================================================

        if this_date == user_stocks.index.max() and position_quantity > 0:
            sale_value = current_price * position_quantity
            purchase_value = my_position * position_quantity
            my_funds += sale_value
            my_funds = round(my_funds, 2)

            my_transactions[this_date] = sale_value - purchase_value
            my_transactions_percent[this_date] = ((sale_value - purchase_value)/purchase_value)*100
            my_sales[this_date] = current_price

    purchase_df = pd.DataFrame(my_purchases.items(), columns=['Date', 'Price'])
    purchase_df.index = purchase_df.pop("Date")

    sales_df = pd.DataFrame(my_sales.items(), columns=['Date', 'Price'])
    sales_df.index = sales_df.pop("Date")

    transactions_df = pd.DataFrame(my_transactions.items(), columns=['Date', 'Price'])
    transactions_df.index = transactions_df.pop("Date")

    transactions_percent_df = pd.DataFrame(my_transactions_percent.items(), columns=['Date', 'Price'])
    transactions_percent_df.index = transactions_percent_df.pop("Date")

    my_tools.make_price_graph(stocks, stock_abbreviation, purchase_df, sales_df, my_funds, initial_investment)
    my_tools.calculate_portfolio(transactions_df, transactions_percent_df, risk_free_rate)


def main(stock_abbreviation, df, prediction_model, prediction_history,
         days_memory, learning_epochs, batch_size, user_dropout):
    test_train_split_size = 0.8
    # my_tools.make_price_graph(df, stock_abbreviation)

    # --------------------------------------------------
    # Scale the data
    # --------------------------------------------------
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

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
    if prediction_model == 'ann':
        print("=====Running ANN mode=====")
        structure = 'ann'
        x_train, y_train = my_tools.create_structure(train_price_scaled, days_memory, structure)
        x_test, y_test = my_tools.create_structure(test_price_scaled, days_memory, structure)

        ann_predictions = ann_model.predict_ann(x_train, y_train, x_test, y_test,
                                                learning_epochs, batch_size, user_dropout)


        # --------------------------------------------------
        # Reversing scaler
        # --------------------------------------------------
        ann_y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
        ann_y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        ann_predictions_values = scaler.inverse_transform(ann_predictions)

        my_tools.make_graph(ann_y_train, ann_y_test, ann_predictions_values, stock_abbreviation, structure)

        print("=====ANN complete======")

    # --------------------------------------------------
    # RNN predictions
    # --------------------------------------------------
    if prediction_model == 'rnn':
        print("=====Running RNN mode=====")
        structure = 'rnn'
        x_train, y_train = my_tools.create_structure(train_price_scaled, days_memory, structure)
        x_test, y_test = my_tools.create_structure(test_price_scaled, days_memory, structure)

        rnn_predictions = rnn_model.predict_rnn(x_train, y_train, x_test, y_test,
                                                learning_epochs, batch_size, user_dropout)

        # --------------------------------------------------
        # Reversing scaler
        # --------------------------------------------------
        rnn_y_train = scaler.inverse_transform(y_train).round(decimals=2)
        rnn_y_test = scaler.inverse_transform(y_test).round(decimals=2)
        rnn_predictions_values = scaler.inverse_transform(rnn_predictions).round(decimals=2)

        my_tools.make_graph(rnn_y_train, rnn_y_test, rnn_predictions_values, stock_abbreviation, structure)

        print("=====RNN complete=====")

    # --------------------------------------------------
    # LSTM predictions
    # --------------------------------------------------
    if prediction_model == 'lstm':
        print("=====Running LSTM mode=====")
        structure = 'lstm'
        x_train, y_train = my_tools.create_structure(train_price_scaled, days_memory, structure)
        x_test, y_test = my_tools.create_structure(test_price_scaled, days_memory, structure)

        lstm_predictions = lstm_model.predict_lstm(x_train, y_train, x_test, y_test,
                                                   learning_epochs, batch_size, user_dropout)

        # --------------------------------------------------
        # Reversing scaler
        # --------------------------------------------------
        lstm_y_train = scaler.inverse_transform(y_train).round(decimals=2)
        lstm_y_test = scaler.inverse_transform(y_test).round(decimals=2)
        lstm_prediction_values = scaler.inverse_transform(lstm_predictions).round(decimals=2)
        # for x in range(len(lstm_y_test)):
        #    print(lstm_y_test[x], lstm_prediction_values[x], lstm_y_test[x] - lstm_prediction_values[x])

        print("\n\n\nReal data:")
        mse, r2, mae, rmse, mape = my_tools.evaluate_model(lstm_y_test, lstm_prediction_values)
        my_tools.make_graph(lstm_y_train, lstm_y_test, lstm_prediction_values, stock_abbreviation, structure)

        print("=====LSTM complete=====")


# --------------------------------------------------
# 1 - Using GPU, start the program
# --------------------------------------------------
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
with tf.device('/gpu:0'):
    input_file_name = 'my_input.txt'
    user_inputs = my_tools.read_input(input_file_name)
    print(user_inputs)

    user_stocks = my_tools.fetch_data(user_inputs['stock_abbreviation'], user_inputs['total_days'])

    if user_inputs['predict'] != '0':
        print('starting predictions')
        main(
            user_inputs['stock_abbreviation'],
            user_stocks,
            user_inputs['prediction_model'],
            int(user_inputs['prediction_history']),
            int(user_inputs['learning_days']),
            int(user_inputs['learning_epochs']),
            int(user_inputs['batch_size']),
            float(user_inputs['dropout'])
        )
    trading(
        user_inputs['stock_abbreviation'],
        user_stocks,
        int(user_inputs['investment_amount']),
        int(user_inputs['short_moving_averages']),
        int(user_inputs['long_moving_averages']),
        float(user_inputs['risk_free_rate'])
    )
    #calc_stock_sharpe_ratio()
