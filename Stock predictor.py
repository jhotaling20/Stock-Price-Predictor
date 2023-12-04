import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import datetime


# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data.dropna()


# Function to prepare data for LSTM
def prepare_lstm_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i - time_step:i])
        y.append(scaled_data[i, 0])  # Predicting next opening price
    X, y = np.array(X), np.array(y)
    return X, y, scaler


# Function to build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Function to plot actual vs. predicted prices
def plot_predictions(actual, predicted, title):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, color='blue', label='Actual Prices')
    plt.plot(predicted, color='red', label='Predicted Prices')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


# Main script
if __name__ == "__main__":
    major_stocks = ['TSLA', 'AAPL', 'QQQ', 'SPY']  # SPY as a proxy for S&P 500
    for ticker in major_stocks:
        print(f"Processing {ticker}...")

        start_date = '2010-01-01'
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        stock_data = fetch_stock_data(ticker, start_date, end_date)

        if not stock_data.empty:
            features = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
            X, y, scaler = prepare_lstm_data(features)

            # Splitting data: All but last year for training, last year for testing
            test_start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
            test_data = stock_data.loc[test_start_date:end_date]
            train_data = stock_data.loc[:test_start_date]

            X_train, y_train, _ = prepare_lstm_data(train_data[features.columns])
            X_test, y_test, _ = prepare_lstm_data(test_data[features.columns])

            lstm_model = build_lstm_model(X_train.shape[1:])
            early_stopping = EarlyStopping(monitor='val_loss', patience=5)
            lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),
                           callbacks=[early_stopping])

            # Making predictions and evaluating the model
            predictions = lstm_model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            print(f"Mean Squared Error for {ticker}: {mse}")

            # Plotting actual vs. predicted prices
            plot_predictions(y_test, predictions.flatten(), f"{ticker} Stock Price Prediction")
        else:
            print(f"No data available for {ticker}.")
