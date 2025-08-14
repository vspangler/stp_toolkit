######################################################################################
# This module provides functions to fetch historical stock data, engineer features, train machine learning models to predict stock trends, and forecast future closing prices using Random Forest algorithms.

# Functions:
# -----------
# fetch_stock_data(symbol, period="6mo", interval="1d"):
#     Downloads historical stock data for a given symbol using yfinance.
# add_features(df):
#     Adds engineered features to the DataFrame, including daily returns, moving averages (5, 10, 20 days), and a binary target indicating if the next day's close is higher.
# train_model(df):
#     Trains a Random Forest Classifier to predict the next-day trend (up or down) based on engineered features. Prints model accuracy.
# predict_trend(model, df):
#     Uses the trained classifier to predict the trend (up or down) for the most recent data point.
# main():
#     Main function to interact with the user, fetch data, train models, make trend predictions, and output a 30-day forecast of predicted closing prices with suggested buy/sell prices.

# Usage:
# ------
# Run this script directly to input a stock symbol, receive AI-driven buy/sell suggestions, and view a 30-day price forecast.
######################################################################################
from sklearn.ensemble import RandomForestRegressor

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def fetch_stock_data(symbol, period="6mo", interval="1d"):
    data = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
    return data

def add_features(df):
    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df = df.dropna()
    return df

def train_model(df):
    features = ['Return', 'MA5', 'MA10', 'MA20']
    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.2f}")
    return model

def predict_trend(model, df):
    features = ['Return', 'MA5', 'MA10', 'MA20']
    latest = df.iloc[-1][features].values.reshape(1, -1)
    pred = model.predict(latest)[0]
    return pred

def main():
    symbol = input("Enter stock symbol (e.g., AAPL): ").upper()
    print(f"Fetching data for {symbol}...")
    df = fetch_stock_data(symbol)
    if df.empty:
        print("No data found for symbol.")
        return
    df = add_features(df)
    model = train_model(df)
    trend = predict_trend(model, df)
    last_val = df['Close'].iloc[-1]
    latest_close = last_val.item() if isinstance(last_val, pd.Series) else float(last_val)
    buy_price = latest_close
    sell_price = latest_close
    target_pct = 0.05  # 5% target
    if trend == 1:
        potential_sell = buy_price * (1 + target_pct)
        print(f"AI suggests a potential BUY signal for {symbol} (trend up)")
        print(f"Potential BUY price: {buy_price:.2f}")
        print(f"Potential SELL price (target +5%): {potential_sell:.2f}")
    else:
        potential_buy = sell_price * (1 - target_pct)
        print(f"AI suggests a potential SELL signal for {symbol} (trend down)")
        print(f"Potential SELL price: {sell_price:.2f}")
        print(f"Potential BUY price (target -5%): {potential_buy:.2f}")

    # 30-day forecast
    print("\n30-day forecast (predicted closing prices):")
    # Use the same features as before, but for regression
    features = ['Return', 'MA5', 'MA10', 'MA20']
    df_reg = df.copy()
    df_reg['TargetClose'] = df_reg['Close'].shift(-1)
    df_reg = df_reg.dropna()
    X = df_reg[features]
    y = df_reg['TargetClose']
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X, y)

    last_row = df.iloc[-1].copy()
    forecast = []
    print(f"{'Day':>3} | {'Predicted Close':>15} | {'Buy Price (-2%)':>16} | {'Sell Price (+2%)':>17}")
    print('-'*60)
    for i in range(30):
        input_features = last_row[features].values.reshape(1, -1)
        next_close = reg.predict(input_features)[0]
        buy_price = next_close * 0.98
        sell_price = next_close * 1.02
        print(f"{i+1:>3} | {next_close:>15.2f} | {buy_price:>16.2f} | {sell_price:>17.2f}")
        # Update last_row for next prediction
        prev_close = last_row['Close']
        if isinstance(prev_close, pd.Series):
            prev_close = prev_close.item()
        else:
            prev_close = float(prev_close)
        last_row['Close'] = float(next_close)
        last_row['Return'] = (next_close - prev_close) / prev_close if prev_close != 0 else 0
        close_series = df['Close'].squeeze()
        last_ma5 = (close_series[-4:].tolist() + [next_close])[-5:] if len(close_series) >= 4 else [next_close]*5
        last_row['MA5'] = float(np.mean(last_ma5))
        last_ma10 = (close_series[-9:].tolist() + [next_close])[-10:] if len(close_series) >= 9 else [next_close]*10
        last_row['MA10'] = float(np.mean(last_ma10))
        last_ma20 = (close_series[-19:].tolist() + [next_close])[-20:] if len(close_series) >= 19 else [next_close]*20
        last_row['MA20'] = float(np.mean(last_ma20))
        # Append a full row to keep DataFrame structure consistent
        new_row = {col: last_row[col] if col in last_row else np.nan for col in df.columns}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

if __name__ == "__main__":
    main()
