######################################################################################
# Main script for stock trend prediction and buy/sell signal generation.

# This script performs the following steps:
# 1. Prompts the user to enter a stock symbol.
# 2. Fetches historical stock data for the given symbol using a custom data-fetching function.
# 3. Enhances the data with additional features (e.g., moving averages, returns).
# 4. Trains a machine learning model to predict stock trends.
# 5. Trains a RandomForestRegressor to predict the next day's closing price.
# 6. Iteratively predicts the next 30 trading days' closing prices, generating buy/sell/hold signals based on predicted price direction.
# 7. Prints a table summarizing the 30-day predictions, including suggested buy and sell prices.
# Functions imported from other modules:
# - fetch_stock_data: Retrieves historical stock data.
# - add_features: Adds technical indicators/features to the data.
# - train_model: Trains a model for trend prediction.
# - predict_trend: Predicts the trend using the trained model.
# - find_best_buying_day: (Optional) Finds the optimal buying day for the stock.

# Dependencies:
# - yfinance
# - numpy
# - pandas
# - scikit-learn

# Usage:
#     Run the script and enter a valid stock symbol when prompted. The script will output a 30-day forecast table with buy/sell signals.
######################################################################################

import yfinance as yf
import numpy as np
import pandas as pd
from ai_estimate import fetch_stock_data, add_features, train_model, predict_trend
from stock_estimate import find_best_buying_day
from sklearn.ensemble import RandomForestRegressor

def main():

    symbol = input("Enter stock symbol (e.g., AAPL): ").upper()
    df = fetch_stock_data(symbol, period="12mo", interval="1d")
    if df.empty:
        return
    df_feat = add_features(df)
    model = train_model(df_feat)
    trend = predict_trend(model, df_feat)

    features = ['Return', 'MA5', 'MA10', 'MA20']
    df_reg = df_feat.copy()
    df_reg['TargetClose'] = df_reg['Close'].shift(-1)
    df_reg = df_reg.dropna()
    X = df_reg[features]
    y = df_reg['TargetClose']
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X, y)
    last_row = df_feat.iloc[-1].copy()


    buy_sell_table = []
    prev_pred = last_row['Close']
    if isinstance(prev_pred, pd.Series):
        prev_pred = float(prev_pred.item())
    else:
        prev_pred = float(prev_pred)

    # Generate next 30 trading (business) days starting from the last date in df_feat
    last_date = df_feat.index[-1]
    if not isinstance(last_date, pd.Timestamp):
        # If index is not datetime, try to convert
        if hasattr(df_feat, 'index') and hasattr(df_feat.index, 'to_datetimeindex'):
            last_date = df_feat.index.to_datetimeindex()[-1]
        else:
            last_date = pd.to_datetime(df_feat.index[-1])
    # Generate 30 trading days after last_date (skip last_date itself)
    trading_days = pd.bdate_range(last_date + pd.tseries.offsets.BDay(1), periods=30)

    for i in range(30):
        input_features = last_row[features].values.reshape(1, -1)
        next_close = float(reg.predict(input_features)[0])
        buy_price = next_close * 0.98
        sell_price = next_close * 1.02
        # Direction: Up if next_close > prev_pred, Down if <, else Flat
        if isinstance(prev_pred, pd.Series):
            prev_pred_val = float(prev_pred.item())
        else:
            prev_pred_val = float(prev_pred)
        if next_close > prev_pred_val:
            direction = 'Up'
        elif next_close < prev_pred_val:
            direction = 'Down'
        else:
            direction = 'Flat'
        # Simple signal logic: Buy if direction Up, Sell if Down, Hold if Flat
        if direction == 'Up':
            signal = 'Buy'
        elif direction == 'Down':
            signal = 'Sell'
        else:
            signal = 'Hold'
        day_label = trading_days[i].strftime('%Y-%m-%d')
        buy_sell_table.append((day_label, next_close, direction, signal, buy_price, sell_price))
        prev_pred = next_close
        # Update last_row for next prediction
        prev_close = last_row['Close']
        if isinstance(prev_close, pd.Series):
            prev_close = prev_close.item()
        else:
            prev_close = float(prev_close)
        last_row['Close'] = float(next_close)
        last_row['Return'] = (next_close - prev_close) / prev_close if prev_close != 0 else 0
        close_series = df_feat['Close'].squeeze()
        last_ma5 = (close_series[-4:].tolist() + [next_close])[-5:] if len(close_series) >= 4 else [next_close]*5
        last_row['MA5'] = float(np.mean(last_ma5))
        last_ma10 = (close_series[-9:].tolist() + [next_close])[-10:] if len(close_series) >= 9 else [next_close]*10
        last_row['MA10'] = float(np.mean(last_ma10))
        last_ma20 = (close_series[-19:].tolist() + [next_close])[-20:] if len(close_series) >= 19 else [next_close]*20
        # Append a full row to keep DataFrame structure consistent
        new_row = {col: last_row[col] if col in last_row else np.nan for col in df_feat.columns}
        df_feat = pd.concat([df_feat, pd.DataFrame([new_row])], ignore_index=True)


    # Add icons for direction and signal
    # Improved direction icons: green up, red down, yellow right
    direction_icons = {
        'Up': '\U0001F53A',      # 🔺
        'Down': '\U0001F53B',    # 🔻
        'Flat': '\U0001F7E1'     # 🟡 (yellow circle for flat)
    }
    signal_icons = {'Buy': '🟢', 'Sell': '🔴', 'Hold': '⏸️'}

    # Set all columns to the same fixed width for even spacing
    col_width = 16
    print(f"{'Date':<{col_width}}| {'Prediction':>{col_width}}| {'Direction':<{col_width}}| {'Signal':<{col_width}}| {'Buy':>{col_width}}| {'Sell':>{col_width}}")
    print('-' * (col_width * 6 + 5))
    for day, pred, direction, signal, buy, sell in buy_sell_table:
        dir_icon = direction_icons.get(direction, '')
        sig_icon = signal_icons.get(signal, '')
        direction_col = f"{direction} {dir_icon}".ljust(col_width)
        signal_col = f"{signal} {sig_icon}".ljust(col_width)
        print(f"{day:<{col_width}}| {pred:>{col_width}.2f}| {direction_col}| {signal_col}| {buy:>{col_width}.2f}| {sell:>{col_width}.2f}")

    # Optionally, suppress output from stock_estimate.py by redirecting stdout if needed
    # import contextlib, io
    # with contextlib.redirect_stdout(io.StringIO()):
    #     find_best_buying_day(symbol)

if __name__ == "__main__":
    main()
