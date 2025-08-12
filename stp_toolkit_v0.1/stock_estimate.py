######################################################################################
# Stock Estimate & AI/Deep Learning Forecast Script
#
# DESCRIPTION:
#   Comprehensive stock analysis and forecasting tool with technical indicators and AI (LSTM) predictions.
#
# REQUIREMENTS:
#   - Python 3.8 or newer
#   - Libraries: yfinance, pandas, numpy, scikit-learn, tensorflow
#     Install with:
#         pip install yfinance pandas numpy scikit-learn tensorflow
#
# WINDOWS LONG PATH SUPPORT (if needed):
#   - Open Start menu, type "regedit" (Registry Editor)
#   - Navigate to: HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
#   - Set LongPathsEnabled to 1, then restart your computer
#
# FEATURES:
#   - 50-day and 200-day moving averages (momentum analysis)
#   - Support and resistance levels (recent high/low)
#   - Volume at support and resistance levels
#   - Recommended buy value based on average of daily highs/lows for last 5, 30, 60, 90, 180, and 252 (1 year) trading days
#   - Forecasted best buy and sell values for next 30 days using linear regression
#   - Technical indicators:
#       - MACD, Stochastic Oscillator, Trendline, Price Alerts
#       - Bollinger Bands, RSI, EMA Crossovers, Candlestick Patterns, Volume Spikes
#   - AI/Deep Learning (LSTM) prediction:
#       - Best buy/sell value for next 30 days
#       - Daily forecast table: predicted price, direction, probability, volatility, confidence interval, anomaly detection, buy/sell/hold signal
#       - Volatility estimate and confidence intervals
#       - Anomaly detection for unusual price changes
#       - Direct buy/sell/hold signals for each day
#
# USAGE:
#   - Run the script and enter a stock symbol (e.g., AAPL, MSFT, TSLA) when prompted
#   - Output will display all analysis and forecasts in the terminal
#
# ARGUMENTS:
#   symbol (str): The stock ticker symbol to analyze
#
# OUTPUT:
#   - Prints all analysis, technical indicators, and AI/Deep Learning forecasts for the given symbol
#
# NOTES:
#   - For best results, ensure all required libraries are installed and up to date
#   - For more advanced predictions, consider adding ensemble models, feature engineering, or backtesting
######################################################################################
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def find_best_buying_day(symbol):
    # Define time range
    end_date = datetime.today()
    start_date = end_date - timedelta(days=400)  # ~19 months to ensure enough data for 200 trading days

    # Download historical data
    data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), auto_adjust=False)
    close_prices = data['Close']

    if data.empty or len(data) < 50:
        print(f"Not enough data to calculate moving averages for {symbol}. Try a different symbol or check the date range.")
        return

    # Calculate moving averages for momentum analysis
    ma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
    ma_200 = data['Close'].rolling(window=200).mean().iloc[-1]

    # Ensure ma_50 and ma_200 are scalars
    if isinstance(ma_50, pd.Series):
        ma_50 = ma_50.iloc[0]
    if isinstance(ma_200, pd.Series):
        ma_200 = ma_200.iloc[0]

    # Compare short-term vs. long-term trend
    print("\n" + "="*40)
    print("        MOMENTUM ANALYSIS        ")
    print("="*40)
    if not pd.isna(ma_50) and not pd.isna(ma_200):
        if ma_50 > ma_200:
            print("  Momentum: Bullish")
            print("    - Short-term average is above long-term")
        elif ma_50 < ma_200:
            print("  Momentum: Bearish")
            print("    - Short-term average is below long-term")
        else:
            print("  Momentum: Neutral")
            print("    - Short-term and long-term averages are equal")
    else:
        print("  Momentum: Insufficient data to determine.")
    print("\n" + "-"*40)

    print("        MOVING AVERAGES        ")
    print("-"*40)
    if pd.isna(ma_50):
        print(f"  50-day MA (short-term):      Insufficient data for {symbol}.")
    else:
        print(f"  50-day MA (short-term):      ${ma_50:>8.2f}")
    if pd.isna(ma_200):
        print(f"  200-day MA (long-term):      Insufficient data for {symbol}.")
    else:
        print(f"  200-day MA (long-term):      ${ma_200:>8.2f}")
    print("\n" + "-"*40)

    # Support and resistance levels
    recent_high = data['High'].max()
    recent_low = data['Low'].min()
    # Ensure recent_high and recent_low are scalars
    if isinstance(recent_high, pd.Series):
        recent_high = recent_high.iloc[0]
    if isinstance(recent_low, pd.Series):
        recent_low = recent_low.iloc[0]
    print("        SUPPORT & RESISTANCE LEVELS        ")
    print("-"*40)
    print(f"  Resistance (recent high): ${float(recent_high):>8.2f}   | Highs may signal resistance.")
    print(f"  Support    (recent low):  ${float(recent_low):>8.2f}   | Lows might show support.")
    # Recommended sell value: average of daily highs over last 60 days
    recent_60 = data.tail(60)
    sell_value = recent_60['High'].mean()
    if isinstance(sell_value, pd.Series):
        sell_value = sell_value.iloc[0]
    print(f"  Recommended sell value (last 60 days avg high): ${float(sell_value):>8.2f}")
    print("\n" + "-"*40)

    import numpy as np
    # Volume at extremes using np.isclose for float comparison
    volume_at_high = data['Volume'][np.isclose(data['High'], recent_high)]
    volume_at_low = data['Volume'][np.isclose(data['Low'], recent_low)]
    # If multiple days match, take the max volume for each extreme
    # Fix for FutureWarning: handle Series from .max()
    def safe_float_max(series):
        if not series.empty:
            m = series.max()
            if isinstance(m, pd.Series):
                return float(m.iloc[0])
            else:
                return float(m)
        return None
    vol_high = safe_float_max(volume_at_high)
    vol_low = safe_float_max(volume_at_low)
    print("        VOLUME AT EXTREMES        ")
    print("-"*40)
    if vol_low is not None:
        print(f"  Volume at support (low):     {vol_low:,.0f}   | High volume at a low could mean strong buying interest.")
    else:
        print("  No volume data available for support (low) level.")
    if vol_high is not None:
        print(f"  Volume at resistance (high): {vol_high:,.0f}   | High volume at a high might suggest a peak.")
    else:
        print("  No volume data available for resistance (high) level.")
    print("\n" + "-"*40)

    # Helper function to calculate recommended buy value
    def calc_recommended_buy_value(df):
        avg_high = df['High'].mean()
        avg_low = df['Low'].mean()
        if isinstance(avg_high, pd.Series):
            avg_high = avg_high.iloc[0]
        if isinstance(avg_low, pd.Series):
            avg_low = avg_low.iloc[0]
        return float((avg_high + avg_low) / 2)

    # Calculate for last N trading days, including 1 year (252 trading days)
    periods = [
        (5, "5"),
        (30, "30"),
        (60, "60"),
        (90, "90"),
        (180, "180"),
        (252, "1 year")
    ]
    results = {}
    for days, label in periods:
        recent = data.tail(days)
        value = calc_recommended_buy_value(recent)
        results[label] = value

    # Print results
    print("        RECOMMENDED BUY VALUES        ")
    print("-"*40)
    for label in ["5", "30", "60", "90", "180", "1 year"]:
        value = results[label]
        if pd.isna(value):
            print(f"  No sufficient data to calculate recommended buy value for {symbol} over last {label} trading days.")
        else:
            if label == "1 year":
                print(f"  Last 1 year (252 trading days):   ${value:>8.2f}")
            else:
                print(f"  Last {label:>3} trading days:         ${value:>8.2f}")
    print("="*40 + "\n")

    # Estimate best buy value for the next 30 days using linear regression
    print("        FORECAST: NEXT 30 DAYS BUY VALUE        ")
    print("-"*40)
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        print("scikit-learn is required for forecasting. Please install with 'pip install scikit-learn'.")
        return
    # Use last 60 days of average high/low as features
    recent = data.tail(60)
    avg_prices = (recent['High'] + recent['Low']) / 2
    high_prices = recent['High']
    if len(avg_prices) < 30:
        print("  Not enough data to forecast next 30 days buy value.")
        return
    X = np.arange(len(avg_prices)).reshape(-1, 1)
    y = avg_prices.values
    model = LinearRegression()
    model.fit(X, y)
    future_X = np.arange(len(avg_prices), len(avg_prices) + 30).reshape(-1, 1)
    future_pred = model.predict(future_X)
    forecast_value = np.mean(future_pred)
    print(f"  Estimated best buy value for next 30 days:   ${forecast_value:>8.2f}")


    # --- EMA Calculations (used for MACD and EMA Crossovers) ---
    ema_short = data['Close'].ewm(span=12, adjust=False).mean()
    ema_long = data['Close'].ewm(span=26, adjust=False).mean()

    # --- MACD ---
    print("        MACD (Moving Average Convergence Divergence)        ")
    print("-"*40)
    macd = ema_short - ema_long
    signal = macd.ewm(span=9, adjust=False).mean()
    if len(macd) >= 9:
        macd_last = macd.iloc[-1].item() if hasattr(macd.iloc[-1], 'item') else float(macd.iloc[-1])
        signal_last = signal.iloc[-1].item() if hasattr(signal.iloc[-1], 'item') else float(signal.iloc[-1])
        macd_prev = macd.iloc[-2].item() if hasattr(macd.iloc[-2], 'item') else float(macd.iloc[-2])
        signal_prev = signal.iloc[-2].item() if hasattr(signal.iloc[-2], 'item') else float(signal.iloc[-2])
        print(f"  MACD: {macd_last:>8.2f}")
        print(f"  Signal line: {signal_last:>8.2f}")
        if macd_prev < signal_prev and macd_last > signal_last:
            print("  Signal: BUY (MACD crossed above signal line)")
        elif macd_prev > signal_prev and macd_last < signal_last:
            print("  Signal: SELL (MACD crossed below signal line)")
        else:
            print("  Signal: No crossover detected.")
    else:
        print("  Not enough data for MACD.")
    print("\n" + "-"*40)

    # --- Stochastic Oscillator ---
    print("        STOCHASTIC OSCILLATOR        ")
    print("-"*40)
    low_min = data['Low'].rolling(window=14).min()
    high_max = data['High'].rolling(window=14).max()
    percent_k = 100 * ((close_prices - low_min) / (high_max - low_min))
    percent_d = percent_k.rolling(window=3).mean()
    if len(percent_k) >= 14:
        k_last = percent_k.iloc[-1].item() if hasattr(percent_k.iloc[-1], 'item') else float(percent_k.iloc[-1])
        d_last = percent_d.iloc[-1].item() if hasattr(percent_d.iloc[-1], 'item') else float(percent_d.iloc[-1])
        print(f"  %K: {k_last:>6.2f}")
        print(f"  %D: {d_last:>6.2f}")
        if k_last < 20 and d_last < 20:
            print("  Signal: Potential BUY (Stochastic oversold)")
        elif k_last > 80 and d_last > 80:
            print("  Signal: Potential SELL (Stochastic overbought)")
        else:
            print("  Signal: Neutral")
    else:
        print("  Not enough data for Stochastic Oscillator.")
    print("\n" + "-"*40)

    # --- Trendline Detection ---
    print("        TRENDLINE DETECTION        ")
    print("-"*40)
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        print("scikit-learn is required for trendline detection. Please install with 'pip install scikit-learn'.")
        return
    X_trend = np.arange(len(close_prices)).reshape(-1, 1)
    y_trend = close_prices.values
    model_trend = LinearRegression()
    model_trend.fit(X_trend, y_trend)
    slope = model_trend.coef_[0]
    print(f"  Trendline slope: {slope.item():>8.4f}")
    if slope > 0:
        print("  Trend: Upward")
    elif slope < 0:
        print("  Trend: Downward")
    else:
        print("  Trend: Flat")
    print("\n" + "-"*40)

    # --- Price Alerts ---
    print("        PRICE ALERTS        ")
    print("-"*40)
    last_close = close_prices.iloc[-1].item() if hasattr(close_prices.iloc[-1], 'item') else float(close_prices.iloc[-1])
    # Example thresholds (user can modify these)
    alert_high = recent_high * 1.02  # 2% above recent high
    alert_low = recent_low * 0.98    # 2% below recent low
    if last_close >= alert_high:
        print(f"  ALERT: Price ${last_close:>8.2f} has crossed above alert high (${alert_high:>8.2f})!")
    elif last_close <= alert_low:
        print(f"  ALERT: Price ${last_close:>8.2f} has crossed below alert low (${alert_low:>8.2f})!")
    else:
        print("  No price alert triggered.")
    print("\n" + "-"*40)
    # Forecast sell value for next 30 days using high prices
    X_high = np.arange(len(high_prices)).reshape(-1, 1)
    y_high = high_prices.values
    model_high = LinearRegression()
    model_high.fit(X_high, y_high)

    # --- Bollinger Bands ---
    print("        BOLLINGER BANDS        ")
    print("-"*40)
    close_prices = data['Close']
    window = 20
    if len(close_prices) >= window:
        ma = close_prices.rolling(window=window).mean()
        std = close_prices.rolling(window=window).std()
        upper_band = ma + 2 * std
        lower_band = ma - 2 * std
        last_close = close_prices.iloc[-1].item() if hasattr(close_prices.iloc[-1], 'item') else float(close_prices.iloc[-1])
        last_upper = upper_band.iloc[-1].item() if hasattr(upper_band.iloc[-1], 'item') else float(upper_band.iloc[-1])
        last_lower = lower_band.iloc[-1].item() if hasattr(lower_band.iloc[-1], 'item') else float(lower_band.iloc[-1])
        print(f"  Last close: ${last_close:>8.2f}")
        print(f"  Upper band: ${last_upper:>8.2f}")
        print(f"  Lower band: ${last_lower:>8.2f}")
        if last_close <= last_lower:
            print("  Signal: Potential BUY (price at/below lower band)")
        elif last_close >= last_upper:
            print("  Signal: Potential SELL (price at/above upper band)")
        else:
            print("  Signal: Neutral")
    else:
        print("  Not enough data for Bollinger Bands.")
    print("\n" + "-"*40)

    # --- RSI ---
    print("        RELATIVE STRENGTH INDEX (RSI)        ")
    print("-"*40)
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    if len(close_prices) >= 14:
        rsi = calc_rsi(close_prices)
        last_rsi = rsi.iloc[-1].item() if hasattr(rsi.iloc[-1], 'item') else float(rsi.iloc[-1])
        print(f"  RSI (last): {last_rsi:>6.2f}")
        if last_rsi < 30:
            print("  Signal: Potential BUY (RSI < 30, oversold)")
        elif last_rsi > 70:
            print("  Signal: Potential SELL (RSI > 70, overbought)")
        else:
            print("  Signal: Neutral")
    else:
        print("  Not enough data for RSI.")
    print("\n" + "-"*40)

    # --- EMA Crossovers ---
    print("        EMA CROSSOVERS        ")
    print("-"*40)
    if len(data['Close']) >= 26:
        ema_short_prev = ema_short.iloc[-2].item() if hasattr(ema_short.iloc[-2], 'item') else float(ema_short.iloc[-2])
        ema_long_prev = ema_long.iloc[-2].item() if hasattr(ema_long.iloc[-2], 'item') else float(ema_long.iloc[-2])
        ema_short_last = ema_short.iloc[-1].item() if hasattr(ema_short.iloc[-1], 'item') else float(ema_short.iloc[-1])
        ema_long_last = ema_long.iloc[-1].item() if hasattr(ema_long.iloc[-1], 'item') else float(ema_long.iloc[-1])
        if ema_short_prev < ema_long_prev and ema_short_last > ema_long_last:
            print("  Signal: BUY (12 EMA crossed above 26 EMA)")
        elif ema_short_prev > ema_long_prev and ema_short_last < ema_long_last:
            print("  Signal: SELL (12 EMA crossed below 26 EMA)")
        else:
            print("  Signal: No crossover detected.")
        print(f"  12 EMA: ${ema_short_last:>8.2f}")
        print(f"  26 EMA: ${ema_long_last:>8.2f}")
    else:
        print("  Not enough data for EMA crossovers.")
    print("\n" + "-"*40)

    # --- Candlestick Patterns (basic) ---
    print("        CANDLESTICK PATTERNS        ")
    print("-"*40)
    def detect_hammer(row):
        # Hammer: small body, long lower shadow
        body = abs(row['Close'] - row['Open'])
        lower_shadow = row['Open'] - row['Low']
        upper_shadow = row['High'] - row['Close']
        return (body < (row['High'] - row['Low']) * 0.3) and (lower_shadow > body * 2)
    row = {col: data[col].iloc[-1].item() if hasattr(data[col].iloc[-1], 'item') else float(data[col].iloc[-1]) for col in ['Open', 'High', 'Low', 'Close']}
    if detect_hammer(row):
        print("  Hammer detected: Potential BUY signal.")
    else:
        print("  No hammer pattern detected.")
    print("\n" + "-"*40)

    # --- Enhanced Volume Analysis ---
    print("        VOLUME SPIKES        ")
    print("-"*40)
    vol = data['Volume']
    avg_vol = vol.rolling(window=20).mean()
    if len(vol) >= 20:
        last_vol = vol.iloc[-1].item() if hasattr(vol.iloc[-1], 'item') else float(vol.iloc[-1])
        last_avg_vol = avg_vol.iloc[-1].item() if hasattr(avg_vol.iloc[-1], 'item') else float(avg_vol.iloc[-1])
        if last_vol > last_avg_vol * 2:
            print("  Volume spike detected: Potential reversal or breakout.")
        else:
            print("  No significant volume spike.")
        print(f"  Last volume: {last_vol:,.0f}")
        print(f"  20-day avg volume: {last_avg_vol:,.0f}")
    else:
        print("  Not enough data for volume spike analysis.")
    print("\n" + "-"*40)
    future_X_high = np.arange(len(high_prices), len(high_prices) + 30).reshape(-1, 1)
    future_pred_high = model_high.predict(future_X_high)
    forecast_sell_value = np.mean(future_pred_high)
    print(f"  Estimated sell value for next 30 days:        ${forecast_sell_value:>8.2f}")
    print("="*40 + "\n")

    # --- AI/Deep Learning (LSTM) Prediction ---
    print("        AI/Deep Learning Prediction: BEST BUY/SELL VALUES        ")
    print("-"*40)
    # Calculate current buy and sell rates for comparison
    # Current buy: average of last 30 days' (high+low)/2
    try:
        high_mean = data['High'].tail(30).mean()
        if isinstance(high_mean, pd.Series):
            high_mean = high_mean.iloc[0]
        low_mean = data['Low'].tail(30).mean()
        if isinstance(low_mean, pd.Series):
            low_mean = low_mean.iloc[0]
        current_buy = float((high_mean + low_mean) / 2)
    except Exception:
        current_buy = None
    # Current sell: average of last 30 days' high
    try:
        high_mean = data['High'].tail(30).mean()
        if isinstance(high_mean, pd.Series):
            high_mean = high_mean.iloc[0]
        current_sell = float(high_mean)
    except Exception:
        current_sell = None

    try:
        import warnings
        warnings.filterwarnings("ignore")
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info/warning/error
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Input
        from sklearn.preprocessing import MinMaxScaler
        # Prepare close price data
        close_vals = data['Close'].dropna().values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_close = scaler.fit_transform(close_vals)
        seq_len = 10
        X_lstm, y_lstm = [], []
        for i in range(len(scaled_close) - seq_len):
            X_lstm.append(scaled_close[i:i+seq_len])
            y_lstm.append(scaled_close[i+seq_len])
        X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
        if len(X_lstm) > 0:
            model = Sequential([
                Input(shape=(seq_len, 1)),
                LSTM(32),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_lstm, y_lstm, epochs=10, batch_size=8, verbose=0)
            # Predict next 30 days
            last_seq = scaled_close[-seq_len:]
            preds = []
            seq = last_seq.copy()
            for _ in range(30):
                pred = model.predict(seq.reshape(1, seq_len, 1), verbose=0)
                preds.append(pred[0, 0])
                seq = np.roll(seq, -1)
                seq[-1] = pred[0, 0]
            preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
            best_buy_dl = np.min(preds)
            best_sell_dl = np.max(preds)
            # Display current buy/sell rates for comparison
            if current_buy is not None:
                print(f"  Current recommended buy value (last 30 days avg):  ${current_buy:>8.2f}")
            else:
                print("  Current recommended buy value: N/A")
            if current_sell is not None:
                print(f"  Current recommended sell value (last 30 days avg high): ${current_sell:>8.2f}")
            else:
                print("  Current recommended sell value: N/A")
            print(f"  AI (LSTM) predicted best buy value:  ${best_buy_dl:>8.2f}")
            print(f"  AI (LSTM) predicted best sell value: ${best_sell_dl:>8.2f}")

            # --- Enhanced AI Output ---
            print("\n  AI (LSTM) Daily Forecast Table (Next 30 Days):")
            print("  Day | Predicted Price | Direction | Prob(Up) | Volatility | Conf. Interval | Anomaly | Signal")
            # Calculate daily direction, probability, volatility, confidence intervals, anomaly, and signal
            prev_price = close_vals[-1][0] if len(close_vals) > 0 else preds[0]
            volatility = np.std(preds)
            for i, price in enumerate(preds):
                direction = "Up" if price > prev_price else "Down"
                prob_up = np.clip((price - prev_price) / (volatility + 1e-6) * 0.5 + 0.5, 0, 1)  # Simple proxy
                conf_int = (price - volatility, price + volatility)
                anomaly = "Yes" if abs(price - prev_price) > 2 * volatility else "No"
                # Simple buy/sell/hold signal
                if price == best_buy_dl:
                    signal = "BUY"
                elif price == best_sell_dl:
                    signal = "SELL"
                else:
                    signal = "HOLD"
                print(f"  {i+1:>3} | ${price:>13.2f} | {direction:>8} | {prob_up:>8.2f} | {volatility:>9.2f} | (${conf_int[0]:.2f}, ${conf_int[1]:.2f}) | {anomaly:>7} | {signal:>5}")
                prev_price = price
            print(f"\n  Volatility estimate for next 30 days: {volatility:.2f}")
            ci_low = np.mean(preds) - volatility
            ci_high = np.mean(preds) + volatility
            print(f"  Expected price range for next 30 days: ${ci_low:.2f} to ${ci_high:.2f} (most predictions fall in this range)")
            print("  Anomaly detection: Days with price change > 2x volatility flagged as 'Yes'.")
            print("  Signal: BUY/SELL/HOLD based on predicted min/max price.")
        else:
            print("  Not enough data for AI/Deep Learning prediction.")
    except Exception as e:
        print(f"  Deep Learning prediction failed: {e}")
    print("="*40 + "\n")

# Prompt user for stock symbol
if __name__ == "__main__":
    symbol = input("Enter the stock symbol to look up: ").strip().upper()
    find_best_buying_day(symbol)  # No output or return value