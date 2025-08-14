## Stock Trend Prediction Toolkit

This repository provides a set of Python scripts for stock analysis, trend prediction, and AI-driven forecasting. The tools leverage machine learning and deep learning to generate buy/sell signals and 30-day forecasts for any stock symbol.

### Scripts Overview

#### `ai_estimate.py`
Performs AI-driven stock trend prediction and 30-day price forecasting using Random Forest algorithms.

- **Features:**
  - Fetches historical stock data (via yfinance)
  - Engineers features (returns, moving averages)
  - Trains a Random Forest Classifier to predict next-day trend (up/down)
  - Provides buy/sell suggestions based on trend
  - Forecasts 30 days of closing prices with suggested buy/sell prices
- **Usage:**
  - Run directly: `python ai_estimate.py`
  - Enter a stock symbol when prompted
  - View AI-driven buy/sell suggestions and a 30-day price forecast

#### `stock_estimate.py`
Comprehensive stock analysis and forecasting tool with technical indicators and AI (LSTM) predictions.

- **Features:**
  - Calculates moving averages, support/resistance, volume at extremes
  - Computes technical indicators: MACD, Stochastic Oscillator, Bollinger Bands, RSI, EMA Crossovers, Candlestick Patterns, Volume Spikes
  - Provides recommended buy/sell values for various periods
  - Forecasts best buy/sell values for the next 30 days (linear regression)
  - AI/Deep Learning (LSTM) prediction for next 30 days with daily forecast table, volatility, confidence intervals, anomaly detection, and buy/sell/hold signals
- **Usage:**
  - Run directly: `python stock_estimate.py`
  - Enter a stock symbol when prompted
  - View detailed technical and AI/Deep Learning analysis in the terminal

#### `main.py`
Acts as the central orchestrator, combining the core machine learning workflow from `ai_estimate.py` with the ability to integrate advanced analytics from `stock_estimate.py`.

- **How it differs from the other scripts:**
  - **Integration:** Imports and utilizes functions from both `ai_estimate.py` (feature engineering, ML model training, trend prediction) and `stock_estimate.py` (advanced analytics, technical indicators).
  - **Workflow:** Automates the end-to-end process: prompts for a stock symbol, fetches and processes data, trains models, and generates a comprehensive 30-day forecast table.
  - **Output:** Presents a clear, tabular summary of predicted prices, trend direction, and buy/sell/hold signals for each of the next 30 trading days.
  - **Customization:** Designed for easy extension—can be modified to include more advanced analytics or to call additional functions from the other scripts.
  - **Comparison:**
    - `ai_estimate.py` is focused on a single-script workflow for AI-based trend and price prediction, with direct user interaction and output.
    - `stock_estimate.py` is a comprehensive technical and deep learning analysis tool, providing detailed technical indicators and LSTM-based forecasts.
    - `main.py` brings these approaches together, providing a streamlined, user-friendly interface for rapid forecasting and signal generation, and can serve as a template for further integration or automation.
- **Usage:**
  - Run directly: `python main.py`
  - Enter a stock symbol when prompted
  - View a 30-day forecast table with buy/sell signals

### Requirements
- Python 3.8+
- Install dependencies:
  ```sh
  pip install yfinance pandas numpy scikit-learn tensorflow
  ```

### WINDOWS LONG PATH SUPPORT (if needed):
- Open Start menu, type "regedit" (Registry Editor)
- Navigate to: HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
- Set LongPathsEnabled to 1, then restart your computer

### Notes
- For best results, ensure all required libraries are installed and up to date.
- For advanced predictions, consider experimenting with ensemble models, feature engineering, or backtesting.
