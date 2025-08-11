import os
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from sklearn.model_selection import train_test_split

# --- Configuration ---
# Define the base path relative to the current script location
# This ensures that all paths are relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw_data.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.csv')
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_data.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_data.csv')

def fetch_data(tickers, start_date, end_date):
    """
    Fetches historical market data for the given tickers from Yahoo Finance.

    Args:
        tickers (list): A list of ticker symbols.
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame containing the raw OHLCV data.
    """
    print(f"Fetching data for {tickers} from {start_date} to {end_date}...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        # Forward-fill and then back-fill to handle missing values
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        print("Data fetching complete.")
        return data
    except Exception as e:
        print(f"An error occurred during data fetching: {e}")
        return pd.DataFrame()

def engineer_features(raw_data, primary_ticker='SPY', secondary_ticker='GLD'):
    """
    Engineers technical indicators and relational features from the raw data.

    Args:
        raw_data (pd.DataFrame): The raw OHLCV data.
        primary_ticker (str): The primary asset ticker.
        secondary_ticker (str): The secondary asset ticker.

    Returns:
        pd.DataFrame: A DataFrame with engineered features.
    """
    print("Engineering features...")
    df = pd.DataFrame()
    
    # Isolate columns for each ticker
    spy_cols = {col: f"{col}_{primary_ticker}" for col in ['Open', 'High', 'Low', 'Close', 'Volume']}
    gld_cols = {col: f"{col}_{secondary_ticker}" for col in ['Open', 'High', 'Low', 'Close', 'Volume']}

    df_spy = raw_data['Close'][[primary_ticker]].rename(columns={primary_ticker: f'Close_{primary_ticker}'})
    df_gld = raw_data['Close'][[secondary_ticker]].rename(columns={secondary_ticker: f'Close_{secondary_ticker}'})
    
    # Combine close prices into a single DataFrame
    df = pd.concat([df_spy, df_gld], axis=1)

    # Calculate returns for both assets
    df[f'Return_{primary_ticker}'] = df[f'Close_{primary_ticker}'].pct_change()
    df[f'Return_{secondary_ticker}'] = df[f'Close_{secondary_ticker}'].pct_change()

    # --- Technical Indicators using pandas-ta ---
    # SPY indicators
    df.ta.rsi(close=df[f'Close_{primary_ticker}'], length=14, append=True, col_names=(f'RSI_{primary_ticker}',))
    df.ta.macd(close=df[f'Close_{primary_ticker}'], length=12, append=True, col_names=(f'MACD_{primary_ticker}', f'MACDh_{primary_ticker}', f'MACDs_{primary_ticker}'))
    
    # GLD indicators
    df.ta.rsi(close=df[f'Close_{secondary_ticker}'], length=14, append=True, col_names=(f'RSI_{secondary_ticker}',))
    df.ta.macd(close=df[f'Close_{secondary_ticker}'], length=12, append=True, col_names=(f'MACD_{secondary_ticker}', f'MACDh_{secondary_ticker}', f'MACDs_{secondary_ticker}'))

    # Relational feature: Price Ratio
    df['Price_Ratio_SPY_GLD'] = df[f'Close_{primary_ticker}'] / df[f'Close_{secondary_ticker}']

    # Clean up by dropping initial rows with NaN values from indicator calculations
    df.dropna(inplace=True)
    
    print("Feature engineering complete.")
    return df

def main():
    """Main data preprocessing pipeline."""
    print("=== Data Preprocessing Pipeline ===")
    
    # 1. Fetch Data
    raw_data = fetch_data(tickers=['SPY', 'GLD'], start_date='2005-01-01', end_date='2022-12-31')
    
    if raw_data.empty:
        print("Halting pipeline due to data fetching failure.")
        return

    # Save the raw data for reference
    raw_data.to_csv(RAW_DATA_PATH)
    print(f"Raw data saved to {RAW_DATA_PATH}")

    # 2. Engineer Features
    processed_df = engineer_features(raw_data)

    # Save fully processed data
    processed_df.to_csv(PROCESSED_DATA_PATH)
    print(f"Processed data with all features saved to {PROCESSED_DATA_PATH}")

    # 3. Split Data into Training and Testing sets
    # Using a standard 80/20 split
    train_df, test_df = train_test_split(processed_df, test_size=0.2, shuffle=False)
    
    # Save the split datasets
    train_df.to_csv(TRAIN_DATA_PATH)
    test_df.to_csv(TEST_DATA_PATH)

    print("\n--- Pipeline Summary ---")
    print(f"Total records processed: {len(processed_df)}")
    print(f"Training data shape: {train_df.shape} (from {train_df.index.min().date()} to {train_df.index.max().date()})")
    print(f"Testing data shape: {test_df.shape} (from {test_df.index.min().date()} to {test_df.index.max().date()})")
    print(f"Training data saved to {TRAIN_DATA_PATH}")
    print(f"Testing data saved to {TEST_DATA_PATH}")
    print("\n✅ Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()