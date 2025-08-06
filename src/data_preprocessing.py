"""
Data Preprocessing Script for RL Trading Agent
Processes raw stock data and calculates technical indicators
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import os
from sklearn.preprocessing import StandardScaler

def calculate_technical_indicators(df, ticker_col='Ticker'):
    """
    Calculate technical indicators for stock data
    
    Args:
        df (pd.DataFrame): Stock data with OHLCV columns
        ticker_col (str): Name of ticker column
    
    Returns:
        pd.DataFrame: DataFrame with technical indicators added
    """
    
    processed_data = []
    
    for ticker in df[ticker_col].unique():
        ticker_data = df[df[ticker_col] == ticker].copy()
        ticker_data = ticker_data.sort_values('Date').reset_index(drop=True)
        
        print(f"Processing technical indicators for {ticker}...")
        
        # Calculate returns
        ticker_data['Returns'] = ticker_data['Close'].pct_change()
        ticker_data['Log_Returns'] = np.log(ticker_data['Close'] / ticker_data['Close'].shift(1))
        
        # Moving averages
        ticker_data['SMA_10'] = ticker_data['Close'].rolling(window=10).mean()
        ticker_data['SMA_30'] = ticker_data['Close'].rolling(window=30).mean()
        ticker_data['EMA_10'] = ticker_data['Close'].ewm(span=10).mean()
        ticker_data['EMA_30'] = ticker_data['Close'].ewm(span=30).mean()
        
        # MACD
        macd_data = ta.macd(ticker_data['Close'])
        ticker_data['MACD'] = macd_data['MACD_12_26_9']
        ticker_data['MACD_Signal'] = macd_data['MACDs_12_26_9']
        ticker_data['MACD_Histogram'] = macd_data['MACDh_12_26_9']
        
        # RSI
        ticker_data['RSI'] = ta.rsi(ticker_data['Close'], length=14)
        
        # ATR (Average True Range)
        ticker_data['ATR'] = ta.atr(ticker_data['High'], ticker_data['Low'], ticker_data['Close'], length=14)
        
        # Bollinger Bands
        bb_data = ta.bbands(ticker_data['Close'], length=20)
        ticker_data['BB_Upper'] = bb_data['BBU_20_2.0']
        ticker_data['BB_Middle'] = bb_data['BBM_20_2.0']
        ticker_data['BB_Lower'] = bb_data['BBL_20_2.0']
        ticker_data['BB_Width'] = (ticker_data['BB_Upper'] - ticker_data['BB_Lower']) / ticker_data['BB_Middle']
        
        # Volume indicators
        ticker_data['Volume_SMA'] = ticker_data['Volume'].rolling(window=20).mean()
        ticker_data['Volume_Ratio'] = ticker_data['Volume'] / ticker_data['Volume_SMA']
        
        # Volatility (rolling standard deviation of returns)
        ticker_data['Volatility'] = ticker_data['Returns'].rolling(window=20).std()
        
        processed_data.append(ticker_data)
    
    return pd.concat(processed_data, ignore_index=True)

def create_pivot_features(df):
    """
    Create pivot table features for multi-asset analysis
    
    Args:
        df (pd.DataFrame): Processed stock data
    
    Returns:
        pd.DataFrame: DataFrame with pivot features
    """
    
    # Select key columns for pivoting
    pivot_cols = ['Close', 'Returns', 'RSI', 'MACD', 'ATR', 'Volatility']
    
    pivot_data = df[['Date', 'Ticker'] + pivot_cols].copy()
    
    # Create pivot tables for each feature
    pivot_features = {}
    
    for col in pivot_cols:
        pivot_table = pivot_data.pivot(index='Date', columns='Ticker', values=col)
        
        # Rename columns to include feature name
        pivot_table.columns = [f'{col}_{ticker}' for ticker in pivot_table.columns]
        pivot_features[col] = pivot_table
    
    # Combine all pivot features
    combined_pivot = pd.concat(pivot_features.values(), axis=1)
    combined_pivot.reset_index(inplace=True)
    
    return combined_pivot

def split_data(df, train_ratio=0.75):
    """
    Split data into training and testing sets
    
    Args:
        df (pd.DataFrame): Processed data
        train_ratio (float): Ratio of data to use for training
    
    Returns:
        tuple: (train_data, test_data)
    """
    
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_ratio)
    
    train_data = df_sorted[:split_idx].copy()
    test_data = df_sorted[split_idx:].copy()
    
    print(f"Training data: {len(train_data)} records ({train_data['Date'].min()} to {train_data['Date'].max()})")
    print(f"Testing data: {len(test_data)} records ({test_data['Date'].min()} to {test_data['Date'].max()})")
    
    return train_data, test_data

def main():
    """Main preprocessing function"""
    
    # Load raw data
    raw_data_path = '../data/raw_data.csv'
    
    if not os.path.exists(raw_data_path):
        print(f"Raw data file not found: {raw_data_path}")
        print("Please run data_collection.py first")
        return False
    
    print("Loading raw data...")
    raw_data = pd.read_csv(raw_data_path)
    raw_data['Date'] = pd.to_datetime(raw_data['Date'])
    
    print(f"Loaded {len(raw_data)} records")
    
    # Calculate technical indicators
    print("\nCalculating technical indicators...")
    processed_data = calculate_technical_indicators(raw_data)
    
    # Create pivot features for multi-asset analysis
    print("\nCreating pivot features...")
    pivot_data = create_pivot_features(processed_data)
    
    # Remove rows with NaN values (due to technical indicators)
    print(f"Data shape before cleaning: {pivot_data.shape}")
    pivot_data = pivot_data.dropna()
    print(f"Data shape after cleaning: {pivot_data.shape}")
    
    # Split into training and testing sets
    print("\nSplitting data...")
    train_data, test_data = split_data(pivot_data)
    
    # Save processed data
    print("\nSaving processed data...")
    pivot_data.to_csv('../data/processed_data.csv', index=False)
    train_data.to_csv('../data/train_data.csv', index=False)
    test_data.to_csv('../data/test_data.csv', index=False)
    
    print("Data preprocessing completed successfully!")
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print(f"Total processed records: {len(pivot_data)}")
    print(f"Training records: {len(train_data)}")
    print(f"Testing records: {len(test_data)}")
    print(f"Features: {pivot_data.shape[1]}")
    
    return True

if __name__ == "__main__":
    main()
