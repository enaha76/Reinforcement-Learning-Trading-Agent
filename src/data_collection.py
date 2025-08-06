"""
Data Collection Script for RL Trading Agent
Downloads SPY and GLD historical data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, date
import os

def download_stock_data(tickers, start_date="2005-01-01", end_date=None):
    """
    Download historical stock data for given tickers
    
    Args:
        tickers (list): List of stock tickers
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format (None for today)
    
    Returns:
        pd.DataFrame: Combined dataframe with stock data
    """
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")
    
    print(f"Downloading data from {start_date} to {end_date}")
    
    # Download data for all tickers
    data_frames = []
    
    for ticker in tickers:
        print(f"Downloading {ticker}...")
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        
        # Add ticker column
        hist['Ticker'] = ticker
        hist.reset_index(inplace=True)
        
        # Rename columns to be more descriptive
        hist.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Ticker']
        
        data_frames.append(hist)
        print(f"Downloaded {len(hist)} records for {ticker}")
    
    # Combine all data
    combined_data = pd.concat(data_frames, ignore_index=True)
    
    # Sort by date and ticker
    combined_data = combined_data.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    
    return combined_data

def main():
    """Main function to download and save stock data"""
    
    # Define the tickers we want
    tickers = ['SPY', 'GLD']
    
    # Create data directory if it doesn't exist
    data_dir = '../data'
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # Download the data
        stock_data = download_stock_data(tickers)
        
        # Save to CSV
        output_path = os.path.join(data_dir, 'raw_data.csv')
        stock_data.to_csv(output_path, index=False)
        
        print(f"\nData successfully saved to {output_path}")
        print(f"Total records: {len(stock_data)}")
        print(f"Date range: {stock_data['Date'].min()} to {stock_data['Date'].max()}")
        print(f"Tickers: {stock_data['Ticker'].unique()}")
        
        # Display basic info
        print("\nData Summary:")
        print(stock_data.groupby('Ticker').agg({
            'Date': ['min', 'max', 'count'],
            'Close': ['min', 'max', 'mean']
        }))
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
