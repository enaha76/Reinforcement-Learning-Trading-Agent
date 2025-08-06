"""
Simplified Data Preprocessing Script
Uses basic Python libraries to calculate technical indicators and prepare data
"""

import csv
import math
import os
from datetime import datetime

def read_csv_data(filename):
    """Read CSV data into list of dictionaries"""
    
    filepath = f'../data/{filename}'
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []
    
    records = []
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            records.append(row)
    
    print(f"Loaded {len(records)} records from {filename}")
    return records

def calculate_sma(prices, window):
    """Calculate Simple Moving Average"""
    sma = []
    for i in range(len(prices)):
        if i < window - 1:
            sma.append(None)
        else:
            avg = sum(prices[i-window+1:i+1]) / window
            sma.append(avg)
    return sma

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average"""
    ema = []
    multiplier = 2 / (window + 1)
    
    for i, price in enumerate(prices):
        if i == 0:
            ema.append(price)
        else:
            ema_val = (price * multiplier) + (ema[i-1] * (1 - multiplier))
            ema.append(ema_val)
    
    return ema

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    rsi = []
    gains = []
    losses = []
    
    # Calculate price changes
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    
    # Calculate RSI
    for i in range(len(gains)):
        if i < window - 1:
            rsi.append(None)
        else:
            avg_gain = sum(gains[i-window+1:i+1]) / window
            avg_loss = sum(losses[i-window+1:i+1]) / window
            
            if avg_loss == 0:
                rsi.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi_val = 100 - (100 / (1 + rs))
                rsi.append(rsi_val)
    
    # Add None for first price (no change calculated)
    return [None] + rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    
    # Calculate MACD line
    macd_line = []
    for i in range(len(prices)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line.append(ema_fast[i] - ema_slow[i])
        else:
            macd_line.append(None)
    
    # Calculate signal line (EMA of MACD)
    macd_values = [x for x in macd_line if x is not None]
    if len(macd_values) >= signal:
        signal_line = calculate_ema(macd_values, signal)
        # Pad with None values to match original length
        full_signal = [None] * (len(macd_line) - len(signal_line)) + signal_line
    else:
        full_signal = [None] * len(macd_line)
    
    return macd_line, full_signal

def calculate_volatility(returns, window=20):
    """Calculate rolling volatility"""
    volatility = []
    
    for i in range(len(returns)):
        if i < window - 1 or returns[i] is None:
            volatility.append(None)
        else:
            # Get window of returns, excluding None values
            window_returns = [r for r in returns[i-window+1:i+1] if r is not None]
            
            if len(window_returns) >= window // 2:  # At least half the window
                mean_return = sum(window_returns) / len(window_returns)
                variance = sum((r - mean_return) ** 2 for r in window_returns) / len(window_returns)
                vol = math.sqrt(variance) * math.sqrt(252)  # Annualized
                volatility.append(vol)
            else:
                volatility.append(None)
    
    return volatility

def process_symbol_data(records, symbol):
    """Process data for a single symbol"""
    
    # Filter records for this symbol and sort by date
    symbol_records = [r for r in records if r['Symbol'] == symbol]
    symbol_records.sort(key=lambda x: x['Date'])
    
    print(f"Processing {len(symbol_records)} records for {symbol}")
    
    # Extract prices and calculate returns
    dates = [r['Date'] for r in symbol_records]
    closes = [float(r['Close']) for r in symbol_records]
    
    # Calculate returns
    returns = [None]  # First day has no return
    for i in range(1, len(closes)):
        ret = (closes[i] - closes[i-1]) / closes[i-1]
        returns.append(ret)
    
    # Calculate technical indicators
    sma_10 = calculate_sma(closes, 10)
    sma_30 = calculate_sma(closes, 30)
    ema_10 = calculate_ema(closes, 10)
    ema_30 = calculate_ema(closes, 30)
    rsi = calculate_rsi(closes)
    macd, macd_signal = calculate_macd(closes)
    volatility = calculate_volatility(returns)
    
    # Create processed records
    processed = []
    for i in range(len(symbol_records)):
        record = {
            'Date': dates[i],
            'Symbol': symbol,
            'Close': closes[i],
            'Returns': returns[i],
            'SMA_10': sma_10[i],
            'SMA_30': sma_30[i],
            'EMA_10': ema_10[i],
            'EMA_30': ema_30[i],
            'RSI': rsi[i],
            'MACD': macd[i],
            'MACD_Signal': macd_signal[i],
            'Volatility': volatility[i]
        }
        processed.append(record)
    
    return processed

def create_pivot_data(spy_data, gld_data):
    """Create pivot table with both symbols"""
    
    # Create date-indexed data
    pivot_data = {}
    
    # Add SPY data
    for record in spy_data:
        date = record['Date']
        if date not in pivot_data:
            pivot_data[date] = {'Date': date}
        
        for key, value in record.items():
            if key not in ['Date', 'Symbol']:
                pivot_data[date][f'{key}_SPY'] = value
    
    # Add GLD data
    for record in gld_data:
        date = record['Date']
        if date not in pivot_data:
            pivot_data[date] = {'Date': date}
        
        for key, value in record.items():
            if key not in ['Date', 'Symbol']:
                pivot_data[date][f'{key}_GLD'] = value
    
    # Convert to list and sort by date
    pivot_list = list(pivot_data.values())
    pivot_list.sort(key=lambda x: x['Date'])
    
    # Remove rows with too many None values
    clean_data = []
    for record in pivot_list:
        none_count = sum(1 for v in record.values() if v is None)
        if none_count < len(record) * 0.5:  # Less than 50% None values
            clean_data.append(record)
    
    print(f"Created pivot data with {len(clean_data)} clean records")
    return clean_data

def split_data(data, train_ratio=0.75):
    """Split data into train and test sets"""
    
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    print(f"Training data: {len(train_data)} records")
    print(f"Testing data: {len(test_data)} records")
    
    return train_data, test_data

def save_processed_data(data, filename):
    """Save processed data to CSV"""
    
    if not data:
        print("No data to save")
        return False
    
    filepath = f'../data/{filename}'
    
    # Get all possible headers
    headers = set()
    for record in data:
        headers.update(record.keys())
    
    headers = sorted(list(headers))
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        for record in data:
            # Fill missing keys with None
            complete_record = {h: record.get(h, None) for h in headers}
            writer.writerow(complete_record)
    
    print(f"Saved {len(data)} records to {filepath}")
    return True

def main():
    """Main preprocessing function"""
    
    print("=== Simple Data Preprocessing ===")
    
    # Load raw data
    raw_data = read_csv_data('raw_data.csv')
    if not raw_data:
        print("No raw data found. Please run simple_data_collection.py first.")
        return False
    
    # Process each symbol
    spy_data = process_symbol_data(raw_data, 'SPY')
    gld_data = process_symbol_data(raw_data, 'GLD')
    
    if not spy_data or not gld_data:
        print("Failed to process symbol data")
        return False
    
    # Create pivot data
    pivot_data = create_pivot_data(spy_data, gld_data)
    
    if not pivot_data:
        print("Failed to create pivot data")
        return False
    
    # Split data
    train_data, test_data = split_data(pivot_data)
    
    # Save processed data
    save_processed_data(pivot_data, 'processed_data.csv')
    save_processed_data(train_data, 'train_data.csv')
    save_processed_data(test_data, 'test_data.csv')
    
    print("\nPreprocessing completed successfully!")
    print(f"Processed data shape: {len(pivot_data)} rows x {len(pivot_data[0]) if pivot_data else 0} columns")
    print(f"Date range: {pivot_data[0]['Date']} to {pivot_data[-1]['Date']}")
    
    # Show sample of features
    if pivot_data:
        print("\nSample features:")
        sample = pivot_data[len(pivot_data)//2]  # Middle record
        for key, value in sample.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    
    return True

if __name__ == "__main__":
    main()
