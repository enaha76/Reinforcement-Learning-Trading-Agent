"""
Simple Trading Simulation
Demonstrates the trading logic without full RL dependencies
Uses basic Python to implement trading strategies and comparison
"""

import csv
import math
import os
from datetime import datetime

def load_data(filename):
    """Load processed data from CSV"""
    
    filepath = f'../data/{filename}'
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []
    
    data = []
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert numeric fields
            for key, value in row.items():
                if value and value != 'None':
                    try:
                        row[key] = float(value)
                    except ValueError:
                        pass  # Keep as string (like Date)
                else:
                    row[key] = None
            data.append(row)
    
    return data

class SimpleTradingStrategy:
    """Simple rule-based trading strategy for comparison"""
    
    def __init__(self, initial_balance=100000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.spy_allocation = 0.5
        self.gld_allocation = 0.5
        self.portfolio_values = []
        self.allocations = []
        self.returns = []
    
    def reset(self):
        """Reset strategy to initial state"""
        self.balance = self.initial_balance
        self.spy_allocation = 0.5
        self.gld_allocation = 0.5
        self.portfolio_values = [self.initial_balance]
        self.allocations = [(0.5, 0.5)]
        self.returns = []
    
    def make_decision(self, data_point):
        """
        Make allocation decision based on technical indicators
        Simple momentum + mean reversion strategy
        """
        
        # Get technical indicators
        spy_rsi = data_point.get('RSI_SPY')
        gld_rsi = data_point.get('RSI_GLD')
        spy_macd = data_point.get('MACD_SPY')
        spy_macd_signal = data_point.get('MACD_Signal_SPY')
        spy_volatility = data_point.get('Volatility_SPY', 0.2)
        gld_volatility = data_point.get('Volatility_GLD', 0.15)
        
        # Default allocation
        new_spy_allocation = 0.5
        
        # RSI-based mean reversion
        if spy_rsi is not None and gld_rsi is not None:
            if spy_rsi < 30:  # SPY oversold, increase allocation
                new_spy_allocation += 0.2
            elif spy_rsi > 70:  # SPY overbought, decrease allocation
                new_spy_allocation -= 0.2
            
            if gld_rsi < 30:  # GLD oversold, decrease SPY allocation
                new_spy_allocation -= 0.1
            elif gld_rsi > 70:  # GLD overbought, increase SPY allocation
                new_spy_allocation += 0.1
        
        # MACD momentum
        if spy_macd is not None and spy_macd_signal is not None:
            if spy_macd > spy_macd_signal:  # Bullish momentum
                new_spy_allocation += 0.1
            else:  # Bearish momentum
                new_spy_allocation -= 0.1
        
        # Volatility adjustment (reduce allocation when volatility is high)
        if spy_volatility > 0.25:  # High volatility
            new_spy_allocation -= 0.1
        
        # Constrain allocation between 0.1 and 0.9
        new_spy_allocation = max(0.1, min(0.9, new_spy_allocation))
        
        return new_spy_allocation
    
    def step(self, data_point):
        """Execute one trading step"""
        
        # Get current prices and returns
        spy_return = data_point.get('Returns_SPY', 0)
        gld_return = data_point.get('Returns_GLD', 0)
        
        if spy_return is None:
            spy_return = 0
        if gld_return is None:
            gld_return = 0
        
        # Calculate portfolio return with current allocation
        portfolio_return = (self.spy_allocation * spy_return + 
                          self.gld_allocation * gld_return)
        
        # Update balance
        self.balance *= (1 + portfolio_return)
        
        # Make new allocation decision
        new_spy_allocation = self.make_decision(data_point)
        
        # Calculate transaction cost (0.1% of rebalanced amount)
        allocation_change = abs(new_spy_allocation - self.spy_allocation)
        transaction_cost = allocation_change * 0.001 * self.balance
        self.balance -= transaction_cost
        
        # Update allocations
        self.spy_allocation = new_spy_allocation
        self.gld_allocation = 1.0 - new_spy_allocation
        
        # Record results
        self.portfolio_values.append(self.balance)
        self.allocations.append((self.spy_allocation, self.gld_allocation))
        self.returns.append(portfolio_return)

class BuyAndHoldStrategy:
    """Buy and Hold SPY strategy for benchmark"""
    
    def __init__(self, initial_balance=100000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.portfolio_values = []
        self.returns = []
    
    def reset(self):
        """Reset strategy to initial state"""
        self.balance = self.initial_balance
        self.portfolio_values = [self.initial_balance]
        self.returns = []
    
    def step(self, data_point):
        """Execute one trading step"""
        
        # Get SPY return
        spy_return = data_point.get('Returns_SPY', 0)
        if spy_return is None:
            spy_return = 0
        
        # Update balance (100% SPY allocation)
        self.balance *= (1 + spy_return)
        
        # Record results
        self.portfolio_values.append(self.balance)
        self.returns.append(spy_return)

def calculate_performance_metrics(returns, portfolio_values):
    """Calculate performance metrics"""
    
    if not returns or len(returns) == 0:
        return {}
    
    # Filter out None values
    valid_returns = [r for r in returns if r is not None]
    
    if len(valid_returns) == 0:
        return {}
    
    # Basic metrics
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    annualized_return = (1 + total_return) ** (252 / len(valid_returns)) - 1
    
    # Volatility
    mean_return = sum(valid_returns) / len(valid_returns)
    variance = sum((r - mean_return) ** 2 for r in valid_returns) / len(valid_returns)
    volatility = math.sqrt(variance) * math.sqrt(252)
    
    # Sharpe ratio (assuming 2% risk-free rate)
    risk_free_rate = 0.02
    daily_rf = risk_free_rate / 252
    excess_returns = [r - daily_rf for r in valid_returns]
    mean_excess = sum(excess_returns) / len(excess_returns)
    sharpe_ratio = mean_excess / math.sqrt(variance) * math.sqrt(252) if variance > 0 else 0
    
    # Maximum drawdown
    peak = portfolio_values[0]
    max_drawdown = 0
    for value in portfolio_values[1:]:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    # Win rate
    win_rate = sum(1 for r in valid_returns if r > 0) / len(valid_returns)
    
    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Final Value': portfolio_values[-1]
    }

def run_simulation():
    """Run the trading simulation"""
    
    print("=== Simple Trading Simulation ===")
    
    # Load test data
    test_data = load_data('test_data.csv')
    if not test_data:
        print("No test data found. Please run simple_preprocessing.py first.")
        return False
    
    print(f"Running simulation on {len(test_data)} test data points")
    print(f"Test period: {test_data[0]['Date']} to {test_data[-1]['Date']}")
    
    # Initialize strategies
    smart_strategy = SimpleTradingStrategy()
    benchmark_strategy = BuyAndHoldStrategy()
    
    # Run simulation
    print("\nRunning strategies...")
    
    for i, data_point in enumerate(test_data):
        smart_strategy.step(data_point)
        benchmark_strategy.step(data_point)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(test_data)} days")
    
    # Calculate performance metrics
    smart_metrics = calculate_performance_metrics(
        smart_strategy.returns, smart_strategy.portfolio_values
    )
    benchmark_metrics = calculate_performance_metrics(
        benchmark_strategy.returns, benchmark_strategy.portfolio_values
    )
    
    # Print results
    print("\n" + "="*70)
    print("SIMULATION RESULTS")
    print("="*70)
    
    print(f"{'Metric':<25} {'Smart Strategy':<20} {'Buy & Hold':<20} {'Difference':<15}")
    print("-"*80)
    
    for metric in smart_metrics.keys():
        if metric in benchmark_metrics:
            smart_val = smart_metrics[metric]
            benchmark_val = benchmark_metrics[metric]
            diff = smart_val - benchmark_val
            
            if 'Return' in metric or 'Ratio' in metric or 'Rate' in metric:
                print(f"{metric:<25} {smart_val:<20.2%} {benchmark_val:<20.2%} {diff:<15.2%}")
            elif 'Drawdown' in metric:
                print(f"{metric:<25} {smart_val:<20.2%} {benchmark_val:<20.2%} {diff:<15.2%}")
            elif 'Value' in metric:
                print(f"{metric:<25} ${smart_val:<19,.2f} ${benchmark_val:<19,.2f} ${diff:<14,.2f}")
            else:
                print(f"{metric:<25} {smart_val:<20.4f} {benchmark_val:<20.4f} {diff:<15.4f}")
    
    # Project objectives analysis
    print("\n" + "="*70)
    print("PROJECT OBJECTIVES ANALYSIS")
    print("="*70)
    
    # Performance improvement
    return_improvement = (smart_metrics['Annualized Return'] - 
                         benchmark_metrics['Annualized Return'])
    relative_improvement = (return_improvement / 
                           benchmark_metrics['Annualized Return'] 
                           if benchmark_metrics['Annualized Return'] != 0 else 0)
    
    print(f"Annualized Return Improvement: {return_improvement:.2%}")
    print(f"Relative Improvement: {relative_improvement:.2%}")
    print(f"Target: 20% better performance - {'✓ ACHIEVED' if relative_improvement >= 0.20 else '✗ NOT ACHIEVED'}")
    
    # Final value improvement
    value_improvement = ((smart_metrics['Final Value'] - benchmark_metrics['Final Value']) / 
                        benchmark_metrics['Final Value'])
    print(f"Final Value Improvement: {value_improvement:.2%}")
    print(f"Target: 50% more final value - {'✓ ACHIEVED' if value_improvement >= 0.50 else '✗ NOT ACHIEVED'}")
    
    # Risk-adjusted performance
    sharpe_improvement = (smart_metrics['Sharpe Ratio'] - benchmark_metrics['Sharpe Ratio'])
    print(f"Sharpe Ratio Improvement: {sharpe_improvement:.4f}")
    print(f"Better Risk-Adjusted Returns: {'✓ YES' if sharpe_improvement > 0 else '✗ NO'}")
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETED!")
    print("This demonstrates the core trading logic that would be learned by the RL agent.")
    print("="*70)
    
    return True

def main():
    """Main function"""
    return run_simulation()

if __name__ == "__main__":
    main()
