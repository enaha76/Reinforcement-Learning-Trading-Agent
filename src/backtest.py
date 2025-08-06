"""
Backtesting Script for RL Trading Agent
Evaluates trained agent performance against Buy-and-Hold benchmark
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from environment import create_environment
import warnings
warnings.filterwarnings('ignore')

class PortfolioAnalyzer:
    """Class for analyzing portfolio performance"""
    
    def __init__(self, returns, benchmark_returns=None):
        """
        Initialize analyzer with returns data
        
        Args:
            returns (list/array): Portfolio returns
            benchmark_returns (list/array): Benchmark returns (optional)
        """
        self.returns = np.array(returns)
        self.benchmark_returns = np.array(benchmark_returns) if benchmark_returns is not None else None
        
    def calculate_metrics(self, risk_free_rate=0.02):
        """
        Calculate portfolio performance metrics
        
        Args:
            risk_free_rate (float): Annual risk-free rate
            
        Returns:
            dict: Dictionary of performance metrics
        """
        if len(self.returns) == 0:
            return {}
        
        # Convert to numpy array
        returns = np.array(self.returns)
        
        # Basic metrics
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio
        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        win_rate = np.sum(returns > 0) / len(returns)
        
        metrics = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Total Days': len(returns)
        }
        
        return metrics

def run_buy_and_hold_benchmark(test_data, initial_balance=100000):
    """
    Run Buy-and-Hold SPY benchmark strategy
    
    Args:
        test_data (pd.DataFrame): Test data with price information
        initial_balance (float): Starting balance
        
    Returns:
        tuple: (portfolio_values, returns, allocations)
    """
    
    spy_prices = test_data['Close_SPY'].values
    portfolio_values = [initial_balance]
    returns = []
    allocations = [(1.0, 0.0)]  # 100% SPY, 0% GLD
    
    # Calculate daily returns and portfolio values
    for i in range(1, len(spy_prices)):
        daily_return = (spy_prices[i] - spy_prices[i-1]) / spy_prices[i-1]
        new_value = portfolio_values[-1] * (1 + daily_return)
        
        portfolio_values.append(new_value)
        returns.append(daily_return)
        allocations.append((1.0, 0.0))
    
    return portfolio_values, returns, allocations

def run_agent_backtest(model_path, test_data_path, initial_balance=100000):
    """
    Run backtest with trained RL agent
    
    Args:
        model_path (str): Path to trained model
        test_data_path (str): Path to test data
        initial_balance (float): Starting balance
        
    Returns:
        tuple: (portfolio_values, returns, allocations, actions)
    """
    
    # Load trained model
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    # Create test environment
    print("Creating test environment...")
    env = create_environment(test_data_path, initial_balance=initial_balance)
    
    # Run backtest
    print("Running agent backtest...")
    obs = env.reset()
    portfolio_values = [initial_balance]
    returns = []
    allocations = []
    actions = []
    
    done = False
    step = 0
    
    while not done:
        # Get action from trained agent
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action[0])
        
        # Take step in environment
        obs, reward, done, info = env.step(action)
        
        # Record results
        portfolio_values.append(info['balance'])
        if 'portfolio_return' in info:
            returns.append(info['portfolio_return'])
        allocations.append((info['spy_allocation'], info['gld_allocation']))
        
        step += 1
        
        if step % 100 == 0:
            print(f"Step {step}, Balance: ${info['balance']:.2f}, SPY: {info['spy_allocation']:.2%}")
    
    print(f"Backtest completed. Final balance: ${portfolio_values[-1]:.2f}")
    
    return portfolio_values, returns, allocations, actions

def create_comparison_plots(agent_results, benchmark_results, test_data, save_path='../results/'):
    """
    Create comprehensive comparison plots
    
    Args:
        agent_results (tuple): Agent backtest results
        benchmark_results (tuple): Benchmark backtest results
        test_data (pd.DataFrame): Test data with dates
        save_path (str): Directory to save plots
    """
    
    agent_values, agent_returns, agent_allocations, agent_actions = agent_results
    benchmark_values, benchmark_returns, benchmark_allocations = benchmark_results
    
    # Create date index
    dates = test_data['Date'].iloc[:len(agent_values)]
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RL Trading Agent vs Buy-and-Hold Benchmark', fontsize=16, fontweight='bold')
    
    # 1. Portfolio Value Comparison
    axes[0, 0].plot(dates, agent_values, label='RL Agent', linewidth=2, color='blue')
    axes[0, 0].plot(dates, benchmark_values, label='Buy & Hold SPY', linewidth=2, color='red')
    axes[0, 0].set_title('Portfolio Value Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Cumulative Returns
    agent_cumret = np.cumprod(1 + np.array(agent_returns))
    benchmark_cumret = np.cumprod(1 + np.array(benchmark_returns))
    
    axes[0, 1].plot(dates[1:], agent_cumret, label='RL Agent', linewidth=2, color='blue')
    axes[0, 1].plot(dates[1:], benchmark_cumret, label='Buy & Hold SPY', linewidth=2, color='red')
    axes[0, 1].set_title('Cumulative Returns')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Cumulative Return')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Asset Allocation Over Time
    spy_allocations = [alloc[0] for alloc in agent_allocations]
    gld_allocations = [alloc[1] for alloc in agent_allocations]
    
    axes[1, 0].fill_between(dates, 0, spy_allocations, alpha=0.7, label='SPY Allocation', color='blue')
    axes[1, 0].fill_between(dates, spy_allocations, np.array(spy_allocations) + np.array(gld_allocations), 
                           alpha=0.7, label='GLD Allocation', color='gold')
    axes[1, 0].set_title('Asset Allocation Over Time')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Allocation %')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylim(0, 1)
    
    # 4. Rolling Sharpe Ratio Comparison
    window = 60  # 60-day rolling window
    agent_rolling_sharpe = []
    benchmark_rolling_sharpe = []
    
    for i in range(window, len(agent_returns)):
        agent_window = agent_returns[i-window:i]
        benchmark_window = benchmark_returns[i-window:i]
        
        agent_sharpe = np.mean(agent_window) / np.std(agent_window) * np.sqrt(252) if np.std(agent_window) > 0 else 0
        benchmark_sharpe = np.mean(benchmark_window) / np.std(benchmark_window) * np.sqrt(252) if np.std(benchmark_window) > 0 else 0
        
        agent_rolling_sharpe.append(agent_sharpe)
        benchmark_rolling_sharpe.append(benchmark_sharpe)
    
    rolling_dates = dates[window+1:len(agent_rolling_sharpe)+window+1]
    axes[1, 1].plot(rolling_dates, agent_rolling_sharpe, label='RL Agent', linewidth=2, color='blue')
    axes[1, 1].plot(rolling_dates, benchmark_rolling_sharpe, label='Buy & Hold SPY', linewidth=2, color='red')
    axes[1, 1].set_title(f'{window}-Day Rolling Sharpe Ratio')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Sharpe Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Performance comparison plot saved to {save_path}/performance_comparison.png")

def print_performance_summary(agent_results, benchmark_results):
    """
    Print detailed performance summary
    
    Args:
        agent_results (tuple): Agent backtest results
        benchmark_results (tuple): Benchmark backtest results
    """
    
    agent_values, agent_returns, _, _ = agent_results
    benchmark_values, benchmark_returns, _ = benchmark_results
    
    # Calculate metrics
    agent_analyzer = PortfolioAnalyzer(agent_returns)
    benchmark_analyzer = PortfolioAnalyzer(benchmark_returns)
    
    agent_metrics = agent_analyzer.calculate_metrics()
    benchmark_metrics = benchmark_analyzer.calculate_metrics()
    
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    print(f"{'Metric':<25} {'RL Agent':<15} {'Buy & Hold':<15} {'Difference':<15}")
    print("-"*70)
    
    for metric in agent_metrics.keys():
        if metric in benchmark_metrics:
            agent_val = agent_metrics[metric]
            benchmark_val = benchmark_metrics[metric]
            diff = agent_val - benchmark_val
            
            if 'Return' in metric or 'Ratio' in metric:
                print(f"{metric:<25} {agent_val:<15.2%} {benchmark_val:<15.2%} {diff:<15.2%}")
            elif 'Drawdown' in metric:
                print(f"{metric:<25} {agent_val:<15.2%} {benchmark_val:<15.2%} {diff:<15.2%}")
            elif 'Rate' in metric:
                print(f"{metric:<25} {agent_val:<15.2%} {benchmark_val:<15.2%} {diff:<15.2%}")
            else:
                print(f"{metric:<25} {agent_val:<15.2f} {benchmark_val:<15.2f} {diff:<15.2f}")
    
    # Final portfolio values
    print("-"*70)
    print(f"{'Final Portfolio Value':<25} ${agent_values[-1]:<14,.2f} ${benchmark_values[-1]:<14,.2f} ${agent_values[-1] - benchmark_values[-1]:<14,.2f}")
    
    # Performance vs objectives
    print("\n" + "="*60)
    print("PROJECT OBJECTIVES ANALYSIS")
    print("="*60)
    
    # Check 20% better performance
    performance_improvement = (agent_metrics['Annualized Return'] - benchmark_metrics['Annualized Return'])
    relative_improvement = performance_improvement / benchmark_metrics['Annualized Return'] if benchmark_metrics['Annualized Return'] != 0 else 0
    
    print(f"Annualized Return Improvement: {performance_improvement:.2%}")
    print(f"Relative Improvement: {relative_improvement:.2%}")
    print(f"Target: 20% better performance - {'✓ ACHIEVED' if relative_improvement >= 0.20 else '✗ NOT ACHIEVED'}")
    
    # Check 50% more final value
    value_improvement = (agent_values[-1] - benchmark_values[-1]) / benchmark_values[-1]
    print(f"Final Value Improvement: {value_improvement:.2%}")
    print(f"Target: 50% more final value - {'✓ ACHIEVED' if value_improvement >= 0.50 else '✗ NOT ACHIEVED'}")

def main():
    """Main backtesting function"""
    
    print("=== RL Trading Agent Backtesting ===")
    
    # Check if required files exist
    model_path = '../models/ppo_spy_gld.zip'
    test_data_path = '../data/test_data.csv'
    
    if not os.path.exists(model_path):
        print(f"Trained model not found: {model_path}")
        print("Please run train.py first to train the agent.")
        return False
    
    if not os.path.exists(test_data_path):
        print(f"Test data not found: {test_data_path}")
        print("Please run data_preprocessing.py first.")
        return False
    
    # Load test data
    test_data = pd.read_csv(test_data_path)
    test_data['Date'] = pd.to_datetime(test_data['Date'])
    
    print(f"Test period: {test_data['Date'].min()} to {test_data['Date'].max()}")
    print(f"Test data points: {len(test_data)}")
    
    try:
        # Run agent backtest
        print("\n1. Running RL Agent backtest...")
        agent_results = run_agent_backtest(model_path, test_data_path)
        
        # Run benchmark
        print("\n2. Running Buy-and-Hold benchmark...")
        benchmark_results = run_buy_and_hold_benchmark(test_data)
        
        # Create comparison plots
        print("\n3. Creating performance plots...")
        create_comparison_plots(agent_results, benchmark_results, test_data)
        
        # Print performance summary
        print("\n4. Analyzing results...")
        print_performance_summary(agent_results, benchmark_results)
        
        print("\n" + "="*60)
        print("BACKTESTING COMPLETED SUCCESSFULLY!")
        print("Check ../results/ for detailed plots and analysis.")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"Backtesting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
