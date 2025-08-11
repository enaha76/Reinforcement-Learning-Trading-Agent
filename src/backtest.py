import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import environment

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_data.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.zip')
VEC_NORMALIZE_STATS_PATH = os.path.join(MODELS_DIR, 'vecnormalize.pkl')
PLOT_SAVE_PATH = os.path.join(RESULTS_DIR, 'performance_comparison.png')

def run_agent_backtest(model, test_env):
    """Runs the backtest for the trained agent."""
    obs = test_env.reset()
    done = False
    
    initial_balance = test_env.get_attr('initial_balance')[0]
    net_worths = [initial_balance]
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = test_env.step(action)
        done = dones[0]
        net_worths.append(infos[0]['net_worth'])

    return net_worths

def run_benchmark(test_df, initial_balance=100000):
    """Runs a simple 50/50 buy-and-hold benchmark strategy."""
    spy_prices = test_df['Close_SPY']
    gld_prices = test_df['Close_GLD']

    spy_shares = (initial_balance * 0.5) / spy_prices.iloc[0]
    gld_shares = (initial_balance * 0.5) / gld_prices.iloc[0]

    benchmark_portfolio = (spy_shares * spy_prices) + (gld_shares * gld_prices)
    return benchmark_portfolio.tolist()

def calculate_metrics(portfolio_values):
    """Calculates key performance metrics."""
    returns = pd.Series(portfolio_values).pct_change().dropna()
    
    if len(returns) < 2:
        return {
            "Final Portfolio Value": portfolio_values[-1], "Total Return": 0, 
            "Annualized Return": 0, "Annualized Volatility": 0,
            "Sharpe Ratio": 0, "Max Drawdown": 0
        }

    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    cumulative = pd.Series(portfolio_values)
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        "Final Portfolio Value": portfolio_values[-1], "Total Return": total_return,
        "Annualized Return": annualized_return, "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio, "Max Drawdown": max_drawdown
    }

def main():
    """Main backtesting pipeline."""
    print("=== RL Trading Agent Backtesting ===")

    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Test data not found at {TEST_DATA_PATH}")
        return
        
    test_df = pd.read_csv(TEST_DATA_PATH, index_col='Date', parse_dates=True)
    print(f"Test data loaded. Shape: {test_df.shape}")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_NORMALIZE_STATS_PATH):
        print("Error: Model or normalization stats not found. Please run train.py first.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)
    
    env_kwargs = {'data_path': TEST_DATA_PATH}
    test_env = make_vec_env('TradingEnv-v0', n_envs=1, env_kwargs=env_kwargs)
    test_env = VecNormalize.load(VEC_NORMALIZE_STATS_PATH, test_env)
    test_env.training = False
    test_env.norm_reward = False

    print("\nRunning RL Agent backtest...")
    agent_values = run_agent_backtest(model, test_env)
    print(f"Backtest completed. Final RL Agent balance: ${agent_values[-1]:,.2f}")

    print("Running Buy-and-Hold benchmark...")
    benchmark_values = run_benchmark(test_df)
    print(f"Backtest completed. Final Buy-and-Hold balance: ${benchmark_values[-1]:,.2f}")

    print("\nCreating performance plots...")
    min_len = min(len(agent_values), len(benchmark_values))
    results_df = pd.DataFrame({
        'RL Agent': agent_values[:min_len],
        'Benchmark (50/50)': benchmark_values[:min_len]
    }, index=test_df.index[:min_len])

    plt.style.use('seaborn-v0_8-darkgrid')
    # --- CORRECTED PLOTTING ---
    # Combine style and color into one argument to prevent the error
    results_df.plot(
        figsize=(14, 7), 
        title='RL Agent vs. Buy-and-Hold Benchmark', 
        ylabel='Portfolio Value ($)',
        style=['-', '--'],  # Agent is solid, Benchmark is dashed
        color=['royalblue', 'grey']
    )
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=300)
    print(f"Performance plot saved to {PLOT_SAVE_PATH}")
    plt.show()

    agent_metrics = calculate_metrics(agent_values)
    benchmark_metrics = calculate_metrics(benchmark_values)

    summary_data = []
    metric_names = [
        "Final Portfolio Value", "Total Return", "Annualized Return", 
        "Annualized Volatility", "Sharpe Ratio", "Max Drawdown"
    ]
    for metric in metric_names:
        row = {"Metric": metric, "RL Agent": agent_metrics.get(metric), "Benchmark (50/50)": benchmark_metrics.get(metric)}
        summary_data.append(row)
    
    summary = pd.DataFrame(summary_data).set_index('Metric')

    # Formatting
    summary['RL Agent'] = summary.apply(lambda row: f"${row['RL Agent']:,.2f}" if row.name == 'Final Portfolio Value' else f"{row['RL Agent']:.2%}" if isinstance(row['RL Agent'], (int, float)) and row.name in ["Total Return", "Annualized Return", "Annualized Volatility", "Max Drawdown"] else f"{row['RL Agent']:.2f}", axis=1)
    summary['Benchmark (50/50)'] = summary.apply(lambda row: f"${row['Benchmark (50/50)']:,.2f}" if row.name == 'Final Portfolio Value' else f"{row['Benchmark (50/50)']:.2%}" if isinstance(row['Benchmark (50/50)'], (int, float)) and row.name in ["Total Return", "Annualized Return", "Annualized Volatility", "Max Drawdown"] else f"{row['Benchmark (50/50)']:.2f}", axis=1)

    print("\n--- Performance Summary ---")
    print(summary)
    print("\n✅ Backtesting completed successfully!")

if __name__ == "__main__":
    main()