import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment import create_environment

MODELS_DIR = os.path.join('..', 'models')
RESULTS_DIR = os.path.join('..', 'results')
SAC_MODEL_PATH = os.path.join(MODELS_DIR, 'sac_spy_tlt')
VECNORM_PATH = os.path.join(MODELS_DIR, 'vecnormalize_sac.pkl')

os.makedirs(RESULTS_DIR, exist_ok=True)


def make_eval_env(data_path):
    def _thunk():
        env = create_environment(
            data_path,
            transaction_cost=0.0005,
            turnover_penalty=0.05,
            max_allocation_change=0.10,
            drawdown_penalty=0.10,
            risk_lambda=0.15,
            primary_symbol='SPY',
            secondary_symbol='TLT',
        )
        return env
    return _thunk


def run_backtest():
    print("=== RL Trading Agent Backtesting (SAC) ===")
    data_path = os.path.join('..', 'data', 'test_data.csv')

    # Create VecEnv + VecNormalize for evaluation
    eval_env = DummyVecEnv([make_eval_env(data_path)])
    if os.path.exists(VECNORM_PATH):
        print("Loading VecNormalize stats for evaluation...")
        eval_env = VecNormalize.load(VECNORM_PATH, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        print("VecNormalize stats not found, evaluating without normalization.")

    # Load SAC model
    model_zip = f"{SAC_MODEL_PATH}.zip"
    if not os.path.exists(model_zip):
        raise FileNotFoundError(f"Model not found at {model_zip}. Train with train_sac.py first.")

    print(f"Loading model from {model_zip}")
    model = SAC.load(model_zip, env=eval_env)

    # Prepare data for benchmark (SPY buy-and-hold)
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    spy_col = 'Close_SPY'
    if spy_col not in df.columns:
        raise ValueError('Close_SPY not found in test data')

    dates = df['Date'].iloc[-(len(df) - 1):].reset_index(drop=True)

    # Evaluation loop
    obs, info = eval_env.reset()
    done = False
    balances = []
    allocations = []

    step_count = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
        step_count += 1

        info0 = infos[0] if isinstance(infos, list) else infos
        bal = info0.get('balance', None)
        alloc_spy = info0.get('spy_allocation', None)
        if bal is not None:
            balances.append(bal)
        if alloc_spy is not None:
            allocations.append(alloc_spy)

        if dones[0]:
            break

        if step_count % 100 == 0 and bal is not None and alloc_spy is not None:
            print(f"Step {step_count}, Balance: ${bal:,.2f}, SPY: {alloc_spy*100:.2f}%")

    if len(balances) == 0:
        raise RuntimeError('No balance data collected during evaluation.')

    final_balance = balances[-1]
    print(f"Backtest completed. Final balance: ${final_balance:,.2f}")

    # Build series for metrics
    pv = np.array(balances, dtype=float)
    pv = np.concatenate(([pv[0]], pv)) if len(pv) < len(dates) else pv
    rl_returns = np.diff(pv) / pv[:-1]

    # Buy-and-Hold SPY
    spy_prices = df[spy_col].values
    spy_vals = spy_prices[1:] / spy_prices[0]
    buy_hold = spy_vals * pv[0]
    if len(buy_hold) > len(pv):
        buy_hold = buy_hold[:len(pv)]
    elif len(buy_hold) < len(pv):
        buy_hold = np.pad(buy_hold, (0, len(pv)-len(buy_hold)), 'edge')

    bh_returns = np.diff(buy_hold) / buy_hold[:-1]

    # Metrics
    def ann_ret(rets):
        if len(rets) == 0:
            return 0.0
        return (1 + np.mean(rets))**252 - 1

    def vol(rets):
        return np.std(rets) * np.sqrt(252)

    def sharpe(rets, rf=0.0):
        v = np.std(rets)
        if v == 0:
            return 0.0
        return (np.mean(rets) - rf/252) / v * np.sqrt(252)

    def max_dd(values):
        arr = np.array(values, dtype=float)
        run_max = np.maximum.accumulate(arr)
        dd = (arr - run_max) / run_max
        return dd.min()

    rl_ann = ann_ret(rl_returns)
    bh_ann = ann_ret(bh_returns)
    rl_sharpe = sharpe(rl_returns)
    bh_sharpe = sharpe(bh_returns)
    rl_mdd = max_dd(pv)
    bh_mdd = max_dd(buy_hold)

    print("\n============================================================")
    print("PERFORMANCE SUMMARY (SAC)")
    print("============================================================")
    print(f"Final Portfolio Value     ${final_balance:,.2f}")
    print(f"Annualized Return (RL)    {rl_ann*100:.2f}%")
    print(f"Annualized Return (B&H)   {bh_ann*100:.2f}%")
    print(f"Sharpe (RL)               {rl_sharpe:.2f}")
    print(f"Sharpe (B&H)              {bh_sharpe:.2f}")
    print(f"Max Drawdown (RL)         {rl_mdd*100:.2f}%")
    print(f"Max Drawdown (B&H)        {bh_mdd*100:.2f}%")
    print("============================================================\n")

    # Plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(pv))
    plt.plot(x, pv, label='RL (SAC)')
    plt.plot(x, buy_hold, label='Buy & Hold (SPY)')
    plt.title('Portfolio Value - RL (SAC) vs Buy & Hold')
    plt.xlabel('Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    out_path = os.path.join(RESULTS_DIR, 'performance_comparison_sac.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Performance comparison plot saved to {out_path}")


if __name__ == '__main__':
    run_backtest()
