import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.envs.registration import register

class TradingEnv(gym.Env):
    """
    A custom trading environment for Reinforcement Learning.
    Accepts a data path and handles data loading internally.
    """
    # Add metadata for rendering
    metadata = {'render_modes': ['human']}

    def __init__(self, data_path, initial_balance=100000, lookback_window=30, transaction_cost=0.001, render_mode=None):
        super(TradingEnv, self).__init__()

        # --- Load Data ---
        # The environment is now responsible for loading its own data
        self.df = pd.read_csv(data_path, index_col='Date', parse_dates=True).copy()
        
        # --- Environment Parameters ---
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.render_mode = render_mode # Handle render_mode argument

        # --- Spaces ---
        # Action space: [proportion_of_SPY]
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Observation space: window of OHLCV + features + current allocations
        # +2 for spy_allocation and gld_allocation
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback_window, len(self.df.columns) + 2),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.current_step = self.lookback_window
        
        # Initial portfolio allocation
        self.spy_allocation = 0.5
        self.gld_allocation = 0.5
        
        self.trades = []
        
        obs = self._next_observation()
        info = {}
        return obs, info

    def _next_observation(self):
        """Prepares the observation for the agent."""
        # Get the historical data window
        frame = self.df.iloc[self.current_step - self.lookback_window : self.current_step].values

        # Append current allocations
        allocations = np.array([self.spy_allocation, self.gld_allocation])
        allocations_frame = np.tile(allocations, (self.lookback_window, 1))
        
        obs = np.concatenate((frame, allocations_frame), axis=1)
        return obs.astype(np.float32)

    def step(self, action):
        target_spy_allocation = np.clip(action[0], 0, 1)

        prev_net_worth = self.net_worth
        
        current_prices = self.df.iloc[self.current_step]
        next_prices = self.df.iloc[self.current_step + 1]

        # Value of holdings before price change
        spy_value = self.spy_allocation * self.net_worth
        gld_value = self.gld_allocation * self.net_worth
        
        # Value after price change
        spy_value_next_day = spy_value * (next_prices['Close_SPY'] / current_prices['Close_SPY'])
        gld_value_next_day = gld_value * (next_prices['Close_GLD'] / current_prices['Close_GLD'])
        
        net_worth_before_rebalance = spy_value_next_day + gld_value_next_day

        current_spy_allocation = spy_value_next_day / net_worth_before_rebalance
        turnover = abs(target_spy_allocation - current_spy_allocation)
        
        transaction_costs = turnover * self.transaction_cost * net_worth_before_rebalance
        self.net_worth = net_worth_before_rebalance - transaction_costs
        self.balance = self.net_worth

        self.spy_allocation = target_spy_allocation
        self.gld_allocation = 1 - self.spy_allocation

        reward = np.log(self.net_worth / prev_net_worth) if prev_net_worth > 0 else 0

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        obs = self._next_observation()
        info = {
            'balance': self.balance,
            'net_worth': self.net_worth,
            'spy_allocation': self.spy_allocation,
            'gld_allocation': self.gld_allocation,
            'reward': reward,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}, Net Worth: {self.net_worth:,.2f}")

# --- Environment Registration ---
# This makes 'TradingEnv-v0' a callable environment ID
register(
    id='TradingEnv-v0',
    entry_point='environment:TradingEnv',
)