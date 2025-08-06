"""
Custom RL Environment for SPY-GLD Trading Agent
Based on FinRL framework with continuous action space
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env
import warnings
warnings.filterwarnings('ignore')

class TradingEnvironment(gym.Env):
    """
    Custom Trading Environment for SPY-GLD allocation
    
    State Space: Current allocation + 30-day window of market features
    Action Space: Continuous [0,1] representing SPY allocation percentage
    Reward: Sharpe ratio-based reward for risk-adjusted returns
    """
    
    def __init__(self, data, initial_balance=100000, lookback_window=30, 
                 transaction_cost=0.001, risk_free_rate=0.02):
        """
        Initialize the trading environment
        
        Args:
            data (pd.DataFrame): Processed market data with features
            initial_balance (float): Starting portfolio value
            lookback_window (int): Number of days to include in state
            transaction_cost (float): Transaction cost as percentage
            risk_free_rate (float): Risk-free rate for Sharpe ratio calculation
        """
        super(TradingEnvironment, self).__init__()
        
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        
        # Ensure data is sorted by date
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        
        # Extract price columns for SPY and GLD
        self.spy_prices = self.data['Close_SPY'].values
        self.gld_prices = self.data['Close_GLD'].values
        
        # Feature columns (excluding Date and Close prices)
        self.feature_columns = [col for col in self.data.columns 
                               if col not in ['Date', 'Close_SPY', 'Close_GLD']]
        self.features = self.data[self.feature_columns].values
        
        # Normalize features
        self.features = self._normalize_features(self.features)
        
        # Environment parameters
        self.max_steps = len(self.data) - self.lookback_window - 1
        
        # Action space: continuous [0, 1] for SPY allocation
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # State space: current allocation (2) + lookback window features
        state_dim = 2 + (self.lookback_window * len(self.feature_columns))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # Initialize environment state
        self.reset()
    
    def _normalize_features(self, features):
        """Normalize features to have zero mean and unit variance"""
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0)
        
        # Normalize each feature column
        normalized_features = np.zeros_like(features)
        for i in range(features.shape[1]):
            col_data = features[:, i]
            if np.std(col_data) > 0:
                normalized_features[:, i] = (col_data - np.mean(col_data)) / np.std(col_data)
            else:
                normalized_features[:, i] = col_data
        
        return normalized_features
    
    def reset(self):
        """Reset the environment to initial state"""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.spy_allocation = 0.5  # Start with 50-50 allocation
        self.gld_allocation = 0.5
        
        # Portfolio tracking
        self.portfolio_values = [self.initial_balance]
        self.returns = []
        self.allocations = [(self.spy_allocation, self.gld_allocation)]
        
        # Get initial state
        state = self._get_state()
        return state.astype(np.float32)
    
    def _get_state(self):
        """
        Construct the current state vector
        
        Returns:
            np.array: State vector containing current allocation and feature window
        """
        # Current allocation
        allocation_state = np.array([self.spy_allocation, self.gld_allocation])
        
        # Feature window (last lookback_window days)
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        feature_window = self.features[start_idx:end_idx].flatten()
        
        # If we don't have enough history, pad with zeros
        if len(feature_window) < self.lookback_window * len(self.feature_columns):
            padding_size = (self.lookback_window * len(self.feature_columns)) - len(feature_window)
            feature_window = np.concatenate([np.zeros(padding_size), feature_window])
        
        # Combine allocation and features
        state = np.concatenate([allocation_state, feature_window])
        
        return state
    
    def step(self, action):
        """
        Execute one step in the environment
        
        Args:
            action (np.array): Action to take [SPY allocation percentage]
        
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Extract action
        new_spy_allocation = np.clip(action[0], 0.0, 1.0)
        new_gld_allocation = 1.0 - new_spy_allocation
        
        # Calculate transaction costs
        allocation_change = abs(new_spy_allocation - self.spy_allocation)
        transaction_cost = allocation_change * self.transaction_cost * self.balance
        
        # Update allocations
        old_spy_allocation = self.spy_allocation
        old_gld_allocation = self.gld_allocation
        
        self.spy_allocation = new_spy_allocation
        self.gld_allocation = new_gld_allocation
        
        # Move to next time step
        self.current_step += 1
        
        # Calculate returns
        if self.current_step < len(self.spy_prices):
            spy_return = (self.spy_prices[self.current_step] - self.spy_prices[self.current_step - 1]) / self.spy_prices[self.current_step - 1]
            gld_return = (self.gld_prices[self.current_step] - self.gld_prices[self.current_step - 1]) / self.gld_prices[self.current_step - 1]
            
            # Portfolio return
            portfolio_return = (old_spy_allocation * spy_return + 
                              old_gld_allocation * gld_return)
            
            # Update balance
            self.balance = self.balance * (1 + portfolio_return) - transaction_cost
            
            # Track portfolio
            self.portfolio_values.append(self.balance)
            self.returns.append(portfolio_return)
            self.allocations.append((self.spy_allocation, self.gld_allocation))
        
        # Calculate reward (Sharpe ratio-based)
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self.current_step >= self.max_steps or self.balance <= 0
        
        # Get next state
        next_state = self._get_state()
        
        # Info dictionary
        info = {
            'balance': self.balance,
            'spy_allocation': self.spy_allocation,
            'gld_allocation': self.gld_allocation,
            'portfolio_return': portfolio_return if self.current_step < len(self.spy_prices) else 0,
            'transaction_cost': transaction_cost
        }
        
        return next_state.astype(np.float32), reward, done, info
    
    def _calculate_reward(self):
        """
        Calculate reward based on Sharpe ratio
        
        Returns:
            float: Reward value
        """
        if len(self.returns) < 2:
            return 0.0
        
        # Calculate recent returns (last 30 days or available)
        recent_returns = self.returns[-min(30, len(self.returns)):]
        
        if len(recent_returns) == 0:
            return 0.0
        
        # Calculate Sharpe ratio
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)
        
        if std_return == 0:
            return mean_return * 100  # If no volatility, reward is just the return
        
        # Annualized Sharpe ratio (assuming daily returns)
        daily_risk_free = self.risk_free_rate / 252
        sharpe_ratio = (mean_return - daily_risk_free) / std_return * np.sqrt(252)
        
        # Scale reward to be more meaningful for RL
        reward = sharpe_ratio / 10.0
        
        # Add penalty for extreme allocations to encourage diversification
        allocation_penalty = 0
        if self.spy_allocation < 0.1 or self.spy_allocation > 0.9:
            allocation_penalty = -0.1
        
        return reward + allocation_penalty
    
    def render(self, mode='human'):
        """Render the environment (optional)"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"SPY Allocation: {self.spy_allocation:.2%}")
            print(f"GLD Allocation: {self.gld_allocation:.2%}")
            if len(self.returns) > 0:
                print(f"Last Return: {self.returns[-1]:.4f}")
            print("-" * 40)

def create_environment(data_path, **kwargs):
    """
    Factory function to create trading environment
    
    Args:
        data_path (str): Path to processed data CSV
        **kwargs: Additional environment parameters
    
    Returns:
        TradingEnvironment: Configured trading environment
    """
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    
    env = TradingEnvironment(data, **kwargs)
    
    # Validate environment
    try:
        check_env(env)
        print("Environment validation passed!")
    except Exception as e:
        print(f"Environment validation warning: {e}")
    
    return env

if __name__ == "__main__":
    # Test the environment
    print("Testing Trading Environment...")
    
    # Create dummy data for testing
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    dummy_data = pd.DataFrame({
        'Date': dates,
        'Close_SPY': 300 + np.cumsum(np.random.randn(100) * 0.01),
        'Close_GLD': 150 + np.cumsum(np.random.randn(100) * 0.005),
        'Returns_SPY': np.random.randn(100) * 0.02,
        'Returns_GLD': np.random.randn(100) * 0.015,
        'RSI_SPY': 50 + np.random.randn(100) * 10,
        'RSI_GLD': 50 + np.random.randn(100) * 10,
    })
    
    env = TradingEnvironment(dummy_data)
    
    # Test environment
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    for i in range(5):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print(f"Step {i+1}: Action={action[0]:.3f}, Reward={reward:.4f}, Balance=${info['balance']:.2f}")
        
        if done:
            break
    
    print("Environment test completed!")
