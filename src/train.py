"""
Training Script for RL Trading Agent
Uses PPO algorithm from stable-baselines3 to train the agent
"""

import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
from environment import create_environment
import warnings
warnings.filterwarnings('ignore')

# Detect tensorboard availability
try:
    import tensorboard  # noqa: F401
    _TB_AVAILABLE = True
except Exception:
    _TB_AVAILABLE = False

def setup_training_environment(train_data_path, eval_data_path=None):
    """
    Set up training and evaluation environments
    
    Args:
        train_data_path (str): Path to training data
        eval_data_path (str): Path to evaluation data (optional)
    
    Returns:
        tuple: (train_env, eval_env)
    """
    print("Setting up training environment...")
    
    # Create vectorized training environment
    def make_train_env():
        env = create_environment(
            train_data_path,
            initial_balance=100000,
            lookback_window=30,
            transaction_cost=0.001,
            risk_free_rate=0.02,
            turnover_penalty=0.05,
        )
        return Monitor(env, filename='../results/training_log.csv', allow_early_resets=True)

    train_env = DummyVecEnv([make_train_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Create evaluation environment if eval data is provided
    eval_env = None
    if eval_data_path and os.path.exists(eval_data_path):
        print("Setting up evaluation environment...")
        def make_eval_env():
            env = create_environment(
                eval_data_path,
                initial_balance=100000,
                lookback_window=30,
                transaction_cost=0.001,
                risk_free_rate=0.02,
                turnover_penalty=0.05,
            )
            return Monitor(env, filename='../results/eval_log.csv', allow_early_resets=True)

        eval_env = DummyVecEnv([make_eval_env])
        # Create VecNormalize for eval, do not update stats; we will sync obs_rms from train
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    return train_env, eval_env

def create_ppo_model(env, learning_rate=1e-4, n_steps=4096, batch_size=256, 
                     n_epochs=10, gamma=0.99, verbose=1):
    """
    Create and configure PPO model
    
    Args:
        env: Training environment
        learning_rate (float): Learning rate for the optimizer
        n_steps (int): Number of steps to run for each environment per update
        batch_size (int): Minibatch size
        n_epochs (int): Number of epochs when optimizing the surrogate loss
        gamma (float): Discount factor
        verbose (int): Verbosity level
    
    Returns:
        PPO: Configured PPO model
    """
    print("Creating PPO model...")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=("../results/tensorboard/" if _TB_AVAILABLE else None),
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        ),
        verbose=verbose,
        seed=42
    )
    
    return model

def setup_callbacks(eval_env, eval_freq=5000, n_eval_episodes=10):
    """
    Set up training callbacks
    
    Args:
        eval_env: Evaluation environment
        eval_freq (int): Frequency of evaluation
        n_eval_episodes (int): Number of episodes for evaluation
    
    Returns:
        list: List of callbacks
    """
    callbacks = []

    if eval_env is not None:
        # Evaluation callback only (no early stopping)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='../models/',
            log_path='../results/',
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)

    return callbacks

def train_agent(total_timesteps=100000, save_path='../models/ppo_spy_gld'):
    """
    Main training function
    
    Args:
        total_timesteps (int): Total number of timesteps to train
        save_path (str): Path to save the trained model
    
    Returns:
        PPO: Trained model
    """
    
    # Create results directory
    os.makedirs('../results', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    
    # Set up environments
    train_env, eval_env = setup_training_environment(
        '../data/train_data.csv',
        '../data/test_data.csv'
    )
    # If using VecNormalize for eval, sync observation statistics
    if eval_env is not None and hasattr(train_env, 'obs_rms') and hasattr(eval_env, 'obs_rms'):
        eval_env.obs_rms = train_env.obs_rms
    
    # Create model
    model = create_ppo_model(train_env)
    
    # Set up callbacks
    callbacks = setup_callbacks(eval_env)
    
    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"Model will be saved to: {save_path}")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name="ppo_spy_gld_training"
    )
    
    # Save the final model
    model.save(save_path)
    # Save VecNormalize statistics for inference/backtest
    try:
        if hasattr(train_env, 'save'):
            train_env.save('../models/vecnormalize.pkl')
            print('Saved VecNormalize stats to ../models/vecnormalize.pkl')
    except Exception as e:
        print(f"Warning: could not save VecNormalize stats: {e}")
    print(f"Training completed! Model saved to {save_path}.zip")
    
    return model

def plot_training_progress():
    """Plot training progress from logs"""
    
    try:
        # Load training log
        if os.path.exists('../results/training_log.csv'):
            train_log = pd.read_csv('../results/training_log.csv')
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Training Progress', fontsize=16)
            
            # Episode rewards
            if 'r' in train_log.columns:
                axes[0, 0].plot(train_log['r'])
                axes[0, 0].set_title('Episode Rewards')
                axes[0, 0].set_xlabel('Episode')
                axes[0, 0].set_ylabel('Reward')
            
            # Episode lengths
            if 'l' in train_log.columns:
                axes[0, 1].plot(train_log['l'])
                axes[0, 1].set_title('Episode Lengths')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Steps')
            
            # Cumulative reward
            if 'r' in train_log.columns:
                axes[1, 0].plot(train_log['r'].cumsum())
                axes[1, 0].set_title('Cumulative Rewards')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Cumulative Reward')
            
            # Moving average reward
            if 'r' in train_log.columns:
                window = min(50, len(train_log))
                moving_avg = train_log['r'].rolling(window=window).mean()
                axes[1, 1].plot(moving_avg)
                axes[1, 1].set_title(f'Moving Average Reward (window={window})')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Average Reward')
            
            plt.tight_layout()
            plt.savefig('../results/training_progress.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("Training progress plot saved to ../results/training_progress.png")
        
    except Exception as e:
        print(f"Error plotting training progress: {e}")

def main():
    """Main function"""
    
    print("=== RL Trading Agent Training ===")
    
    # Check if processed data exists
    if not os.path.exists('../data/train_data.csv'):
        print("Training data not found. Please run data_preprocessing.py first.")
        return False
    
    try:
        # Train the agent
        model = train_agent(total_timesteps=500000)  # Increased for better performance
        
        # Plot training progress
        plot_training_progress()
        
        print("\nTraining completed successfully!")
        print("Next steps:")
        print("1. Run backtest.py to evaluate the trained agent")
        print("2. Check ../results/ for training logs and plots")
        print("3. The trained model is saved in ../models/")
        
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        return False

if __name__ == "__main__":
    main()
