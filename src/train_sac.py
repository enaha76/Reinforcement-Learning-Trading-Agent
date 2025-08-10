import os
import numpy as np
import pandas as pd
import gymnasium as gym
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from environment import create_environment

MODELS_DIR = os.path.join('..', 'models')
RESULTS_DIR = os.path.join('..', 'results')
LOG_DIR = None  # Disable TensorBoard unless installed
SAC_MODEL_PATH = os.path.join(MODELS_DIR, 'sac_spy_tlt')
VECNORM_PATH = os.path.join(MODELS_DIR, 'vecnormalize_sac.pkl')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
if LOG_DIR is not None:
    os.makedirs(LOG_DIR, exist_ok=True)


def make_env(data_path, primary_symbol='SPY', secondary_symbol='TLT'):
    def _thunk():
        env = create_environment(
            data_path,
            transaction_cost=0.0005,
            turnover_penalty=0.05,
            max_allocation_change=0.10,
            drawdown_penalty=0.10,
            risk_lambda=0.15,
            primary_symbol=primary_symbol,
            secondary_symbol=secondary_symbol,
        )
        return Monitor(env)
    return _thunk


def setup_envs(train_data_path, eval_data_path, n_envs: int = 4):
    # Training: parallel envs for faster step collection
    if n_envs and n_envs > 1:
        train_env = SubprocVecEnv([make_env(train_data_path) for _ in range(n_envs)])
    else:
        train_env = DummyVecEnv([make_env(train_data_path)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Evaluation: keep single env, no reward norm, not training
    eval_env = DummyVecEnv([make_env(eval_data_path)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # Share obs stats
    eval_env.obs_rms = train_env.obs_rms
    return train_env, eval_env


def train_sac(total_timesteps=2_000_000,
              train_data_path='../data/train_data.csv',
              eval_data_path='../data/test_data.csv',
              n_envs: int = 4,
              device: str | None = None):
    # Select device
    if device is None or device == 'auto':
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        DEVICE = device
    print(f"Using {DEVICE} device")

    train_env, eval_env = setup_envs(train_data_path, eval_data_path, n_envs=n_envs)

    policy_kwargs = dict(net_arch=[512, 512, 256])

    model = SAC(
        'MlpPolicy',
        train_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        batch_size=256,
        gamma=0.995,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        policy_kwargs=policy_kwargs,
        target_update_interval=1,
        device=DEVICE,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODELS_DIR,
        log_path=RESULTS_DIR,
        eval_freq=20000,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=MODELS_DIR, name_prefix='sac_checkpoint')

    callbacks = CallbackList([eval_callback, checkpoint_callback])
    # Do not pass tb_log_name when tensorboard is disabled
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    model.save(SAC_MODEL_PATH)
    # Save VecNormalize stats
    train_env.save(VECNORM_PATH)
    print(f"Saved SAC model to {SAC_MODEL_PATH}.zip and VecNormalize to {VECNORM_PATH}")

    return model


def main():
    print('=== Training SAC for Portfolio Allocation (SPY + TLT fallback to GLD) ===')
    model = train_sac()
    print('Training completed! Next: run backtest_sac.py')


if __name__ == '__main__':
    main()
