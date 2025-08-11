import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
import environment 

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
TENSORBOARD_LOG_DIR = os.path.join(RESULTS_DIR, 'tensorboard')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_data.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_data.csv')
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'ppo_trading_agent.zip')
VEC_NORMALIZE_STATS_PATH = os.path.join(MODELS_DIR, 'vecnormalize.pkl')

def main():
    """Main training pipeline for the RL trading agent."""
    print("=== RL Trading Agent Training Pipeline (Final Stable Run) ===")

    if not os.path.exists(TRAIN_DATA_PATH):
        print("Error: Training data not found. Please run data_preprocessing.py first.")
        return

    # --- Create Vectorized Environments using the registered ID ---
    train_env_kwargs = {'data_path': TRAIN_DATA_PATH}
    train_env = make_vec_env('TradingEnv-v0', n_envs=1, env_kwargs=train_env_kwargs) # CORRECT ID
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    eval_env_kwargs = {'data_path': TEST_DATA_PATH}
    eval_env = make_vec_env('TradingEnv-v0', n_envs=1, env_kwargs=eval_env_kwargs) # CORRECT ID
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # --- Callbacks ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODELS_DIR,
        log_path=RESULTS_DIR,
        eval_freq=10000,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
        verbose=1
    )

    # --- PPO Model Configuration ---
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device for training.")

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=42,
        device=device,
        tensorboard_log=TENSORBOARD_LOG_DIR
    )

    # --- Train ---
    total_timesteps = 500000
    print(f"Starting model training for {total_timesteps} timesteps...")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name="ppo_trading_agent_stable_final_1"
    )

    # --- Save Final Artifacts ---
    print("\nTraining complete. Saving final artifacts...")
    model.save(MODEL_SAVE_PATH)
    train_env.save(VEC_NORMALIZE_STATS_PATH)
    
    print(f"   - Model saved to: {MODEL_SAVE_PATH}")
    print(f"   - Normalization stats saved to: {VEC_NORMALIZE_STATS_PATH}")
    
    print("\n✅ Training pipeline finished successfully!")

if __name__ == "__main__":
    main()