import os
import torch as th
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from envs.bidding import BiddingEnv  # Assuming this is your custom environment

# Create the environment
env = BiddingEnv()

# Monitor for logging rewards
log_dir = "./runs/baselines"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

# Wrap with VecNormalize for observation and reward normalization
vec_env = DummyVecEnv([lambda: env])  # Single process
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

# Choose an algorithm
algo = "TD3"  # Choose from 'SAC', 'TD3', or 'PPO'
if algo == "SAC":
    model = SAC(
        "MultiInputPolicy",  # Use MultiInputPolicy for Dict observation spaces
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        device="auto",
    )
elif algo == "TD3":
    model = TD3(
        "MultiInputPolicy",  # Use MultiInputPolicy for Dict observation spaces
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        device="auto",
    )
elif algo == "PPO":
    model = PPO(
        "MultiInputPolicy",  # Use MultiInputPolicy for Dict observation spaces
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        device="auto",
    )
else:
    raise ValueError(f"Unsupported algorithm: {algo}")

# Evaluation Callback
eval_env = DummyVecEnv([lambda: BiddingEnv()])  # Separate evaluation environment
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(log_dir, algo),
    log_path=log_dir,
    eval_freq=5000,  # Evaluate every 5000 steps
    deterministic=True,
)

# Checkpoint Callback
checkpoint_callback = CheckpointCallback(
    save_freq=10000,  # Save model every 10000 steps
    save_path=os.path.join(log_dir, "checkpoints"),
    name_prefix=f"{algo}_model"
)

# Train the model
model.learn(
    total_timesteps=int(1e6),  # 1M steps
    callback=[eval_callback, checkpoint_callback]
)

# Save the final model
model.save(os.path.join(log_dir, f"{algo}_final_model"))

# Evaluate the model
mean_reward, std_reward = eval_env.get_attr("env")[0].evaluate_policy(model, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Close environments
vec_env.close()
eval_env.close()