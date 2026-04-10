from stable_baselines3 import PPO
import os
import numpy as np

model = PPO.load("joint_ppo_model_8drones.zip", device="cpu")
print(f"Model timesteps: {model.num_timesteps}")
print(f"Model type: {type(model)}")

# Check the environment the model was trained with
print(f"\nModel env type: {type(model.env)}")
if hasattr(model.env, 'observation_space'):
    print(f"Env observation space: {model.env.observation_space}")
if hasattr(model.env, 'action_space'):
    print(f"Env action space: {model.env.action_space}")

# Check model attributes
print(f"\nModel attributes:")
print(f"  num_timesteps: {model.num_timesteps}")
print(f"  _last_obs: {type(model._last_obs) if model._last_obs is not None else 'None'}")

# Sample first layer weights
first_layer = list(model.policy.parameters())[0]
print(f"\nFirst layer stats:")
print(f"  Shape: {first_layer.shape}")
print(f"  Min: {first_layer.min():.6f}")
print(f"  Max: {first_layer.max():.6f}")
print(f"  Mean: {first_layer.mean():.6f}")
print(f"  Std: {first_layer.std():.6f}")

# Test prediction
try:
    dummy_obs = np.zeros((1, 1200), dtype=np.float32)
    action, _ = model.predict(dummy_obs, deterministic=False)
    print(f"\nPrediction works: action shape = {action.shape}, actions = {action}")
except Exception as e:
    print(f"\nPrediction failed: {e}")
