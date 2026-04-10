#!/usr/bin/env python3
"""Test if CurriculumWrapper is causing the issue"""
import sys
sys.path.append('..')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from JOINT_PPO.train_JOINT_PPO import CurriculumWrapper
from JOINT_PPO.Environment import GridWorldEnvironment
from Maps import map_15x15
import torch

print("\nTesting CurriculumWrapper training...")
print("="*70)

def make_curriculum_env():
    return CurriculumWrapper(
        GridWorldEnvironment, 
        cumulativeTimestepsSoFar=0, 
        total_training_timesteps=10_000_000, 
        num_drones=8
    )

# Create env with CurriculumWrapper
n_envs = 2
env = DummyVecEnv([make_curriculum_env for _ in range(n_envs)])
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

print(f"Environment created (n_envs={n_envs})")
print(f"  Observation space: {env.observation_space}")
print(f"  Action space: {env.action_space}")

# Create model
model = PPO(
    "MlpPolicy",
    env,
    verbose=0,
    learning_rate=3e-4,
    n_steps=256,
    batch_size=64,
    device="cpu"
)

print(f"\nModel created")
print(f"  Initial num_timesteps: {model.num_timesteps}")

# Get initial weights
initial_params = {name: param.data.clone() for name, param in model.policy.named_parameters()}

# Try training
print(f"\nTraining for 256 timesteps...")
try:
    model.learn(total_timesteps=256, log_interval=1)
    print(f"  learn() completed")
    print(f"  Final num_timesteps: {model.num_timesteps}")
    
    # Check weights
    params_changed = False
    for name, param in model.policy.named_parameters():
        if name in initial_params:
            if not (initial_params[name] == param.data).all():
                params_changed = True
                print(f"  WEIGHTS CHANGED: {name}")
                break
    
    if params_changed:
        print(f"\nSUCCESS: CurriculumWrapper training works!")
    else:
        print(f"\nFAILURE: Weights still random after training")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

env.close()
