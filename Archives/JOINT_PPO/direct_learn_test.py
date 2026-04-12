#!/usr/bin/env python3
"""Direct test of model.learn() to debug why weights aren't updating"""
import sys
sys.path.append('..')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from JOINT_PPO.train_JOINT_PPO import CurriculumWrapper
from JOINT_PPO.Environment import GridWorldEnvironment
import torch
import numpy as np

print("\n[DIRECT TEST] model.learn() weight tracking")
print("="*70)

# Exact same setup as trainJointAgent for 8 drones
def make_curriculum_env():
    return CurriculumWrapper(
        GridWorldEnvironment, 
        cumulativeTimestepsSoFar=0, 
        total_training_timesteps=10_000_000, 
        num_drones=8
    )

n_envs = 8  # THIS IS THE DIFFERENCE - using 8 like trainJointAgent
env = DummyVecEnv([make_curriculum_env for _ in range(n_envs)])
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

print(f"Environment created with n_envs={n_envs}")

model = PPO(
    "MlpPolicy",
    env,
    verbose=0,
    learning_rate=3e-4,
    n_steps=2048,  # FIXED value
    batch_size=256,  # FIXED value
    device="cpu"
)

print(f"Model created")
print(f"  Policy type: {type(model.policy)}")

# Get layer 0 weight before training
layer0_before = list(model.policy.parameters())[0].data.clone()
print(f"  Layer 0: mean={layer0_before.mean():.6f}, std={layer0_before.std():.6f}")

print(f"\nCalling model.learn(512)...")
model.learn(total_timesteps=512)

# Get layer 0 weight after training  
layer0_after = list(model.policy.parameters())[0].data
diff = (layer0_before - layer0_after).abs().mean()

print(f"\nAfter training:")
print(f"  Layer 0: mean={layer0_after.mean():.6f}, std={layer0_after.std():.6f}")
print(f"  Change: {diff:.6f}")

if diff > 0.0001:
    print(f"\n[SUCCESS] Weights were updated!")
else:
    print(f"\n[FAILURE] Weights NOT updated (change too small)")
    print(f"  This confirms gradients are not being applied...")

env.close()
