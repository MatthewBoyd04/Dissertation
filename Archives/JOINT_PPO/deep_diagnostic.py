#!/usr/bin/env python3
"""
Deep diagnostic to understand why model.learn() isn't updating weights
"""
import sys
sys.path.append('..')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from JOINT_PPO.single_agent_wrapper import JointAgentWrapper
from JOINT_PPO.Environment import GridWorldEnvironment
from Maps import map_15x15
from LoggerConfig import log
import torch
import numpy as np

print("\n" + "="*70)
print("DEEP TRAINING DIAGNOSTIC")
print("="*70)

print("\n[1] Create fresh environment and model")
print("-" * 70)

def make_env():
    base_env = GridWorldEnvironment(
        mapPreset=map_15x15,
        maxCycles=256,
        visionRange=2,
        use_map_memory=True,
        num_drones=2
    )
    return JointAgentWrapper(base_env, use_reward_averaging=True)

env = DummyVecEnv([make_env for _ in range(2)])
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

print(f"Environment created")
print(f"  Observation space: {env.observation_space}")
print(f"  Action space: {env.action_space}")

model = PPO(
    "MlpPolicy",
    env,
    verbose=0,
    learning_rate=3e-4,
    n_steps=256,
    batch_size=64,
    device="cpu"
)

print(f"Model created")
print(f"  Initial num_timesteps: {model.num_timesteps}")

# Save initial weights
initial_weights = {name: param.data.clone() for name, param in model.policy.named_parameters()}
print(f"  Saved {len(initial_weights)} parameter tensors")

print(f"\n[2] Collect trajectories manually (no learn)")
print("-" * 70)

obs, _ = env.reset()
actions_collected = 0
rewards_collected = []

for step in range(256):
    action = env.action_space.sample()
    result = env.step(action)
    if len(result) == 5:
        obs, reward, done, truncated, info = result
    else:
        obs, reward, done, info = result  # DummyVecEnv returns 4 values
    rewards_collected.append(reward)
    actions_collected += 1

print(f"  Collected {actions_collected} steps")
print(f"  Reward stats: mean={np.mean(rewards_collected):.4f}, max={np.max(rewards_collected):.4f}")

print(f"\n[3] Check model gradient flow")
print("-" * 70)

# Reset environment
obs, _ = env.reset()

# Get one batch
actions = env.action_space.sample()
result = env.step(actions)
if len(result) == 5:
    obs, reward, done, truncated, info = result
else:
    obs, reward, done, info = result  # DummyVecEnv returns 4 values

# Check if we can compute gradients
print(f"  Testing forward pass...")
with torch.no_grad():
    policy_output = model.policy(torch.from_numpy(obs).float().to("cpu"))
    print(f"    Policy output type: {type(policy_output)}")
    print(f"    Policy output len: {len(policy_output) if hasattr(policy_output, '__len__') else 'N/A'}")

print(f"\nmodel.policy attributes:")
print(f"  model.policy.optimizer: {model.policy.optimizer}")
print(f"  model.learning_rate: {model.learning_rate}")

print(f"\n[4] Try ONE model.learn() call (64 timesteps)")
print("-" * 70)

print(f"  Before learn:")
print(f"    num_timesteps: {model.num_timesteps}")
print(f"    First param mean: {list(model.policy.parameters())[0].mean():.6f}")

try:
    print(f"  Calling model.learn(total_timesteps=64, log_interval=1)...")
    model.learn(total_timesteps=64, log_interval=1)
    print(f"  learn() completed")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

print(f"\n  After learn:")
print(f"    num_timesteps: {model.num_timesteps}")

# Check if weights changed
params_changed = False
for name, param in model.policy.named_parameters():
    if name in initial_weights:
        if not (initial_weights[name] == param.data).all():
            params_changed = True
            change_magnitude = (initial_weights[name] - param.data).abs().mean()
            print(f"    {name}: CHANGED (mean delta = {change_magnitude:.6f})")
            break

if not params_changed:
    print(f"    First param mean: {list(model.policy.parameters())[0].mean():.6f} (NO CHANGE!)")
    print(f"    **WARNING**: Parameters were not updated!")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70 + "\n")
