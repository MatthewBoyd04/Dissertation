#!/usr/bin/env python3
"""
Comprehensive training diagnostic script
Tests each component of the training pipeline independently
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from JOINT_PPO.single_agent_wrapper import JointAgentWrapper
from JOINT_PPO.Environment import GridWorldEnvironment
from Maps import map_15x15
from LoggerConfig import log
import torch

print("\n" + "="*70)
print("JOINT PPO TRAINING DIAGNOSTIC")
print("="*70)

# Test 1: Curriculum wrapper with closure
print("\n[TEST 1] Curriculum Wrapper Closure Test")
print("-" * 70)

# Test 1: Curriculum wrapper with closure
print("\n[TEST 1] Curriculum Wrapper Map Selection")
print("-" * 70)

timesteps_values = [100, 50000, 1000001, 3000001, 6000001]

from JOINT_PPO.train_JOINT_PPO import CurriculumWrapper

for ts in timesteps_values:
    curriculum = CurriculumWrapper(GridWorldEnvironment, ts, 10_000_000, 8)
    # CurriculumWrapper.env is a JointAgentWrapper which wraps the GridWorldEnvironment
    base_env = curriculum.env.env
    print(f"  At {ts:>7,} timesteps: maxCycles = {base_env.maxCycles:4d} (map={len(base_env.grid)}x{len(base_env.grid)})")
    curriculum.close()

# Test 2: Model path consistency
print("\n[TEST 2] Model Save/Load Path Consistency")
print("-" * 70)

model_path = "joint_ppo_model_2drones.zip"
save_model_path = "joint_ppo_model_2drones"  # Without .zip

print(f"  Loading from: {model_path}")
print(f"  Saving to:    {save_model_path}")
print(f"  After save(), file will be: {save_model_path}.zip")
print(f"  → Next iteration loads from: {model_path} ✓")

# Test 3: Check existing model
print("\n[TEST 3] Existing Model State Check")
print("-" * 70)

model_8d = "joint_ppo_model_8drones.zip"
if os.path.exists(model_8d):
    try:
        model = PPO.load(model_8d, device="cpu")
        print(f"  ✓ Model loaded: {model_8d}")
        print(f"    - num_timesteps: {model.num_timesteps}")
        print(f"    - Policy weights stats:")
        
        for i, param in enumerate(list(model.policy.parameters())[:2]):
            print(f"      Layer {i}: min={param.min():.4f}, max={param.max():.4f}, mean={param.mean():.6f}")
        
        # Check if weights look trained
        first_param = list(model.policy.parameters())[0]
        if abs(first_param.mean()) < 0.1:
            print(f"    → Weights appear UNTRAINED (near random init)")
        else:
            print(f"    → Weights appear TRAINED (diverged from init)")
            
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
else:
    print(f"  ✗ Model not found: {model_8d}")

# Test 4: Full training pipeline simulation
print("\n[TEST 4] Mini Training Run (256 timesteps)")
print("-" * 70)

try:
    def make_test_env():
        base_env = GridWorldEnvironment(
            mapPreset=map_15x15,
            maxCycles=256,
            visionRange=2,
            use_map_memory=True,
            num_drones=2
        )
        return JointAgentWrapper(base_env, use_reward_averaging=True)
    
    env = DummyVecEnv([make_test_env for _ in range(2)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    print(f"  ✓ Vectorized environment created")
    print(f"    Observation space: {env.single_observation_space}")
    print(f"    Action space: {env.single_action_space}")
    
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
    print(f"  ✓ Model created")
    
    # Get initial params
    initial_params = {}
    for name, param in model.policy.named_parameters():
        initial_params[name] = param.data.clone()
    
    print(f"  ⊳ Training for 256 timesteps...")
    model.learn(total_timesteps=256)
    print(f"  ✓ model.learn() completed")
    print(f"    - Model num_timesteps after learn: {model.num_timesteps}")
    
    # Check if params changed
    params_changed = False
    for name, param in model.policy.named_parameters():
        if name in initial_params:
            if not (initial_params[name] == param.data).all():
                params_changed = True
                print(f"    ✓ Parameter '{name[:30]}...' was updated")
                break
    
    if not params_changed:
        print(f"    ✗ NO PARAMETERS WERE UPDATED!")
    else:
        print(f"    ✓ Training worked - weights were updated")
    
    env.close()
    
except Exception as e:
    print(f"  ✗ Error during training test: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Main configuration mismatch
print("\n[TEST 5] Configuration Consistency Check")
print("-" * 70)

main_num_drones = 2  # From Main_JOINT_PPO.py
existing_model_drones = 8  # Model file has 8

print(f"  Main_JOINT_PPO.py: num_drones = {main_num_drones}")
print(f"  Existing model:    num_drones = {existing_model_drones}")

if main_num_drones != existing_model_drones:
    print(f"  ✗ MISMATCH! Will create NEW model on next run")
    print(f"    (observation space changes with agent count)")
else:
    print(f"  ✓ Match")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70 + "\n")
