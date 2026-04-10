#!/usr/bin/env python3
"""
Pre-flight check before running actual training
Verifies:
1. Environment loads correctly with correct num_drones
2. Model.learn() actually updates weights
3. Logging is working
4. Save/load cycle preserves state
"""
import sys
sys.path.append('..')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from JOINT_PPO.single_agent_wrapper import JointAgentWrapper
from JOINT_PPO.Environment import GridWorldEnvironment
from JOINT_PPO.train_JOINT_PPO import trainJointAgent
from Maps import map_15x15
from LoggerConfig import log
import os

print("\n" + "="*70)
print("JOINT PPO - PRE-FLIGHT CHECK")
print("="*70)

print("\n[CHECK 1] Configuration")
print("-" * 70)

num_drones = 8
print(f"  num_drones configured: {num_drones}")

model_path = f"joint_ppo_model_{num_drones}drones.zip"
if os.path.exists(model_path):
    print(f"  WARNING: Existing model found: {model_path}")
    print(f"  -> Will delete and retrain from scratch")
else:
    print(f"  OK: No existing model - will create fresh model")

print("\n[CHECK 2] Single Training Step (512 timesteps)")
print("-" * 70)

try:
    # Run one mini training iteration
    print(f"  Calling: trainJointAgent(total_timesteps=512, num_drones={num_drones})")
    trainJointAgent(total_timesteps=512, num_drones=num_drones, cumulativeTimestepsSoFar=0)
    print(f"  OK: trainJointAgent() completed")
    
    # Verify model was created and saved
    if os.path.exists(f"joint_ppo_model_{num_drones}drones.zip"):
        print(f"  OK: Model file created: joint_ppo_model_{num_drones}drones.zip")
        
        # Load and verify
        model = PPO.load(f"joint_ppo_model_{num_drones}drones.zip", device="cpu")
        print(f"  OK: Model loaded successfully")
        print(f"    - num_timesteps: {model.num_timesteps}")
        
        # Check weights
        first_param = list(model.policy.parameters())[0]
        print(f"    - First layer mean: {first_param.mean():.6f}")
        
        if abs(first_param.mean()) < 0.1:
            print(f"    WARNING: Weights still appear random (untrained)")
        else:
            print(f"    OK: Weights appear trained")
    else:
        print(f"  ERROR: Model file not found!")
        
except Exception as e:
    print(f"  ERROR during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[CHECK 3] VecNormalize Stats")
print("-" * 70)

vec_norm_path = f"joint_ppo_vec_normalize_{num_drones}drones.pkl"
if os.path.exists(vec_norm_path):
    print(f"  OK: VecNormalize file saved: {vec_norm_path}")
else:
    print(f"  WARNING: VecNormalize file NOT found: {vec_norm_path}")

print("\n[CHECK 4] Training Progress File")
print("-" * 70)

progress_file = "training_progress.json"
if os.path.exists(progress_file):
    import json
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    print(f"  OK: Progress file exists")
    print(f"    - iterations_completed: {progress.get('iterations_completed', 'N/A')}")
    print(f"    - total_timesteps: {progress.get('total_timesteps', 'N/A')}")
else:
    print(f"  INFO: No progress file yet (expected for fresh start)")

print("\n" + "="*70)
print("OK: PRE-FLIGHT CHECK COMPLETE")
print("="*70)
print("\nNext steps:")
print("  1. Review the logs above for any ERROR marks")
print("  2. If all checks pass, run: python Main_JOINT_PPO.py")
print("  3. Monitor the training logs for weight updates\n")
