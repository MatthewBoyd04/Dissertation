#!/usr/bin/env python3
"""Debug script to test training step by step"""
import sys
sys.path.append('..')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from JOINT_PPO.single_agent_wrapper import JointAgentWrapper
from JOINT_PPO.Environment import GridWorldEnvironment
from Maps import map_15x15
from LoggerConfig import log
import os

print("=" * 60)
print("JOINT PPO TRAINING DEBUG")
print("=" * 60)

# Create environment
print("\n1. Creating environment...")
try:
    base_env = GridWorldEnvironment(
        mapPreset=map_15x15,
        maxCycles=256,
        visionRange=2,
        render_every=None,
        use_map_memory=True,
        num_drones=8
    )
    print("   ✓ Base environment created")
    
    wrapper_env = JointAgentWrapper(base_env, use_reward_averaging=True)
    print("   ✓ Wrapper applied")
    print(f"   Observation space: {wrapper_env.observation_space}")
    print(f"   Action space: {wrapper_env.action_space}")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create vectorized environment
print("\n2. Creating vectorized environment...")
try:
    def make_env():
        base_env = GridWorldEnvironment(
            mapPreset=map_15x15,
            maxCycles=256,
            visionRange=2,
            use_map_memory=True,
            num_drones=8
        )
        return JointAgentWrapper(base_env, use_reward_averaging=True)
    
    env = DummyVecEnv([make_env for _ in range(2)])  # Just 2 envs for testing
    print(f"   ✓ DummyVecEnv created")
    
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    print(f"   ✓ VecNormalize applied")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create model
print("\n3. Creating PPO model...")
try:
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=256,  # Reduced for testing
        batch_size=64,  # Reduced for testing
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        gae_lambda=0.95,
        device="cpu",
        tensorboard_log="./test_tensorboard"
    )
    print("   ✓ PPO model created")
    print(f"   Policy: {type(model.policy)}")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test a single step
print("\n4. Testing single step...")
try:
    obs, _ = env.reset()
    print(f"   Initial obs shape: {obs.shape}")
    
    action = env.action_space.sample()
    print(f"   Sample action: {action}")
    print(f"   Action type: {type(action)}")
    
    obs, reward, done, info = env.step(action)
    print(f"   ✓ Single step worked")
    print(f"   Obs shape: {obs.shape}, Reward: {reward}, Done: {done}")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Try training for 1 batch
print("\n5. Testing model.learn() with 512 timesteps...")
try:
    initial_params = [p.clone() for p in model.policy.parameters()]
    
    model.learn(total_timesteps=512, log_interval=1)
    print("   ✓ model.learn() completed")
    
    # Check if parameters changed
    params_changed = False
    for p_init, p_curr in zip(initial_params, model.policy.parameters()):
        if not (p_init == p_curr).all():
            params_changed = True
            break
    
    if params_changed:
        print("   ✓ Model parameters WERE updated (training occurred)")
    else:
        print("   ✗ Model parameters NOT updated (no training!)")
        
    print(f"   Model timesteps: {model.num_timesteps}")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
