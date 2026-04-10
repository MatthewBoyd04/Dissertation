#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
import torch

model_path = "joint_ppo_model_8drones.zip"

if os.path.exists(model_path):
    print(f"✓ Model file exists: {model_path}")
    print(f"  File size: {os.path.getsize(model_path)} bytes")
    
    try:
        model = PPO.load(model_path, device="cpu")
        print(f"✓ Model loaded successfully")
        print(f"  Policy type: {type(model.policy).__name__}")
        print(f"  Total parameters: {sum(p.numel() for p in model.policy.parameters())}")
        
        # Check if model has training steps recorded
        print(f"  Num timesteps: {model.num_timesteps}")
        
        # Sample some weights to see if they're random or learned
        first_layer_weights = list(model.policy.parameters())[0]
        print(f"  First layer shape: {first_layer_weights.shape}")
        print(f"  First layer weight range: [{first_layer_weights.min():.4f}, {first_layer_weights.max():.4f}]")
        print(f"  First layer weight mean: {first_layer_weights.mean():.4f}")
        
        # Check if weights look like random init (should be centered near 0)
        if abs(first_layer_weights.mean()) < 0.1 and abs(first_layer_weights.std()) < 1.0:
            print("  → Weights appear to be random initialization (untrained)")
        else:
            print("  → Weights appear to be trained")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"✗ Model file not found: {model_path}")
