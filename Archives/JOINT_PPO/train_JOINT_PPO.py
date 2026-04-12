import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Archives.JOINT_PPO.single_agent_wrapper import JointAgentWrapper
from Archives.JOINT_PPO.Environment import GridWorldEnvironment
import torch
import random
import gymnasium as gym
from LoggerConfig import log
from Maps import map_15x15, map_30x30, map_45x45


# Check GPU availability
if torch.cuda.is_available():
    device = "cuda"
    log.i(f"CUDA GPU available: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.version, 'hip') and torch.version.hip is not None:
    device = "cuda"  # ROCm uses 'cuda' as device string
    log.i(f"ROCm GPU available")
else:
    device = "cpu"
    log.i("No GPU available, using CPU")

# Get JOINT_PPO directory path
joint_ppo_dir = os.path.dirname(os.path.abspath(__file__))

class CurriculumWrapper(gym.Env):
    """Wrapper that implements curriculum learning by selecting maps per episode"""
    
    def __init__(self, base_env_class, cumulativeTimestepsSoFar, total_training_timesteps, num_drones):
        self.base_env_class = base_env_class
        self.cumulativeTimestepsSoFar = cumulativeTimestepsSoFar
        self.total_training_timesteps = total_training_timesteps
        self.num_drones = num_drones
        self.env = None
        self._create_new_env()
        
        # Copy spaces from the current env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
    
    def _create_new_env(self):
        # Smooth curriculum learning: gradually transition through map difficulties
        if self.cumulativeTimestepsSoFar < 1_000_000:
            # Stage 1: Pure 15x15 (0-1M steps)
            map_choice = map_15x15
            maxCycles = 512
        elif self.cumulativeTimestepsSoFar < 1_500_000:
            # Stage 2a: Gentle 30x30 intro (1M-1.5M steps)
            map_choice = random.choice([map_15x15] * 95 + [map_30x30] * 5)  # 5% 30x30
            maxCycles = 512 if map_choice is map_15x15 else 1024
        elif self.cumulativeTimestepsSoFar < 2_000_000:
            # Stage 2b: Gradual increase (1.5M-2M steps)
            map_choice = random.choice([map_15x15] * 85 + [map_30x30] * 15)  # 15% 30x30
            maxCycles = 512 if map_choice is map_15x15 else 1024
        elif self.cumulativeTimestepsSoFar < 3_000_000:
            # Stage 3: Balanced increase (2M-3M steps)
            map_choice = random.choice([map_15x15] * 70 + [map_30x30] * 30)  # 30% 30x30
            maxCycles = 512 if map_choice is map_15x15 else 1024
        elif self.cumulativeTimestepsSoFar < 4_000_000:
            # Stage 4: Even split (3M-4M steps)
            map_choice = random.choice([map_15x15] * 50 + [map_30x30] * 50)  # 50% 30x30
            maxCycles = 512 if map_choice is map_15x15 else 1024
        elif self.cumulativeTimestepsSoFar < 5_000_000:
            # Stage 5a: 30x30 heavy with 45x45 intro (4M-5M steps)
            map_choice = random.choice([map_15x15] * 20 + [map_30x30] * 70 + [map_45x45] * 10)  # 70% 30x30, 10% 45x45
            maxCycles = 512 if map_choice is map_15x15 else 1024 if map_choice is map_30x30 else 2048
        elif self.cumulativeTimestepsSoFar < 6_000_000:
            # Stage 5b: Gradual 45x45 increase (5M-6M steps)
            map_choice = random.choice([map_30x30] * 70 + [map_45x45] * 30)  # 70% 30x30, 30% 45x45
            maxCycles = 1024 if map_choice is map_30x30 else 2048
        elif self.cumulativeTimestepsSoFar < 7_000_000:
            # Stage 6: More 45x45 (6M-7M steps)
            map_choice = random.choice([map_30x30] * 60 + [map_45x45] * 40)  # 60% 30x30, 40% 45x45
            maxCycles = 1024 if map_choice is map_30x30 else 2048
        elif self.cumulativeTimestepsSoFar < 8_000_000:
            # Stage 7: Balanced 30x30/45x45 (7M-8M steps)
            map_choice = random.choice([map_30x30] * 50 + [map_45x45] * 50)  # 50/50 split
            maxCycles = 1024 if map_choice is map_30x30 else 2048
        elif self.cumulativeTimestepsSoFar < 9_000_000:
            # Stage 8: 45x45 focus begins (8M-9M steps)
            map_choice = random.choice([map_30x30] * 40 + [map_45x45] * 60)  # 40% 30x30, 60% 45x45
            maxCycles = 1024 if map_choice is map_30x30 else 2048
        elif self.cumulativeTimestepsSoFar < 10_000_000:
            # Stage 9: Heavy 45x45 training (9M-10M steps)
            map_choice = random.choice([map_30x30] * 30 + [map_45x45] * 70)  # 30% 30x30, 70% 45x45
            maxCycles = 1024 if map_choice is map_30x30 else 2048
        else:
            # Stage 10: Frequent 45x45 (10M+ steps)
            map_choice = random.choice([map_30x30] * 25 + [map_45x45] * 75)  # 25% 30x30, 75% 45x45
            maxCycles = 1024 if map_choice is map_30x30 else 2048
        
        base_env = self.base_env_class(
            mapPreset=map_choice,
            maxCycles=maxCycles,
            visionRange=2,
            render_every=None,
            use_map_memory=True,
            num_drones=self.num_drones
        )
        self.env = JointAgentWrapper(base_env, use_reward_averaging=True)
        
        # Update spaces in case they changed
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
    
    def reset(self, seed=None, options=None):
        # Create new environment with potentially different map for curriculum learning
        self._create_new_env()
        return self.env.reset(seed, options)
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
    
    def close(self):
        if self.env:
            self.env.close()


def trainJointAgent(total_timesteps, num_drones, cumulativeTimestepsSoFar=0, total_training_timesteps=10_000_000):
    """Train a single shared agent for all drones using Joint PPO"""
    
    def make_curriculum_env():
        return CurriculumWrapper(
            GridWorldEnvironment, 
            cumulativeTimestepsSoFar, 
            total_training_timesteps, 
            num_drones
        )

    n_envs = 8
    env = DummyVecEnv([make_curriculum_env for _ in range(n_envs)])
    # Normalize observations but NOT rewards to preserve reward signal structure
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model_path = os.path.join(joint_ppo_dir, f"joint_ppo_model_{num_drones}drones.zip")
    vec_normalize_path = os.path.join(joint_ppo_dir, f"joint_ppo_vec_normalize_{num_drones}drones.pkl")
    save_model_path = os.path.join(joint_ppo_dir, f"joint_ppo_model_{num_drones}drones")

    # Calculate learning rate with decay
    progress = cumulativeTimestepsSoFar / total_training_timesteps
    current_lr = 3e-4 * max(0.1, (1.0 - progress))  # linear decay, min 10%
    
    # Load existing model or create new one
    model = None
    if os.path.exists(model_path):
        log.i(f"Found existing joint model: {model_path}")
        try:
            # Load WITHOUT env first to check compatibility
            model_to_check = PPO.load(model_path, device="cpu")
            log.i(f"  Model loaded. num_timesteps: {model_to_check.num_timesteps}")
            
            # Now load with new env for continued training
            model = PPO.load(model_path, env=env, device="cpu")
            log.i(f"[OK] Successfully loaded existing model and attached new env")
        except Exception as e:
            log.w(f"Could not load saved model: {e}")
            import traceback
            traceback.print_exc()
            model = None

    if model is None:
        log.i("Creating NEW joint model with standard MlpPolicy")
        log.i(f"  Observation space: {env.observation_space}")
        log.i(f"  Action space: {env.action_space}")
        model = PPO(
            "MlpPolicy",  # Use standard PPO policy for centralized training
            env,
            verbose=1,
            learning_rate=current_lr,
            n_steps=2048,  # Reduced from 4096 (was causing insufficient data collection with shared policy)
            batch_size=256,  # Reduced from 512 (was too large relative to n_steps)
            gamma=0.99,
            clip_range=0.2,
            ent_coef=0.01,  # Increased entropy to encourage more exploration
            gae_lambda=0.95,
            tensorboard_log=os.path.join(joint_ppo_dir, "joint_ppo_tensorboard"),
            device="cpu"
        )
    
    log.i("Training joint agent")
    log.i(f"  Model timesteps BEFORE training: {model.num_timesteps}")
    
    # Get initial weights for verification
    initial_params = {name: param.data.clone() for name, param in model.policy.named_parameters()}
    
    try:
        model.learn(total_timesteps=total_timesteps, log_interval=10)
        log.i(f"[OK] model.learn() completed successfully")
    except Exception as e:
        log.e(f"[ERROR] Error during model.learn(): {e}")
        import traceback
        traceback.print_exc()
        raise
    
    log.i(f"  Model timesteps AFTER training: {model.num_timesteps}")
    
    # Verify weights were updated
    params_changed = False
    for name, param in model.policy.named_parameters():
        if name in initial_params:
            if not (initial_params[name] == param.data).all():
                params_changed = True
                log.i(f"  [OK] Parameter '{name[:30]}' was updated")
                break
    
    if not params_changed:
        log.w(f"  [WARNING] Model weights were NOT updated during training!")
    else:
        log.i(f"  [OK] Model weights successfully updated")
    
    model.save(save_model_path)
    log.i(f"  Model saved to {save_model_path}.zip")
    
    env.save(vec_normalize_path)  # Save normalization stats
    log.i(f"  VecNormalize stats saved to {vec_normalize_path}")
    
    log.i(f"[OK] Finished training joint agent")

def trainAgents(total_timesteps, num_drones=4, parallel=True, cumulativeTimestepsSoFar=0, total_training_timesteps=10_000_000):
    # For Joint PPO, we train a single shared model
    trainJointAgent(total_timesteps, num_drones, cumulativeTimestepsSoFar, total_training_timesteps)

def getMapChoice(selectionMethod, total_timesteps_so_far):
    """Get map choice based on selection method"""
    if selectionMethod == "random":
        return random.choice([map_15x15, map_30x30, map_45x45])
    
    elif selectionMethod == "cirriculum": #Presuming 10_000_000 timestps total, 1_000_000 for 15x15, 4_000_000 for 30x30, 5_000_000 for 45x45
        if total_timesteps_so_far < 500_000: #Convergence normally occurs before 500_000 timesteps for 15x15, so we can start with that
            return map_15x15
        elif total_timesteps_so_far < 5_000_000:
            return map_30x30
        else:
            return map_45x45
        
    elif selectionMethod == "cirriculum_Random": #Start with cirriculum, but randomize between maps within each stage
        if total_timesteps_so_far < 2_500_000:
            return map_15x15
        elif total_timesteps_so_far < 10_000_000:
            return random.choice([map_15x15, map_30x30 , map_30x30 , map_30x30 , map_30x30]) #80% chance of 30x30, 20% chance of 15x15 to allow some continued training on easier map
        else:
            return random.choice([map_15x15, map_30x30, map_45x45 , map_45x45, map_45x45, map_45x45 , map_45x45 , map_45x45 , map_45x45 , map_45x45]) #
    else:
        raise ValueError("Invalid selection method")


