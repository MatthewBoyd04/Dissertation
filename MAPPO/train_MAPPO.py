import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Environment import GridWorldEnvironment
import torch
import random
from LoggerConfig import log
from Maps import map_15x15, map_30x30, map_45x45
import numpy as np
import gymnasium as gym

if torch.cuda.is_available():
    device = "cuda"
    log.i(f"CUDA GPU available: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    log.i("No GPU available, using CPU")

mappo_dir = os.path.dirname(os.path.abspath(__file__))

class MultiAgentWrapper(gym.Env):
    """Wrapper that collects experiences from ALL agents for training"""
    def __init__(self, base_env):
        super().__init__()
        self.env = base_env
        first_agent = base_env.possible_agents[0]
        self.action_space = base_env.action_spaces[first_agent]
        self.observation_space = base_env.observation_spaces[first_agent]
        self.episode_reward = 0.0
        self.episode_length = 0
        self.current_agent_idx = 0
        
    def reset(self, seed=None, options=None):
        obs_dict, _ = self.env.reset(seed=seed, options=options)
        self.current_obs_dict = obs_dict
        self.episode_reward = 0.0
        self.episode_length = 0
        self.current_agent_idx = 0
        # Return first agent's observation
        return obs_dict[self.env.agents[0]], {}
    
    def step(self, action):
        """Collect experience from current agent, rotate through all agents"""
        if not self.env.agents:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, {}
        
        # Get current agent
        current_agent = self.env.agents[self.current_agent_idx % len(self.env.agents)]
        
        # Build actions: current agent uses policy, others random
        actions = {}
        for agent in self.env.agents:
            if agent == current_agent:
                actions[agent] = int(action)
            else:
                actions[agent] = self.action_space.sample()
        
        obs_dict, rewards, terminations, truncations, infos = self.env.step(actions)
        self.current_obs_dict = obs_dict
        
        # Use current agent's reward
        reward = rewards.get(current_agent, 0.0)
        self.episode_reward += reward
        self.episode_length += 1
        
        done = len(self.env.agents) == 0
        
        if done:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            infos['episode'] = {'r': self.episode_reward, 'l': self.episode_length}
        else:
            # Rotate to next agent for next step
            self.current_agent_idx = (self.current_agent_idx + 1) % len(self.env.agents)
            obs = obs_dict[self.env.agents[self.current_agent_idx]]
        
        return obs, reward, done, False, infos

class MAPPOWrapper(gym.Env):
    """Wrapper for evaluation/simulation"""
    def __init__(self, env, num_agents=4):
        super().__init__()
        self.env = env
        first_agent = env.possible_agents[0]
        self.action_space = env.action_spaces[first_agent]
        self.observation_space = env.observation_spaces[first_agent]
        self.num_agents = num_agents
        self._model = None
    
    def reset(self, seed=None, options=None):
        obs_dict, _ = self.env.reset(seed=seed, options=options)
        self.current_obs_dict = obs_dict
        return obs_dict[self.env.agents[0]], {}
    
    def step(self, action):
        actions = {}
        obs_dict = self.env._get_obs()
        
        for agent in self.env.agents:
            obs = obs_dict[agent]
            with torch.no_grad():
                if self._model is not None:
                    action_pred, _ = self._model.policy.predict(obs, deterministic=False)
                    actions[agent] = int(action_pred)
                else:
                    actions[agent] = self.action_space.sample()
        
        obs_dict, rewards, terminations, truncations, infos = self.env.step(actions)
        self.current_obs_dict = obs_dict
        
        first_agent = list(actions.keys())[0]
        reward = rewards.get(first_agent, 0)
        done = len(self.env.agents) == 0
        
        if self.env.agents:
            obs = obs_dict[self.env.agents[0]]
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        return obs, reward, done, False, infos
    
    def set_model(self, model):
        self._model = model

def make_env(map_choice, maxCycles, num_drones):
    def _init():
        base_env = GridWorldEnvironment(
            mapPreset=map_choice,
            maxCycles=maxCycles,
            visionRange=2,
            render_every=None,
            use_map_memory=True,
            num_drones=num_drones
        )
        return MultiAgentWrapper(base_env)
    return _init

def trainAgents(total_timesteps, num_drones=4, cumulativeTimestepsSoFar=0, total_training_timesteps=50_000_000, force_map=None):
    """Train shared MAPPO policy"""
    if force_map is not None:
        map_choice = force_map
    else:
        map_choice = getMapChoice("cirriculum_Random", cumulativeTimestepsSoFar)
    
    if map_choice is map_15x15:
        maxCycles = 256  # Increased from 128
    elif map_choice is map_30x30:
        maxCycles = 1024  # Increased from 512
    else:
        maxCycles = 2048  # Increased from 1024
    
    n_envs = 8
    env = DummyVecEnv([make_env(map_choice, maxCycles, num_drones) for _ in range(n_envs)])
    
    model_path = os.path.join(mappo_dir, "shared_mappo_model.zip")
    vecnorm_path = os.path.join(mappo_dir, "vecnormalize.pkl")
    
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False  # Freeze normalization stats after 2M timesteps
        if cumulativeTimestepsSoFar < 2_000_000:
            env.training = True
        log.i(f"Loaded VecNormalize (training={env.training})")
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        log.i("Created VecNormalize")
    
    progress = cumulativeTimestepsSoFar / total_training_timesteps
    current_lr = 3e-4 * max(0.1, 1 - 0.8 * progress)  # Faster decay, min 10%
    
    if os.path.exists(model_path):
        log.i("Loading model")
        model = PPO.load(model_path, env=env, device=device)
        model.learning_rate = current_lr
        model.n_epochs = 4  # Update existing model's n_epochs
    else:
        log.i("Creating model")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=current_lr,
            n_steps=8192,
            batch_size=2048,
            n_epochs=4,  # Reduced from 10 to prevent overfitting
            ent_coef=0.01,
            clip_range=0.2, 
            gamma=0.99,
            gae_lambda=0.95,
            max_grad_norm=0.5,
            vf_coef=0.5,
            tensorboard_log=os.path.join(mappo_dir, "mappo_tensorboard"),
            device=device
        )
    
    log.i(f"Training with {num_drones} agents, {n_envs} parallel envs")
    model.learn(total_timesteps=total_timesteps)
    model.save(os.path.join(mappo_dir, "shared_mappo_model"))
    env.save(vecnorm_path)
    log.i("Training complete")
    env.close()

def getMapChoice(selectionMethod, total_timesteps_so_far):
    """Get map choice based on selection method"""
    if selectionMethod == "random":
        return random.choice([map_15x15, map_30x30, map_45x45])
    elif selectionMethod == "cirriculum":
        if total_timesteps_so_far < 500_000:
            return map_15x15
        elif total_timesteps_so_far < 5_000_000:
            return map_30x30
        else:
            return map_45x45
    elif selectionMethod == "cirriculum_Random":
        if total_timesteps_so_far < 500_000:
            return map_15x15
        elif total_timesteps_so_far < 2_500_000:
            return random.choice([map_15x15, map_30x30, map_30x30, map_30x30])
        else:
            return random.choice([map_30x30, map_45x45, map_45x45, map_45x45])
    else:
        raise ValueError("Invalid selection method")