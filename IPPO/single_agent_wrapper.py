import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import os

class SingleAgentWrapper(gym.Env):
    """
    Wrap a PettingZoo ParallelEnv for a single agent so it can be trained with SB3.
    Other agents can be controlled by frozen policies if available.
    """
    def __init__(self, env, agent_name, frozen_models=None):
        super().__init__()
        self.env = env
        self.agent_name = agent_name
        self.frozen_models = frozen_models or {}  # Dict of {agent_name: PPO_model}

        # Set SB3-compatible spaces
        self.action_space = env.action_spaces[agent_name]
        self.observation_space = env.observation_spaces[agent_name]

        # Reset flag
        self.done = False

    def reset(self, seed=None, options=None):
        obs_dict, _ = self.env.reset(seed=seed, options=options)
        self.done = False
        return obs_dict[self.agent_name], {}

    def step(self, action):
        # Collect actions for all agents
        actions = {self.agent_name: action}
        
        # Get actions from frozen models — one _get_obs call for all frozen agents at once
        frozen_agents = [a for a in self.env.agents
                         if a != self.agent_name and a in self.frozen_models]
        if frozen_agents:
            frozen_obs = self.env._get_obs(frozen_agents)
            for agent in frozen_agents:
                frozen_action, _ = self.frozen_models[agent].predict(
                    frozen_obs[agent], deterministic=True)
                actions[agent] = int(frozen_action)
        
        # Step environment with all actions
        obs_dict, rewards, terminations, truncations, infos = self.env.step(actions)
        self.done = terminations.get(self.agent_name, True) or truncations.get(self.agent_name, True)

        # If agent was removed, return a -1.0-filled observation (represents "unseen/unknown")
        # Using zeros would be out-of-range for the [-1, 1] obs space and confuse the policy.
        if self.agent_name not in obs_dict:
            obs = np.full(self.observation_space.shape, -1.0, dtype=np.float32)
            return obs, rewards.get(self.agent_name, 0.0), self.done, False, {}

        # Return Gym-compatible tuple
        return obs_dict[self.agent_name], rewards[self.agent_name], self.done, False, infos[self.agent_name]

    def render(self):
        if hasattr(self.env, "render_env"):
            self.env.render_env()
