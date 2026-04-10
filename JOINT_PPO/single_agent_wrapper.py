import gymnasium as gym
from gymnasium import spaces
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
        
        # Get actions from frozen models for other agents
        for agent in self.env.agents:
            if agent != self.agent_name and agent in self.frozen_models:
                obs = self.env._get_obs([agent])[agent]
                frozen_action, _ = self.frozen_models[agent].predict(obs, deterministic=False)
                actions[agent] = int(frozen_action)
        
        # Step environment with all actions
        obs_dict, rewards, terminations, truncations, infos = self.env.step(actions)
        self.done = terminations.get(self.agent_name, True) or truncations.get(self.agent_name, True)

        # If agent was removed, return dummy observation
        if self.agent_name not in obs_dict:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, rewards.get(self.agent_name, 0.0), self.done, False, {}

        # Return Gym-compatible tuple
        return obs_dict[self.agent_name], rewards[self.agent_name], self.done, False, infos[self.agent_name]

    def render(self):
        if hasattr(self.env, "render_env"):
            self.env.render_env()


class JointAgentWrapper(gym.Env):
    """
    Wrap a PettingZoo ParallelEnv for joint training of all agents with a shared policy.
    Treats the multi-agent environment as a single agent with joint observations and actions.
    Uses possible_agents (not agents) for consistent observation ordering even when agents terminate.
    
    Args:
        env: The base environment to wrap
        use_reward_averaging: If True, averages rewards across agents instead of summing.
                             This prevents reward scale explosion when rewards are found.
    """
    def __init__(self, env, use_reward_averaging=False):
        super().__init__()
        self.env = env
        self.possible_agents = env.possible_agents  # Use possible_agents for consistency
        self.agents = env.possible_agents
        self.use_reward_averaging = use_reward_averaging  # Better credit assignment

        # Joint observation space: concatenation of all agents' observations
        obs_dims = [np.prod(env.observation_spaces[agent].shape) for agent in self.possible_agents]
        total_obs_dim = sum(obs_dims)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32)

        # Joint action space: MultiDiscrete with one sub-action per agent
        if not all(isinstance(env.action_spaces[agent], spaces.Discrete) for agent in self.possible_agents):
            raise ValueError("All agents must have Discrete action spaces for joint training.")
        n_actions_list = [env.action_spaces[agent].n for agent in self.possible_agents]
        self.n_actions_per_agent = n_actions_list
        self.num_agents = len(self.possible_agents)
        self.action_space = spaces.MultiDiscrete(n_actions_list)

    def reset(self, seed=None, options=None):
        obs_dict, _ = self.env.reset(seed=seed, options=options)
        joint_obs = self._concat_observations(obs_dict)
        return joint_obs, {}

    def step(self, joint_action):
        # Convert joint action vector to per-agent actions
        if isinstance(joint_action, np.ndarray) or isinstance(joint_action, list):
            actions = {agent: int(joint_action[i]) for i, agent in enumerate(self.possible_agents)}
        else:
            # If single integer (legacy), decode from base
            actions = {}
            remaining = int(joint_action)
            for i, agent in enumerate(self.possible_agents):
                actions[agent] = remaining % self.n_actions_per_agent[i]
                remaining //= self.n_actions_per_agent[i]

        # Only pass actions for active agents
        active_actions = {agent: actions[agent] for agent in self.env.agents if agent in actions}
        
        obs_dict, rewards, terminations, truncations, infos = self.env.step(active_actions)

        # Always use possible_agents order for consistent observation space
        joint_obs = self._concat_observations(obs_dict)
        
        # Aggregate rewards from all agents for joint training
        # Use averaging instead of summing to prevent reward scale explosion when targets are found
        if self.use_reward_averaging:
            joint_reward = np.mean(list(rewards.values())) if rewards else 0.0
        else:
            joint_reward = sum(rewards.values())
        
        # Episode ends ONLY when:
        # 1. All agents have been removed/terminated (no observations returned), OR
        # 2. All remaining agents are truncated (max steps reached)
        # Individual agent terminations should NOT end the episode - only remove that agent
        no_agents_left = len(obs_dict) == 0
        all_truncated = all(truncations.values()) if truncations else False
        joint_done = no_agents_left or all_truncated
        
        joint_info = {agent: infos.get(agent, {}) for agent in self.possible_agents}

        return joint_obs, joint_reward, joint_done, False, joint_info

    def _concat_observations(self, obs_dict):
        """Concatenate observations in consistent order (possible_agents) padding missing with zeros"""
        obs_list = []
        for agent in self.possible_agents:
            obs = obs_dict.get(agent, np.zeros(self.env.observation_spaces[agent].shape, dtype=np.float32))
            obs_list.append(obs.flatten())
        return np.concatenate(obs_list)

    def render(self):
        if hasattr(self.env, "render_env"):
            self.env.render_env()
