import gymnasium as gym
import numpy as np

class SingleAgentWrapper(gym.Env):
    """
    Wrap a PettingZoo ParallelEnv for a single agent so it can be trained with SB3.
    """
    def __init__(self, env, agent_name):
        super().__init__()
        self.env = env
        self.agent_name = agent_name

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
        # Step only the single agent
        obs_dict, rewards, terminations, truncations, infos = self.env.step({self.agent_name: action})
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
