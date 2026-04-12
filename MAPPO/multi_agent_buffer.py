import numpy as np
import torch
from stable_baselines3.common.buffers import RolloutBuffer
from gymnasium import spaces


class MultiAgentRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer that stores transitions from multiple agents.
    Collects N agent transitions per environment step for efficiency.
    """
    
    def __init__(self, buffer_size, observation_space, action_space, device="cpu", 
                 gae_lambda=0.95, gamma=0.99, n_envs=1, num_agents=4):
        # Multiply buffer size by num_agents to store all agent transitions
        # Use the same buffer_size as the parent — standard SB3 collect_rollouts
        # adds one entry per call, so we need exactly n_steps entries to fill.
        # (The old * num_agents multiplier was for the now-unused add_multi_agent() path.)
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs
        )
        self.num_agents = num_agents
        # Track previous done flags so episode_starts is correctly one step ahead
        self._prev_dones: dict = {}

        # Add storage for centralized observations
        single_obs_shape = observation_space.shape
        centralized_obs_shape = (single_obs_shape[0] * num_agents, *single_obs_shape[1:])
        self._centralized_obs_shape = centralized_obs_shape  # store for reset()
        self.centralized_observations = np.zeros(
            (self.buffer_size, n_envs, *centralized_obs_shape),
            dtype=np.float32
        )

    def reset(self) -> None:
        """Re-initialise centralized_observations after swap_and_flatten in get()."""
        super().reset()
        # _centralized_obs_shape is set after super().__init__() calls reset() during init,
        # so guard against the first (early) reset call.
        if hasattr(self, '_centralized_obs_shape'):
            self.centralized_observations = np.zeros(
                (self.buffer_size, self.n_envs, *self._centralized_obs_shape),
                dtype=np.float32
            )

    def add_multi_agent(self, obs_dict, actions_dict, rewards_dict, dones_dict, 
                        values_dict, log_probs_dict, centralized_obs):
        """
        Add transitions from all agents at once.
        
        Args:
            obs_dict: Dict of agent observations
            actions_dict: Dict of agent actions
            rewards_dict: Dict of agent rewards
            dones_dict: Dict of agent done flags
            values_dict: Dict of agent value estimates
            log_probs_dict: Dict of agent log probabilities
            centralized_obs: Centralized observation for critic
        """
        for agent in obs_dict.keys():
            if self.pos >= self.buffer_size:
                break
                
            self.observations[self.pos] = np.array(obs_dict[agent]).copy()
            self.actions[self.pos] = np.array(actions_dict[agent]).copy()
            self.rewards[self.pos] = np.array(rewards_dict[agent]).copy()
            # episode_starts[t] should be True when t is the FIRST step of a new episode,
            # i.e. when the PREVIOUS step ended the episode. Use _prev_dones for this.
            self.episode_starts[self.pos] = np.array(self._prev_dones.get(agent, False)).copy()
            self.values[self.pos] = values_dict[agent].clone().cpu().numpy().flatten()
            self.log_probs[self.pos] = log_probs_dict[agent].clone().cpu().numpy()
            self.centralized_observations[self.pos] = np.array(centralized_obs).copy()
            # Store current done as previous for next call
            self._prev_dones[agent] = bool(dones_dict.get(agent, False))

            self.pos += 1
            if self.pos == self.buffer_size:
                self.full = True
    
    def get(self, batch_size=None):
        """Get all data from buffer with centralized observations"""
        assert self.full, "Buffer must be full before sampling"
        
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        
        # Prepare data
        if not self.generator_ready:
            for tensor in [
                "observations",
                "actions", 
                "values",
                "log_probs",
                "advantages",
                "returns",
                "centralized_observations"
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True
        
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
        
        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
    
    def _get_samples(self, batch_inds):
        """Get samples with centralized observations"""
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.centralized_observations[batch_inds],
        )
        return tuple(map(self.to_torch, data))
