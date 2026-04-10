import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
import numpy as np


class CentralizedCritic(nn.Module):
    """Centralized critic that sees all agents' observations"""
    def __init__(self, observation_space, num_agents, features_dim=64):
        super().__init__()
        self.num_agents = num_agents
        
        # Calculate input size: all agents' observations concatenated
        single_obs_shape = observation_space.shape
        total_obs_size = int(np.prod(single_obs_shape)) * num_agents
        
        self.critic_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, centralized_obs):
        """
        Args:
            centralized_obs: Concatenated observations from all agents
        Returns:
            value: Centralized value estimate
        """
        return self.critic_net(centralized_obs)


class MAPPOPolicy(ActorCriticPolicy):
    """MAPPO policy with decentralized actor and centralized critic"""
    
    def __init__(self, observation_space, action_space, lr_schedule, num_agents=4, **kwargs):
        self.num_agents = num_agents
        self.centralized_obs_buffer = []
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        
        # Replace the standard critic with centralized critic
        self.centralized_critic = CentralizedCritic(observation_space, num_agents)
        self.centralized_critic.to(self.device)
    
    def forward(self, obs, deterministic=False):
        """Standard forward for actor (decentralized)"""
        return super().forward(obs, deterministic)
    
    def evaluate_actions(self, obs, actions, centralized_obs=None):
        """Evaluate actions using centralized critic"""
        # Get action distribution from decentralized actor
        distribution = self._get_action_dist_from_latent(self.mlp_extractor.forward_actor(self.extract_features(obs)))
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        # Use centralized critic if available
        if centralized_obs is not None:
            values = self.centralized_critic(centralized_obs)
        else:
            # Fallback to standard critic
            values = self.value_net(self.mlp_extractor.forward_critic(self.extract_features(obs)))
        
        return values, log_prob, entropy
    
    def predict_values(self, obs, centralized_obs=None):
        """Predict values using centralized critic"""
        if centralized_obs is not None:
            return self.centralized_critic(centralized_obs)
        else:
            # Fallback to standard critic
            return self.value_net(self.mlp_extractor.forward_critic(self.extract_features(obs)))
