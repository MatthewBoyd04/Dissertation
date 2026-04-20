import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
import numpy as np


class CentralizedCritic(nn.Module):
    """Centralized critic with per-agent CNN encoder + shared MLP.
    Each agent's (C, H, W) observation slice is processed by a shared CNN encoder.
    The resulting feature vectors are concatenated and fed into an MLP value head.
    This preserves the 2D spatial structure of each agent's observation."""
    def __init__(self, observation_space, num_agents, features_dim=128):
        super().__init__()
        self.num_agents = num_agents
        C, H, W = observation_space.shape  # e.g. (6, 7, 7) with vision_range=3
        self._C = C
        self._H = H
        self._W = W

        # Shared CNN encoder — same weights applied independently to each agent's obs slice
        self.agent_encoder = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * H * W, features_dim), nn.ReLU()
        )

        # MLP value head — receives concatenated features from all agents
        self.value_head = nn.Sequential(
            nn.Linear(features_dim * num_agents, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, centralized_obs):
        """
        Args:
            centralized_obs: (batch, C * num_agents, H, W) — all agents' obs concatenated
        Returns:
            value: (batch, 1) centralized value estimate
        """
        batch = centralized_obs.shape[0]
        C, H, W = self._C, self._H, self._W
        # Split into per-agent slices and encode with shared CNN
        agent_slices = centralized_obs.view(batch * self.num_agents, C, H, W)
        agent_features = self.agent_encoder(agent_slices)          # (batch*N, features_dim)
        joint_features = agent_features.view(batch, -1)            # (batch, features_dim*N)
        return self.value_head(joint_features)


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
