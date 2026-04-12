import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance
from Environment import GridWorldEnvironment
from centralized_critic import MAPPOPolicy
from multi_agent_buffer import MultiAgentRolloutBuffer
import torch
import torch.nn.functional as F
import random
from LoggerConfig import log
from Maps import map_15x15, map_30x30, map_45x45
import numpy as np
import gymnasium as gym
from gymnasium import spaces

if torch.cuda.is_available():
    device = "cuda"
    log.i(f"CUDA GPU available: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    log.i("No GPU available, using CPU")

mappo_dir = os.path.dirname(os.path.abspath(__file__))

class MultiAgentWrapper(gym.Env):
    """Wrapper that collects experiences from ALL agents using trained policy"""
    def __init__(self, base_env):
        super().__init__()
        self.env = base_env
        first_agent = base_env.possible_agents[0]
        self.action_space = base_env.action_spaces[first_agent]
        self.observation_space = base_env.observation_spaces[first_agent]
        self.episode_reward = 0.0
        self.episode_length = 0
        self.current_agent_idx = 0
        self._model = None
        self._vec_normalize = None  # for normalising co-agent observations

    def reset(self, seed=None, options=None):
        obs_dict, _ = self.env.reset(seed=seed, options=options)
        self.current_obs_dict = obs_dict
        self.episode_reward = 0.0
        self.episode_length = 0
        self.current_agent_idx = 0
        # Return first agent's observation
        return obs_dict[self.env.agents[0]], {}
    
    def step(self, action):
        """FIX 1: Use trained policy for ALL agents, not just one"""
        if not self.env.agents:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, {}

        # Build centralized obs from PRE-STEP observations (all agents' current obs concatenated
        # along the channel axis). This is used by the centralized critic during training.
        all_obs = [
            self.current_obs_dict.get(a, np.zeros(self.observation_space.shape, dtype=np.float32))
            for a in self.env.possible_agents
        ]
        centralized_obs = np.concatenate(all_obs, axis=0)  # (channels * num_agents, H, W)

        # Get current agent (for returning observations in rotation)
        current_agent = self.env.agents[self.current_agent_idx % len(self.env.agents)]

        # Build actions: current agent gets the action from the training step,
        # all other agents use policy prediction
        actions = {}
        actions[current_agent] = int(action)

        # All other agents use the policy (if model is available)
        for agent in self.env.agents:
            if agent != current_agent:
                if self._model is not None:
                    agent_obs = self.current_obs_dict.get(agent, np.zeros(self.observation_space.shape, dtype=np.float32))
                    agent_obs = self._normalize_obs(agent_obs)
                    with torch.no_grad():
                        action_pred, _ = self._model.policy.predict(agent_obs, deterministic=False)
                        actions[agent] = int(action_pred)
                else:
                    # Fallback to random if model not set (during initialization)
                    actions[agent] = self.action_space.sample()

        obs_dict, rewards, terminations, truncations, step_infos = self.env.step(actions)
        self.current_obs_dict = obs_dict

        # Use current agent's reward
        reward = rewards.get(current_agent, 0.0)
        self.episode_reward += reward
        self.episode_length += 1

        done = len(self.env.agents) == 0

        # Build a single gym-compatible infos dict and inject centralized obs for CTDE
        infos = {'centralized_obs': centralized_obs}

        if done:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            infos['episode'] = {'r': self.episode_reward, 'l': self.episode_length}
        else:
            # Rotate to next agent for next step
            self.current_agent_idx = (self.current_agent_idx + 1) % len(self.env.agents)
            obs = obs_dict[self.env.agents[self.current_agent_idx]]

        return obs, reward, done, False, infos
    
    def set_model(self, model):
        """Set reference to trained model for multi-agent action prediction"""
        self._model = model

    def set_obs_normalizer(self, vec_normalize):
        """Store VecNormalize reference so co-agent observations are normalised correctly"""
        self._vec_normalize = vec_normalize

    def _normalize_obs(self, obs):
        if self._vec_normalize is None:
            return obs
        vn = self._vec_normalize
        obs = (obs.copy() - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon)
        return np.clip(obs, -vn.clip_obs, vn.clip_obs)

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
        # Use observations from the previous step (stored in reset/last step) for action selection,
        # not a fresh _get_obs() call which would be out-of-sync with the current state.
        for agent in self.env.agents:
            obs = self.current_obs_dict.get(agent)
            with torch.no_grad():
                if self._model is not None and obs is not None:
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

class CentralizedObsCallback(BaseCallback):
    """Injects centralized observations (from infos) into the rollout buffer at each step.
    Required for CTDE: the centralized critic sees all agents' obs during training."""

    def _on_step(self) -> bool:
        if not hasattr(self.model.rollout_buffer, 'centralized_observations'):
            return True
        # SB3 has just added the transition at position (pos - 1)
        pos = self.model.rollout_buffer.pos - 1
        if pos < 0:
            return True
        infos = self.locals.get('infos', [])
        for env_idx, info in enumerate(infos):
            c_obs = info.get('centralized_obs') if isinstance(info, dict) else None
            if c_obs is not None:
                n_envs = self.model.rollout_buffer.centralized_observations.shape[1]
                if env_idx < n_envs:
                    self.model.rollout_buffer.centralized_observations[pos, env_idx] = c_obs
        return True


class MAPPOPPO(PPO):
    """PPO subclass that uses MultiAgentRolloutBuffer with the CentralizedCritic.
    Overrides _setup_model() to install the multi-agent buffer and train() to pass
    centralized observations to MAPPOPolicy.evaluate_actions()."""

    def __init__(self, policy, env, num_agents: int = 4, **kwargs):
        self._num_agents = num_agents
        super().__init__(policy, env, **kwargs)

    def _setup_model(self) -> None:
        # SB3's load() updates __dict__ with kwargs BEFORE calling _setup_model(),
        # so 'num_agents' may have been injected as a plain dict key rather than _num_agents.
        if hasattr(self, 'num_agents'):
            self._num_agents = self.num_agents
        super()._setup_model()
        # Replace the standard RolloutBuffer with one that stores centralized observations
        self.rollout_buffer = MultiAgentRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
            n_envs=self.n_envs,
            num_agents=self._num_agents,
        )

    def train(self) -> None:
        """PPO training loop with centralized critic support."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, pg_losses, value_losses, clip_fractions, approx_kl_divs = [], [], [], [], []
        continue_training = True

        for epoch in range(self.n_epochs):
            epoch_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # MultiAgentRolloutBuffer._get_samples returns a 7-tuple:
                # (obs, actions, old_values, old_log_prob, advantages, returns, centralized_obs)
                obs_t, actions_t, old_values, old_log_prob, advantages, returns, centralized_obs_t = rollout_data

                if isinstance(self.action_space, spaces.Discrete):
                    actions_t = actions_t.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    obs_t, actions_t, centralized_obs=centralized_obs_t
                )
                values = values.flatten()

                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(torch.mean((torch.abs(ratio - 1) > clip_range).float()).item())

                values_pred = (
                    values if self.clip_range_vf is None
                    else old_values + torch.clamp(values - old_values, -clip_range_vf, clip_range_vf)
                )
                value_loss = F.mse_loss(returns, values_pred)
                value_losses.append(value_loss.item())

                entropy_loss = -torch.mean(entropy) if entropy is not None else -torch.mean(-log_prob)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with torch.no_grad():
                    log_ratio = log_prob - old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    epoch_kl_divs.append(approx_kl_div)
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs) if approx_kl_divs else 0.0)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


def make_env(map_choice, maxCycles, num_drones, vision_range=3):
    def _init():
        base_env = GridWorldEnvironment(
            mapPreset=map_choice,
            maxCycles=maxCycles,
            visionRange=vision_range,
            use_map_memory=True,
            num_drones=num_drones
        )
        return MultiAgentWrapper(base_env)
    return _init

def trainAgents(total_timesteps, num_drones=4, cumulativeTimestepsSoFar=0, total_training_timesteps=50_000_000, force_map=None, vision_range=3):
    """Train shared MAPPO policy"""
    if force_map is not None:
        map_choice = force_map
    else:
        map_choice = getMapChoice("cirriculum_Random", cumulativeTimestepsSoFar, total_training_timesteps)
    
    if map_choice is map_15x15:
        maxCycles = 256
    elif map_choice is map_30x30:
        maxCycles = 512
    else:
        maxCycles = 1024
    
    n_envs = 8
    env = DummyVecEnv([make_env(map_choice, maxCycles, num_drones, vision_range) for _ in range(n_envs)])
    
    model_path = os.path.join(mappo_dir, "shared_mappo_model.zip")
    vecnorm_path = os.path.join(mappo_dir, "vecnormalize.pkl")
    
    # FIX 2: Keep VecNormalize updating throughout training + normalize obs only
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = True  # FIXED: Keep training throughout to adapt to distribution shifts
        log.i("Loaded VecNormalize (training=True - adapts throughout training)")
    else:
        # FIXED: norm_reward=False to preserve carefully tuned reward weights
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        log.i("Created VecNormalize (obs only - preserves reward shaping)")
    
    progress = cumulativeTimestepsSoFar / total_training_timesteps
    current_lr = 3e-4 * max(0.1, 1 - 0.8 * progress)  # Faster decay, min 10%
    
    # FIX 3: Entropy decay callback to gradually reduce exploration
    class EntropyDecayCallback(BaseCallback):
        def __init__(self, initial_ent=0.1, final_ent=0.03, cumulative_steps_so_far=0, total_training_steps=10_000_000):
            super().__init__()
            self.initial_ent = initial_ent
            self.final_ent = final_ent
            self.cumulative_steps_so_far = cumulative_steps_so_far
            self.total_training_steps = total_training_steps
            self.last_log = 0

        def _on_step(self) -> bool:
            global_steps = self.cumulative_steps_so_far + self.num_timesteps
            progress_frac = global_steps / max(self.total_training_steps, 1)
            new_ent = max(self.final_ent, self.initial_ent * (1 - progress_frac))
            self.model.ent_coef = new_ent

            if self.num_timesteps - self.last_log > 100000:
                log.i(f"Entropy: {new_ent:.6f} at global step {global_steps}")
                self.last_log = self.num_timesteps
            return True
    
    # FIX 5 & 8: Diagnostics callback to monitor training
    class DiagnosticsCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.last_log = 0
        
        def _on_step(self) -> bool:
            if self.num_timesteps - self.last_log > 100000:
                log.i(f"LR: {self.model.learning_rate:.6f} | Ent: {self.model.ent_coef:.6f} | Steps: {self.num_timesteps}")
                self.last_log = self.num_timesteps
            return True
    
    policy_kwargs = {"num_agents": num_drones}

    if os.path.exists(model_path):
        log.i("Loading existing MAPPOPPO model")
        try:
            model = MAPPOPPO.load(model_path, env=env, device=device, num_agents=num_drones)
            if not isinstance(model.policy, MAPPOPolicy):
                raise ValueError("Loaded model uses old policy format; creating fresh MAPPOPPO")
        except Exception as load_err:
            log.i(f"Could not load as MAPPOPPO ({load_err}); creating fresh model")
            model = MAPPOPPO(
                MAPPOPolicy, env, num_agents=num_drones,
                verbose=1, learning_rate=current_lr, n_steps=2048,
                batch_size=512, n_epochs=4, ent_coef=0.1,
                clip_range=0.2, gamma=0.99, gae_lambda=0.95,
                max_grad_norm=1.0, vf_coef=0.5,
                policy_kwargs=policy_kwargs,
                tensorboard_log=os.path.join(mappo_dir, "mappo_tensorboard"),
                device=device
            )
        model.learning_rate = current_lr
        model.n_epochs = 4
        ent_progress = cumulativeTimestepsSoFar / max(total_training_timesteps, 1)
        model.ent_coef = max(0.03, 0.1 * (1 - ent_progress))
    else:
        log.i("Creating new MAPPOPPO model with CentralizedCritic (CTDE)")
        model = MAPPOPPO(
            MAPPOPolicy,
            env,
            num_agents=num_drones,
            verbose=1,
            learning_rate=current_lr,
            n_steps=2048,
            batch_size=512,
            n_epochs=4,
            ent_coef=0.1,
            clip_range=0.2,
            gamma=0.99,
            gae_lambda=0.95,
            max_grad_norm=1.0,
            vf_coef=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=os.path.join(mappo_dir, "mappo_tensorboard"),
            device=device
        )

    # Set model and obs normalizer in wrappers so co-agent actions use correct obs scale
    for i in range(n_envs):
        env.envs[i].set_model(model)
        env.envs[i].set_obs_normalizer(env)

    log.i(f"Training with {num_drones} agents, {n_envs} parallel envs (CTDE enabled)")
    log.i(f"CONFIG: Batch=512 | Epochs=10 | Entropy=0.1->0.01 decay | MaxGradNorm=1.0 | RewardNorm=OFF | CentralizedCritic=ON")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[
            CentralizedObsCallback(),
            EntropyDecayCallback(initial_ent=0.1, final_ent=0.03, cumulative_steps_so_far=cumulativeTimestepsSoFar, total_training_steps=total_training_timesteps),
            DiagnosticsCallback()
        ]
    )
    model.save(os.path.join(mappo_dir, "shared_mappo_model"))
    env.save(vecnorm_path)
    log.i("Training complete")
    env.close()

def getMapChoice(selectionMethod, total_timesteps_so_far, total_training_timesteps=50_000_000):
    """Get map choice based on selection method.
    Uses fractional progress so the schedule adapts if total training budget changes."""
    progress = total_timesteps_so_far / max(total_training_timesteps, 1)

    if selectionMethod == "random":
        return random.choice([map_15x15, map_30x30, map_45x45])
    elif selectionMethod == "cirriculum":
        if progress < 0.01:    # First 1%: pure 15x15
            return map_15x15
        elif progress < 0.10:  # 1-10%: pure 30x30
            return map_30x30
        else:                  # 10%+: pure 45x45
            return map_45x45
    elif selectionMethod == "cirriculum_Random":
        # Absolute-timestep milestones (consistent across IPPO and MAPPO)
        T1, T2, T3, T4 = 500_000, 1_000_000, 1_500_000, 2_000_000
        steps = total_timesteps_so_far

        if steps < T1:
            w15, w30, w45 = 1.0, 0.0, 0.0
        elif steps < T2:
            t = (steps - T1) / (T2 - T1)
            w15, w30, w45 = 1.0 - 0.5 * t, 0.5 * t, 0.0
        elif steps < T3:
            w15, w30, w45 = 0.5, 0.5, 0.0
        elif steps < T4:
            t = (steps - T3) / (T4 - T3)
            w15, w30, w45 = 0.5 - 0.25 * t, 0.5 - 0.25 * t, 0.5 * t
        else:
            w15, w30, w45 = 0.25, 0.25, 0.50

        return random.choices(
            [map_15x15, map_30x30, map_45x45],
            weights=[w15, w30, w45],
            k=1
        )[0]
    else:
        raise ValueError("Invalid selection method")