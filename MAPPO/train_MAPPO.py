import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
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
    def __init__(self, base_env, agent_idx=0):
        super().__init__()
        self.env = base_env
        self.agent_idx = agent_idx
        self.current_agent = base_env.possible_agents[agent_idx]
        first_agent = base_env.possible_agents[0]
        self._num_agents = len(base_env.possible_agents)
        self.action_space = base_env.action_spaces[first_agent]
        base_obs_space = base_env.observation_spaces[first_agent]
        C, H, W = base_obs_space.shape
        self._base_obs_C = C
        # Expand observation space by one channel for the agent's normalised identity (0 → 1)
        self.observation_space = spaces.Box(
            low=np.full((C + 1, H, W), base_obs_space.low.min(), dtype=np.float32),
            high=np.full((C + 1, H, W), base_obs_space.high.max(), dtype=np.float32),
            dtype=np.float32,
        )
        self.episode_reward = 0.0
        self.episode_length = 0
        self._policy = None        # policy object for co-agent inference
        self._obs_mean = None      # obs normalisation stats (pickle-safe numpy)
        self._obs_var = None
        self._obs_eps = 1e-8
        self._obs_clip = 10.0

    def reset(self, seed=None, options=None):
        obs_dict, _ = self.env.reset(seed=seed, options=options)
        self.current_obs_dict = obs_dict
        self.episode_reward = 0.0
        self.episode_length = 0
        _, H, W = self.observation_space.shape
        raw_obs = obs_dict.get(self.current_agent, np.zeros((self._base_obs_C, H, W), dtype=np.float32))
        return self._add_agent_id(raw_obs, self.agent_idx), {}

    def _add_agent_id(self, obs, agent_idx):
        _, H, W = obs.shape
        id_val = agent_idx / max(self._num_agents - 1, 1)
        id_channel = np.full((1, H, W), id_val, dtype=np.float32)
        return np.concatenate([obs, id_channel], axis=0)

    def step(self, action):
        if self.current_agent not in self.env.agents:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, {}

        # Build action dict: training agent uses the provided action,
        # all co-agents use the shared policy prediction (same policy, different obs).
        actions = {self.current_agent: int(action)}
        _, H, W = self.observation_space.shape
        for agent in self.env.agents:
            if agent != self.current_agent:
                if self._policy is not None:
                    co_idx = self.env.possible_agents.index(agent)
                    raw_co_obs = self.current_obs_dict.get(agent, np.zeros((self._base_obs_C, H, W), dtype=np.float32))
                    agent_obs = self._normalize_obs(self._add_agent_id(raw_co_obs, co_idx))
                    with torch.no_grad():
                        action_pred, _ = self._policy.predict(agent_obs, deterministic=False)
                    actions[agent] = int(action_pred)
                else:
                    actions[agent] = self.action_space.sample()

        obs_dict, rewards, terminations, truncations, step_infos = self.env.step(actions)
        self.current_obs_dict = obs_dict

        reward = rewards.get(self.current_agent, 0.0)
        self.episode_reward += reward
        self.episode_length += 1

        done = (
            terminations.get(self.current_agent, False)
            or truncations.get(self.current_agent, False)
            or len(self.env.agents) == 0
        )

        # Build post-step centralized obs from the updated current_obs_dict.
        # Returning it in infos piggybacks on the existing IPC step response,
        # eliminating the need for a separate env_method call in collect_rollouts.
        c_obs = self.get_centralized_obs()

        if done:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            infos = {'episode': {'r': self.episode_reward, 'l': self.episode_length}, 'centralized_obs': c_obs}
        else:
            raw_obs = obs_dict.get(self.current_agent, np.zeros((self._base_obs_C, H, W), dtype=np.float32))
            obs = self._add_agent_id(raw_obs, self.agent_idx)
            infos = {'centralized_obs': c_obs}

        return obs, reward, done, False, infos

    def get_centralized_obs(self):
        _, H, W = self.observation_space.shape
        all_obs = [
            self._normalize_obs(
                self._add_agent_id(
                    self.current_obs_dict.get(a, np.zeros((self._base_obs_C, H, W), dtype=np.float32)),
                    i,
                )
            )
            for i, a in enumerate(self.env.possible_agents)
        ]
        return np.concatenate(all_obs, axis=0)  # (C+1) * num_agents, H, W
    
    def set_policy(self, policy):
        self._policy = policy

    def update_policy_weights(self, cpu_state_dict):
        if self._policy is not None:
            self._policy.load_state_dict(cpu_state_dict)

    def set_obs_stats(self, mean, var, epsilon, clip_obs):
        self._obs_mean = mean
        self._obs_var  = var
        self._obs_eps  = epsilon
        self._obs_clip = clip_obs

    def _normalize_obs(self, obs):
        if self._obs_mean is None:
            return obs
        obs = (obs.copy() - self._obs_mean) / np.sqrt(self._obs_var + self._obs_eps)
        return np.clip(obs, -self._obs_clip, self._obs_clip)

class MAPPOWrapper(gym.Env):
    """Wrapper for evaluation/simulation"""
    def __init__(self, env, num_agents=4):
        super().__init__()
        self.env = env
        first_agent = env.possible_agents[0]
        self._num_agents = len(env.possible_agents)
        self.action_space = env.action_spaces[first_agent]
        base_obs_space = env.observation_spaces[first_agent]
        C, H, W = base_obs_space.shape
        self._base_obs_C = C
        # Match expanded obs space used during training
        self.observation_space = spaces.Box(
            low=np.full((C + 1, H, W), base_obs_space.low.min(), dtype=np.float32),
            high=np.full((C + 1, H, W), base_obs_space.high.max(), dtype=np.float32),
            dtype=np.float32,
        )
        self.num_agents = num_agents
        self._model = None

    def _add_agent_id(self, obs, agent_idx):

        _, H, W = obs.shape
        id_val = agent_idx / max(self._num_agents - 1, 1)
        id_channel = np.full((1, H, W), id_val, dtype=np.float32)
        return np.concatenate([obs, id_channel], axis=0)

    def reset(self, seed=None, options=None):
        obs_dict, _ = self.env.reset(seed=seed, options=options)
        self.current_obs_dict = obs_dict
        first_agent = self.env.agents[0]
        agent_idx = self.env.possible_agents.index(first_agent)
        return self._add_agent_id(obs_dict[first_agent], agent_idx), {}

    def step(self, action):
        actions = {}
        # Use observations from the previous step (stored in reset/last step) for action selection,
        # not a fresh _get_obs() call which would be out-of-sync with the current state.
        for agent in self.env.agents:
            raw_obs = self.current_obs_dict.get(agent)
            with torch.no_grad():
                if self._model is not None and raw_obs is not None:
                    agent_idx = self.env.possible_agents.index(agent)
                    obs = self._add_agent_id(raw_obs, agent_idx)
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
            first_live = self.env.agents[0]
            obs = self._add_agent_id(obs_dict[first_live], self.env.possible_agents.index(first_live))
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return obs, reward, done, False, infos

    def set_model(self, model):
        self._model = model

class CentralizedObsCallback(BaseCallback):

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

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):

        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        # One env_method call per rollout to seed the centralized obs cache.
        # All subsequent updates come from infos returned by env.step().
        c_obs_init = env.env_method('get_centralized_obs')
        self._last_c_obs = np.stack(c_obs_init).astype(np.float32)

        while n_steps < n_rollout_steps:
            # Snapshot pre-step joint obs — stored in buffer and used for V(s_t)
            current_c_obs = self._last_c_obs

            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                c_obs_tensor = torch.as_tensor(current_c_obs, device=self.device)
                actions, _, log_probs = self.policy(obs_tensor)
                values = self.policy.centralized_critic(c_obs_tensor).flatten()

            actions_np = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions_np)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions_np = actions_np.reshape(-1, 1)

            rollout_buffer.add(
                self._last_obs,
                actions_np,
                rewards,
                self._last_episode_starts,
                values.cpu(),
                log_probs.cpu(),
                centralized_obs=current_c_obs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

            self._last_c_obs = np.stack([
                info.get('centralized_obs', np.zeros_like(self._last_c_obs[i]))
                for i, info in enumerate(infos)
            ]).astype(np.float32)


        with torch.no_grad():
            c_obs_tensor = torch.as_tensor(self._last_c_obs, device=self.device)
            last_values = self.policy.centralized_critic(c_obs_tensor).flatten()

        rollout_buffer.compute_returns_and_advantage(last_values=last_values.cpu(), dones=dones)
        callback.on_rollout_end()
        return True

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


def make_env(map_choice, maxCycles, num_drones, vision_range=3, agent_idx=0):
    def _init():
        # Limit PyTorch to 1 thread per worker — inference only, no backprop.
        # Prevents each of the 8 subprocesses from spinning up a full thread pool
        # and fighting the main process during the PPO update phase.
        import torch as _torch
        _torch.set_num_threads(1)
        _torch.set_num_interop_threads(1)
        base_env = GridWorldEnvironment(
            mapPreset=map_choice,
            maxCycles=maxCycles,
            visionRange=vision_range,
            use_map_memory=True,
            num_drones=num_drones
        )
        return MultiAgentWrapper(base_env, agent_idx=agent_idx)
    return _init

class SubprocSyncCallback(BaseCallback):
    """After each PPO update, ships fresh policy weights + obs stats to all worker subprocesses.
    This keeps co-agent predictions at most 1 rollout stale, similar to IPPO frozen-agent approach."""

    def __init__(self, subproc_env):
        super().__init__()
        self._subproc_env = subproc_env  # the raw SubprocVecEnv 
        self._first = True

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> bool:
        # Fires AFTER the previous PPO update has completed — ideal sync point.
        if self._first:
            self._first = False
            return True
        sd = {k: v.cpu() for k, v in self.model.policy.state_dict().items()}
        self._subproc_env.env_method('update_policy_weights', sd)
        # Keep obs-normalisation stats current 
        vn = self.training_env
        self._subproc_env.env_method(
            'set_obs_stats',
            vn.obs_rms.mean.copy(), vn.obs_rms.var.copy(),
            float(vn.epsilon), float(vn.clip_obs)
        )
        return True


def trainAgents(total_timesteps, num_drones=4, cumulativeTimestepsSoFar=0, total_training_timesteps=50_000_000, force_map=None, vision_range=3):
    """Train shared MAPPO policy"""
    # Thread budget 
    # Reserve ~5 physical cores (10 threads) for system/gaming. Each worker
    # subprocess is capped at 1 thread (set inside make_env._init).
    torch.set_num_threads(14)

    # Process priority 
    # Yield CPU to games/browser when they need it; no-op when PC is idle.
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass  # psutil optional; graceful fallback

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
    env = SubprocVecEnv(
        # Each env is assigned to one specific agent (agent_idx = env index mod num_drones).
        # This ensures correct per-agent temporal sequences for GAE — no cross-agent mixing.
        [make_env(map_choice, maxCycles, num_drones, vision_range, agent_idx=i % num_drones) for i in range(n_envs)],
        start_method='spawn'  # explicit — default on Windows, required for CUDA-free envs
    )
    
    model_path = os.path.join(mappo_dir, "shared_mappo_model.zip")
    vecnorm_path = os.path.join(mappo_dir, "vecnormalize.pkl")
    
    # Keep VecNormalize updating throughout training + normalize obs only
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = True  # FIXED: Keep training throughout to adapt to distribution shifts
        log.i("Loaded VecNormalize (training=True - adapts throughout training)")
    else:
        #norm_reward=False to preserve carefully tuned reward weights
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        log.i("Created VecNormalize (obs only - preserves reward shaping)")
    
    # Decaying LR schedule: SB3 evaluates this callable each optimizer step, passing
    # progress_remaining (1.0 → 0.0), giving 3e-4 → 3e-5 decay within each session.
    current_lr = lambda p: 3e-4 * max(0.1, p)
    
    # Entropy decay callback to gradually reduce exploration
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
    
#Diagnostics callback to monitor training
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
            if model.observation_space.shape != env.observation_space.shape:
                raise ValueError(f"Obs space mismatch: model={model.observation_space.shape} vs env={env.observation_space.shape}; creating fresh model")
        except Exception as load_err:
            log.i(f"Could not load as MAPPOPPO ({load_err}); creating fresh model")
            model = MAPPOPPO(
                MAPPOPolicy, env, num_agents=num_drones,
                verbose=1, learning_rate=current_lr, n_steps=2048,
                batch_size=512, n_epochs=10, ent_coef=0.1,
                clip_range=0.2, gamma=0.99, gae_lambda=0.95,
                max_grad_norm=1.0, vf_coef=0.5, target_kl=0.02,
                policy_kwargs=policy_kwargs,
                tensorboard_log=os.path.join(mappo_dir, "mappo_tensorboard"),
                device=device
            )
        model.lr_schedule = current_lr
        model.learning_rate = current_lr
        model.n_epochs = 10
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
            n_epochs=10,
            ent_coef=0.1,
            clip_range=0.2,
            gamma=0.99,
            gae_lambda=0.95,
            max_grad_norm=1.0,
            vf_coef=0.5,
            target_kl=0.02,
            policy_kwargs=policy_kwargs,
            tensorboard_log=os.path.join(mappo_dir, "mappo_tensorboard"),
            device=device
        )

    env.env_method('set_policy', model.policy)
    env.env_method(
        'set_obs_stats',
        env.obs_rms.mean.copy(), env.obs_rms.var.copy(),
        float(env.epsilon), float(env.clip_obs)
    )

    log.i(f"Training with {num_drones} agents, {n_envs} SubprocVecEnv workers (CTDE, agent-per-env)")
    log.i(f"CONFIG: Batch=512 | Epochs=10 | LR=3e-4->3e-5 | Entropy=0.1->0.03 | TargetKL=0.01 | CentralizedCritic=ON")

    subproc_env = env.venv  # unwrap VecNormalize to reach the SubprocVecEnv for sync callback

    model.learn(
        total_timesteps=total_timesteps,
        callback=[
            SubprocSyncCallback(subproc_env),
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
        if progress < 0.01:    
            return map_15x15
        elif progress < 0.10:  
            return map_30x30
        else:                  
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