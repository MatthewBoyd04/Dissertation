import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from single_agent_wrapper import SingleAgentWrapper
from Environment import GridWorldEnvironment
import torch
import random
from concurrent.futures import ProcessPoolExecutor
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

# Get IPPO directory path
ippo_dir = os.path.dirname(os.path.abspath(__file__))

def trainSingleAgent(agent, total_timesteps, num_drones, cumulativeTimestepsSoFar=0, total_training_timesteps=10_000_000, use_frozen_agents=True, vision_range=3):
    """Train a single agent - used for parallel training"""
    torch.set_num_threads(1)  # prevent intra-op thread contention across worker processes
    # Randomize map for each training session
    map_choice = getMapChoice("cirriculum_Random", cumulativeTimestepsSoFar, total_training_timesteps)  # Change to "random" for random map selection each time
    
    if map_choice is map_15x15:
        maxCycles = 256
    elif map_choice is map_30x30:
        maxCycles = 512
    else:
        maxCycles = 1024
    
    # Temp env to enumerate agent names for frozen model loading
    temp_env = GridWorldEnvironment(
        mapPreset=map_choice,
        maxCycles=maxCycles,
        visionRange=vision_range,
        use_map_memory=True,
        num_drones=num_drones
    )

    # Load frozen models for other agents if enabled
    frozen_models = {}
    if use_frozen_agents:
        for other_agent in temp_env.possible_agents:
            if other_agent != agent:
                other_model_path = os.path.join(ippo_dir, f"{other_agent}_ppo_model.zip")
                if os.path.exists(other_model_path):
                    frozen_models[other_agent] = PPO.load(other_model_path, device="cpu")
                    log.i(f"Loaded frozen model for {other_agent}")
    else:
        log.i(f"Frozen agent behavior disabled for {agent}")

    # 8 parallel envs — each gets its own GridWorldEnvironment instance.
    # frozen_models is shared read-only across all workers (DummyVecEnv is single-threaded).
    n_envs = 8
    def make_env():
        def _init():
            env = GridWorldEnvironment(
                mapPreset=map_choice,
                maxCycles=maxCycles,
                visionRange=vision_range,
                use_map_memory=True,
                num_drones=num_drones
            )
            return SingleAgentWrapper(env, agent, frozen_models=frozen_models)
        return _init

    vec_env = DummyVecEnv([make_env() for _ in range(n_envs)])
    model_path = os.path.join(ippo_dir, f"{agent}_ppo_model.zip")
    
    # Calculate learning rate with linear decay (3e-4 → 3e-5 over full training budget)
    progress = cumulativeTimestepsSoFar / total_training_timesteps
    lr_schedule = lambda progress_remaining: 3e-4 * max(0.1, progress_remaining)

    # Load existing model or create new one
    if os.path.exists(model_path):
        log.i(f"Loading existing model for {agent}")
        model = PPO.load(model_path, env=vec_env, device="cpu")
        model.lr_schedule = lr_schedule
        model.learning_rate = lr_schedule
        log.i(f"Updated LR schedule: decaying 3e-4 -> 3e-5 (progress {progress:.2%})")
    else:
        log.i(f"Creating new model for {agent}")
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=lr_schedule,
            n_steps=2048,
            batch_size=512,
            ent_coef=0.05,
            tensorboard_log=os.path.join(ippo_dir, f"ppo_{agent}_tensorboard"),
            device="cpu"
        )
    
    log.i(f"Training agent: {agent}")
    model.learn(total_timesteps=total_timesteps)
    model.save(os.path.join(ippo_dir, f"{agent}_ppo_model"))
    log.i(f"Finished training {agent}")

def trainAgents(total_timesteps, num_drones=4, parallel=True, cumulativeTimestepsSoFar=0, total_training_timesteps=10_000_000, use_frozen_agents=True, vision_range=3):
    # Create temp env to get agent names
    temp_env = GridWorldEnvironment(mapPreset=map_15x15, num_drones=num_drones)
    agents = temp_env.possible_agents
    
    if parallel:
        """Train all agents in parallel using threads"""
        max_workers = min(num_drones, 8)  # Limit workers
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(trainSingleAgent, agent, total_timesteps, num_drones, cumulativeTimestepsSoFar, total_training_timesteps, use_frozen_agents, vision_range) for agent in agents]
            # Wait for all to complete
            for future in futures:
                future.result()
    else:
        for agent in agents:
            trainSingleAgent(agent, total_timesteps, num_drones, cumulativeTimestepsSoFar, total_training_timesteps, use_frozen_agents, vision_range)

def getMapChoice(selectionMethod, total_timesteps_so_far, total_training_timesteps=10_000_000):
    """Get map choice based on selection method.
    Uses fractional progress so the schedule adapts if total training budget changes."""
    progress = total_timesteps_so_far / max(total_training_timesteps, 1)

    if selectionMethod == "random":
        return random.choice([map_15x15, map_30x30, map_45x45])

    elif selectionMethod == "cirriculum":
        if progress < 0.05:    # First 5%: pure 15x15
            return map_15x15
        elif progress < 0.50:  # 5-50%: pure 30x30
            return map_30x30
        else:                  # 50%+: pure 45x45
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


