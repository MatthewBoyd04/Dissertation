import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from single_agent_wrapper import SingleAgentWrapper
from Environment import GridWorldEnvironment
import torch
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

agents = ["Drone_1", "Drone_2", "Drone_3", "Drone_4"]

def trainAgents(total_timesteps):
    env = GridWorldEnvironment(
        mapPreset=map_30x30, 
        agents=agents, 
        maxCycles=100, 
        visionRange=2,
        render_every=None
    )

    models = {}

    for agent in agents:
        agent_env = SingleAgentWrapper(env, agent)
        model_path = f"{agent}_ppo_model.zip"
        
        # Load existing model or create new one
        if os.path.exists(model_path):
            log.i(f"Loading existing model for {agent}")
            models[agent] = PPO.load(model_path, env=agent_env, device=device)
        else:
            log.i(f"Creating new model for {agent}")
            models[agent] = PPO(
                "MlpPolicy",
                agent_env,
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                ent_coef=0.01,
                tensorboard_log=f"./ppo_{agent}_tensorboard/",
                device=device
            )

        log.i(f"Training agent: {agent}")
        models[agent].learn(total_timesteps=total_timesteps)
        models[agent].save(f"{agent}_ppo_model")


