import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from single_agent_wrapper import SingleAgentWrapper
from Environment import GridWorldEnvironment
import torch
from concurrent.futures import ThreadPoolExecutor
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

# Get IPPO directory path
ippo_dir = os.path.dirname(os.path.abspath(__file__))

def trainSingleAgent(agent, total_timesteps):
    """Train a single agent - used for parallel training"""
    env = GridWorldEnvironment(
        mapPreset=map_30x30, 
        agents=agents, 
        maxCycles=100, 
        visionRange=2,
        render_every=None
    )
    
    agent_env = SingleAgentWrapper(env, agent)
    model_path = os.path.join(ippo_dir, f"{agent}_ppo_model.zip")
    
    # Load existing model or create new one
    if os.path.exists(model_path):
        log.i(f"Loading existing model for {agent}")
        model = PPO.load(model_path, env=agent_env, device="cpu")  # Use CPU for parallel training
    else:
        log.i(f"Creating new model for {agent}")
        model = PPO(
            "MlpPolicy",
            agent_env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            ent_coef=0.01,
            tensorboard_log=os.path.join(ippo_dir, f"ppo_{agent}_tensorboard"),
            device="cpu"
        )
    
    log.i(f"Training agent: {agent}")
    model.learn(total_timesteps=total_timesteps)
    model.save(os.path.join(ippo_dir, f"{agent}_ppo_model"))
    log.i(f"Finished training {agent}")

def trainAgents(total_timesteps, parallel=True):
    if parallel:
        """Train all agents in parallel using threads"""
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(trainSingleAgent, agent, total_timesteps) for agent in agents]
            # Wait for all to complete
            for future in futures:
                future.result()
    else:
        for agent in agents:
            trainSingleAgent(agent, total_timesteps)

        


