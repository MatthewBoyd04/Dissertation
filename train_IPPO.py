from stable_baselines3 import PPO
from single_agent_wrapper import SingleAgentWrapper
from Environment import GridWorldEnvironment
import numpy as np
import torch
from LoggerConfig import log
from Maps import map_15x15, map_30x30, map_45x45

agents = ["Drone_1", "Drone_2", "Drone_3", "Drone_4"]
map_data = np.array([[0]*15 for _ in range(15)])
env = GridWorldEnvironment(
    mapPreset=map_30x30, 
    agents=agents, 
    maxCycles=100, 
    visionRange=2,
    render_every=500
    )

models = {}
total_timesteps = 500_000

for agent in agents:
    agent_env = SingleAgentWrapper(env, agent)
    
    models[agent] = PPO(
        "MlpPolicy",
        agent_env,
        verbose=1,
        tensorboard_log=f"./ppo_{agent}_tensorboard/",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    log.i(f"Training agent: {agent}")
    models[agent].learn(total_timesteps=total_timesteps)
    models[agent].save(f"{agent}_ppo_model")