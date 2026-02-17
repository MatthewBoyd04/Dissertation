from stable_baselines3 import PPO
from single_agent_wrapper import SingleAgentWrapper
from Environment import GridWorldEnvironment
import numpy as np
import time
from Maps import map_15x15, map_30x30, map_45x45

# Agent names
agents = ["Drone_1", "Drone_2", "Drone_3", "Drone_4"]

# Map for evaluation
map_data = map_15x15

# Create environment (make sure render_every=None or controlled)
env = GridWorldEnvironment(mapPreset=map_data, agents=agents, maxCycles=100, visionRange=2, render_every=1)

# Load trained models
models = {}
for agent in agents:
    models[agent] = PPO.load(f"{agent}_ppo_model")

obs, _ = env.reset()
done = {agent: False for agent in agents}

while len(env.agents) > 0:
    actions = {}
    
    # Get each agent's action from its trained model
    for a in env.agents:
        single_obs = obs[a]
        action, _states = models[a].predict(single_obs, deterministic=True)
        actions[a] = int(action)

    # Step the environment
    obs, rewards, terminations, truncations, infos = env.step(actions)

    # Optional: render environment if implemented
    if hasattr(env, "render_env"):
        env.render_env()

print("Simulation finished")
env.close()

if hasattr(env, "render_env"):
    env.render_env()
    time.sleep(0.1)  # slow down visualization