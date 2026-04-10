import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from single_agent_wrapper import SingleAgentWrapper
from Environment import GridWorldEnvironment
import torch
import random
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

# Get IPPO directory path
ippo_dir = os.path.dirname(os.path.abspath(__file__))

def trainSingleAgent(agent, total_timesteps, num_drones, cumulativeTimestepsSoFar=0, total_training_timesteps=10_000_000, use_frozen_agents=True):
    """Train a single agent - used for parallel training"""
    # Randomize map for each training session
    map_choice = getMapChoice("cirriculum_Random", cumulativeTimestepsSoFar)  # Change to "random" for random map selection each time
    
    if map_choice is map_15x15:
        maxCycles = 128
    elif map_choice is map_30x30:
        maxCycles = 512
    else:
        maxCycles = 1024
    
    env = GridWorldEnvironment(
        mapPreset=map_choice, 
        maxCycles=maxCycles, 
        visionRange=2,
        render_every=None,
        use_map_memory=True,  # Enable map memory
        num_drones=num_drones
    )
    
    # Load frozen models for other agents if enabled
    frozen_models = {}
    if use_frozen_agents:
        for other_agent in env.possible_agents:
            if other_agent != agent:
                other_model_path = os.path.join(ippo_dir, f"{other_agent}_ppo_model.zip")
                if os.path.exists(other_model_path):
                    frozen_models[other_agent] = PPO.load(other_model_path, device="cpu")
                    log.i(f"Loaded frozen model for {other_agent}")
    else:
        log.i(f"Frozen agent behavior disabled for {agent}")
    
    agent_env = SingleAgentWrapper(env, agent, frozen_models=frozen_models)
    model_path = os.path.join(ippo_dir, f"{agent}_ppo_model.zip")
    
    # Calculate learning rate with decay
    progress = cumulativeTimestepsSoFar / total_training_timesteps
    current_lr = 3e-4
    
    # Load existing model or create new one
    if os.path.exists(model_path):
        log.i(f"Loading existing model for {agent}")
        model = PPO.load(model_path, env=agent_env, device="cpu", learning_rate=current_lr)
        # Ensure the loaded model uses a constant learning rate schedule
        model.learning_rate = current_lr
        model.lr_schedule = lambda progress_remaining: current_lr
        log.i(f"Updated learning rate to constant {current_lr:.6f}")
    else:
        log.i(f"Creating new model for {agent}")
        model = PPO(
            "MlpPolicy",
            agent_env,
            verbose=1,
            learning_rate=current_lr,
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

def trainAgents(total_timesteps, num_drones=4, parallel=True, cumulativeTimestepsSoFar=0, total_training_timesteps=10_000_000, use_frozen_agents=True):
    # Create temp env to get agent names
    temp_env = GridWorldEnvironment(mapPreset=map_15x15, num_drones=num_drones)
    agents = temp_env.possible_agents
    
    if parallel:
        """Train all agents in parallel using threads"""
        max_workers = min(num_drones, 8)  # Limit workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(trainSingleAgent, agent, total_timesteps, num_drones, cumulativeTimestepsSoFar, total_training_timesteps, use_frozen_agents) for agent in agents]
            # Wait for all to complete
            for future in futures:
                future.result()
    else:
        for agent in agents:
            trainSingleAgent(agent, total_timesteps, num_drones, cumulativeTimestepsSoFar, total_training_timesteps, use_frozen_agents)

def getMapChoice(selectionMethod, total_timesteps_so_far):
    """Get map choice based on selection method"""
    if selectionMethod == "random":
        return random.choice([map_15x15, map_30x30, map_45x45])
    
    elif selectionMethod == "cirriculum": #Presuming 10_000_000 timestps total, 1_000_000 for 15x15, 4_000_000 for 30x30, 5_000_000 for 45x45
        if total_timesteps_so_far < 500_000: #Convergence normally occurs before 500_000 timesteps for 15x15, so we can start with that
            return map_15x15
        elif total_timesteps_so_far < 5_000_000:
            return map_30x30
        else:
            return map_45x45
        
    elif selectionMethod == "cirriculum_Random": #Start with cirriculum, but randomize between maps within each stage
        if total_timesteps_so_far < 2_500_000:
            return map_15x15
        elif total_timesteps_so_far < 10_000_000:
            return random.choice([map_15x15, map_15x15, map_15x15, map_30x30, map_30x30]) #60% 15x15, 40% 30x30 for a smoother transition
        else:
            return random.choice([map_15x15, map_30x30, map_45x45 , map_45x45, map_45x45, map_45x45 , map_45x45 , map_45x45 , map_45x45 , map_45x45]) #
    else:
        raise ValueError("Invalid selection method")


