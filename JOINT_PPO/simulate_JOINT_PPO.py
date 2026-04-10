import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from JOINT_PPO.single_agent_wrapper import JointAgentWrapper
from JOINT_PPO.Environment import GridWorldEnvironment
import time
import csv
import numpy as np
from Maps import map_15x15, map_30x30, map_45x45
from LoggerConfig import log
import pickle

# Get JOINT_PPO directory path
joint_ppo_dir = os.path.dirname(os.path.abspath(__file__))

def runSimulations(simulations = 100, timeStepsRan = 0, num_drones=4):
    for map_data in [map_15x15, map_30x30, map_45x45]:
        
        # Determine map name
        if map_data is map_15x15:
            map_name = "map_15x15"
            maxCycles = 1024
        elif map_data is map_30x30:
            map_name = "map_30x30"
            maxCycles = 1024
        else:
            map_name = "map_45x45"
            maxCycles = 1024
        
        # Create environment with map memory enabled
        env = GridWorldEnvironment(mapPreset=map_data, maxCycles=maxCycles, visionRange=2, use_map_memory=True, num_drones=num_drones)
        agents = env.possible_agents  # Use possible_agents for consistent ordering
        
        # Wrap environment for joint training compatibility (use reward averaging for consistency)
        wrapper_env = JointAgentWrapper(env, use_reward_averaging=True)

        # Load trained joint model
        joint_model_path = os.path.join(joint_ppo_dir, f"joint_ppo_model_{num_drones}drones.zip")
        vec_normalize_path = os.path.join(joint_ppo_dir, f"joint_ppo_vec_normalize_{num_drones}drones.pkl")
        
        if os.path.exists(joint_model_path):
            try:
                joint_model = PPO.load(joint_model_path, device="cpu")
                log.i("Loaded joint model for simulation")
                
                # Load normalization stats if they exist
                if os.path.exists(vec_normalize_path):
                    with open(vec_normalize_path, 'rb') as f:
                        vec_norm_stats = pickle.load(f)
                    log.i("Loaded VecNormalize statistics")
                else:
                    log.w("VecNormalize file not found; simulation may have incorrect observations")
                    vec_norm_stats = None
                
                # Get action info
                n_actions_per_agent = env.action_spaces[agents[0]].n
                num_agents = len(agents)
                
            except Exception as e:
                log.e(f"Error loading model: {e}")
                continue
        else:
            log.e("Joint model not found, skipping simulation")
            continue

        #Analysis Variables
        analysisList = []

        for i in range(simulations):
            # Create fresh environment for each simulation
            env = GridWorldEnvironment(mapPreset=map_data, maxCycles=maxCycles, visionRange=2, use_map_memory=True, num_drones=num_drones)
            wrapper_env = JointAgentWrapper(env, use_reward_averaging=True)
            
            joint_obs, _ = wrapper_env.reset()
            done = False

            while not done:
                # Get joint action from model (stochastic sampling for exploration)
                joint_action, _states = joint_model.predict(joint_obs, deterministic=False)

                # Introduce occasional random action for exploration
                if np.random.rand() < 0.01:
                    joint_action = wrapper_env.action_space.sample()

                # Step the wrapped environment (returns single joint observation/reward/done)
                joint_obs, joint_reward, done, truncated, info = wrapper_env.step(joint_action)

                # Optional: render environment if implemented
                if hasattr(env, "render_env"):
                    env.render_env()

            #----------------------------------------------------------------------------
            # Analysis
            #----------------------------------------------------------------------------
            analysis = env.getEpisodeAnalysis()
            analysisList.append(analysis)

            if hasattr(env, "render_env"):
                env.render_env()
                time.sleep(0.1)  # slow down visualization
            
            env.close()

        log.i("All simulations completed")

        reward_found_pct = sum([a['reward_found'] for a in analysisList])/len(analysisList)
        avg_steps = sum([a['steps_taken'] for a in analysisList])/len(analysisList)
        avg_tiles = sum([a['tiles_discovered'] for a in analysisList])/len(analysisList)
        avg_tiles_per_step = sum([a['TilesDiscoveredPerStep'] for a in analysisList])/len(analysisList)
        reward_found_list = [a for a in analysisList if a['reward_found']==1]
        avg_steps_to_reward = sum([a['Steps_to_find_reward_if_found'] for a in reward_found_list])/len(reward_found_list) if reward_found_list else 0
        avg_score = sum([a['analysis_score'] for a in analysisList])/len(analysisList)

        log.i(f"% of rewards discovered: {reward_found_pct:.2%}")
        log.i(f"Average steps taken: {avg_steps:.2f}")
        log.i(f"Average tiles discovered: {avg_tiles:.2f}")
        log.i(f"Average tiles discovered per step taken: {avg_tiles_per_step:.4f}")
        log.i(f"Average steps taken to find reward (only counting episodes where reward was found): {avg_steps_to_reward:.2f}")
        log.i(f"Average analysis score: {avg_score:.4f}")

        # Write to CSV - append mode
        csv_filename = os.path.join(joint_ppo_dir, f"{map_name}_analysis_Results.csv")
        file_exists = False
        try:
            with open(csv_filename, 'r'):
                file_exists = True
        except FileNotFoundError:
            pass
        
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['TimeSteps', 'Reward Found %', 'Avg Steps', 'Avg Tiles', 'Avg Tiles Per Step', 'Avg Steps to Reward', 'Avg Score'])
            writer.writerow([timeStepsRan, f'{reward_found_pct*100:.2f}', f'{avg_steps:.2f}', f'{avg_tiles:.2f}', f'{avg_tiles_per_step:.4f}', f'{avg_steps_to_reward:.2f}', f'{avg_score:.4f}'])

        log.i(f"Results appended to {csv_filename}")


if __name__  == "__main__":
    simulations = 50
    timeStepsRan = 0
    num_drones = 4  # Set number of drones here
    for i in range(2):
        runSimulations(simulations, timeStepsRan, num_drones)
        timeStepsRan += 500_000