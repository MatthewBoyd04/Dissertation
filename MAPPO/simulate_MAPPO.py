import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Environment import GridWorldEnvironment

import csv
from Maps import map_15x15, map_30x30, map_45x45
from LoggerConfig import log
import numpy as np

mappo_dir = os.path.dirname(os.path.abspath(__file__))

def runSimulations(simulations = 100, timeStepsRan = 0, num_drones=4, force_map=None):
    model = PPO.load(os.path.join(mappo_dir, "shared_mappo_model"))
    vecnorm_path = os.path.join(mappo_dir, "vecnormalize.pkl")
    
    # Load VecNormalize stats if available
    vecnorm = None
    if os.path.exists(vecnorm_path):
        import pickle
        with open(vecnorm_path, 'rb') as f:
            vecnorm_data = pickle.load(f)
        log.i("Loaded VecNormalize stats for simulation")
    else:
        vecnorm_data = None
    
    # Select maps to simulate
    if force_map is not None:
        map_list = [force_map]
    else:
        map_list = [map_15x15, map_30x30, map_45x45]
    
    for map_data in map_list:
        if map_data is map_15x15:
            map_name = "map_15x15"
            maxCycles = 256
        elif map_data is map_30x30:
            map_name = "map_30x30"
            maxCycles = 1024
        else:
            map_name = "map_45x45"
            maxCycles = 2048
        
        analysisList = []
        
        for i in range(simulations):
            base_env = GridWorldEnvironment(mapPreset=map_data, maxCycles=maxCycles, visionRange=2, use_map_memory=True, num_drones=num_drones)
            
            obs_dict, _ = base_env.reset()
            step_counter = 0
            
            while base_env.agents:
                actions = {}
                for agent in base_env.agents:
                    obs = obs_dict[agent]
                    # Apply normalization if available
                    if vecnorm_data is not None:
                        obs = (obs - vecnorm_data.obs_rms.mean) / np.sqrt(vecnorm_data.obs_rms.var + 1e-8)
                        obs = np.clip(obs, -10.0, 10.0)
                    action, _ = model.predict(obs, deterministic=False)
                    actions[agent] = int(action)
                
                obs_dict, rewards, terminations, truncations, infos = base_env.step(actions)
                step_counter += 1
            
            reward_was_found = base_env.reward_found
            
            analysis = {
                'reward_found': 1 if reward_was_found else 0,
                'steps_taken': step_counter,
                'tiles_discovered': base_env.getNumTilesDiscovered(),
                'analysis_score': 50 * int(reward_was_found) - step_counter + 0.1 * base_env.getNumTilesDiscovered(),
                'Steps_to_find_reward_if_found': step_counter if reward_was_found else None,
                'TilesDiscoveredPerStep': base_env.getNumTilesDiscovered() / step_counter if step_counter > 0 else 0
            }
            
            analysisList.append(analysis)
            
            if (i+1) % 2 == 0:
                log.i(f"Simulation {i+1}/{simulations} finished")
        
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
        log.i(f"Average steps taken to find reward: {avg_steps_to_reward:.2f}")
        log.i(f"Average analysis score: {avg_score:.4f}")
        
        csv_filename = os.path.join(mappo_dir, f"{map_name}_analysis_Results.csv")
        file_exists = os.path.exists(csv_filename)
        
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['TimeSteps', 'Reward Found %', 'Avg Steps', 'Avg Tiles', 'Avg Tiles Per Step', 'Avg Steps to Reward', 'Avg Score'])
            writer.writerow([timeStepsRan, f'{reward_found_pct*100:.2f}', f'{avg_steps:.2f}', f'{avg_tiles:.2f}', f'{avg_tiles_per_step:.4f}', f'{avg_steps_to_reward:.2f}', f'{avg_score:.4f}'])
        
        log.i(f"Results appended to {csv_filename}")

if __name__  == "__main__":
    runSimulations(50, 0, 4)