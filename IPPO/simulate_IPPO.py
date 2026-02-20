import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from single_agent_wrapper import SingleAgentWrapper
from Environment import GridWorldEnvironment
import time
import csv
from Maps import map_15x15, map_30x30, map_45x45
from LoggerConfig import log

 # Agent names
agents = ["Drone_1", "Drone_2", "Drone_3", "Drone_4"]

# Map for evaluation
map_data = map_30x30

def runSimulations(simulations = 100, timeStepsRan = 0):
    for map_data in [map_15x15, map_30x30, map_45x45]:
        
        # Determine map name
        if map_data is map_15x15:
            map_name = "map_15x15"
        elif map_data is map_30x30:
            map_name = "map_30x30"
        else:
            map_name = "map_45x45"
        
        # Create environment
        env = GridWorldEnvironment(mapPreset=map_data, agents=agents, maxCycles=512, visionRange=2)

        # Load trained models
        models = {}
        for agent in agents:
            models[agent] = PPO.load(f"{agent}_ppo_model")

        #Anaysis Variables
        analysisList = []


        for i in range (simulations):

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

            log.i("Simulation finished")
            env.close()

            #----------------------------------------------------------------------------
            # Analysis
            #----------------------------------------------------------------------------
            analysis = env.getEpisodeAnalysis()
            analysisList.append(analysis)

                
            if hasattr(env, "render_env"):
                env.render_env()
                time.sleep(0.1)  # slow down visualization

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
        csv_filename = f"{map_name}_analysis_Results.csv"
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
    for i in range(2):
        runSimulations(simulations, timeStepsRan)
        timeStepsRan += 500_000