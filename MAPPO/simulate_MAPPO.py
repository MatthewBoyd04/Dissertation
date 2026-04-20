import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from train_MAPPO import MAPPOPPO, MAPPOPolicy
from Environment import GridWorldEnvironment
from concurrent.futures import ProcessPoolExecutor

import csv
import json as _json
import pickle
from Maps import map_15x15, map_30x30, map_45x45
from LoggerConfig import log
import numpy as np

mappo_dir = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(mappo_dir)

def _write_render_queue(data: dict, root_dir: str) -> None:
    """Atomically write render frame data so the UI can pick it up."""
    import json as _j
    target = os.path.join(root_dir, "render_queue.json")
    tmp = target + ".new"
    with open(tmp, "w") as f:
        _j.dump(data, f)
    os.replace(tmp, target)


def _compute_render_flags(simulations, sim_render_every, n_workers):
    """Return a bool list of length simulations.
    At most 1 True per n_workers gap so concurrent processes never render simultaneously."""
    if not sim_render_every:
        return [False] * simulations
    flags = []
    last_render = -n_workers  # makes episode 0 eligible
    for i in range(simulations):
        if i % sim_render_every == 0 and (i - last_render) >= n_workers:
            flags.append(True)
            last_render = i
        else:
            flags.append(False)
    return flags


def _run_episode(model_path, vecnorm_path, map_data, maxCycles, num_drones, vision_range, should_render=False):
    """Run a single simulation episode in an isolated process."""
    try:
        model = MAPPOPPO.load(model_path, device="cpu", num_agents=num_drones)
        if not isinstance(model.policy, MAPPOPolicy):
            raise ValueError("Old policy format")
    except Exception:
        model = PPO.load(model_path, device="cpu")

    obs_rms = None
    clip_obs = 10.0
    eps = 1e-8
    if os.path.exists(vecnorm_path):
        with open(vecnorm_path, 'rb') as f:
            vecnorm_data = pickle.load(f)
        obs_rms = vecnorm_data.obs_rms
        clip_obs = float(getattr(vecnorm_data, 'clip_obs', 10.0))
        eps = float(getattr(vecnorm_data, 'epsilon', 1e-8))

    base_env = GridWorldEnvironment(
        mapPreset=map_data, maxCycles=maxCycles, visionRange=vision_range,
        use_map_memory=True, num_drones=num_drones,
    )
    obs_dict, _ = base_env.reset()

    # --- frame collection setup ---
    if should_render:
        initial_grid = base_env.grid.tolist()
        grid_size = base_env.grid_size
        known: set = set()
        initial_discovered = []
        for r in range(grid_size):
            for c in range(grid_size):
                if base_env.discovered[r, c]:
                    known.add((r, c))
                    initial_discovered.append([r, c])
        frames = [{
            "step": 0,
            "agents": {a: list(base_env.agent_positions[a]) for a in base_env.agents},
            "new_discovered": initial_discovered,
        }]
        step_num = 0

    num_possible_agents = len(base_env.possible_agents)
    step_counter = 0
    while base_env.agents:
        actions = {}
        for agent in base_env.agents:
            raw_obs = obs_dict[agent]
            # Append agent identity channel to match the expanded training observation space
            agent_idx = base_env.possible_agents.index(agent)
            id_val = agent_idx / max(num_possible_agents - 1, 1)
            H, W = raw_obs.shape[1], raw_obs.shape[2]
            id_channel = np.full((1, H, W), id_val, dtype=np.float32)
            obs = np.concatenate([raw_obs, id_channel], axis=0)
            if obs_rms is not None:
                obs = (obs - obs_rms.mean) / np.sqrt(obs_rms.var + eps)
                obs = np.clip(obs, -clip_obs, clip_obs)
            action, _ = model.predict(obs, deterministic=True)
            actions[agent] = int(action)
        obs_dict, _, _, _, _ = base_env.step(actions)
        step_counter += 1

        if should_render:
            step_num += 1
            new_disc = []
            for r in range(grid_size):
                for c in range(grid_size):
                    if base_env.discovered[r, c] and (r, c) not in known:
                        known.add((r, c))
                        new_disc.append([r, c])
            frames.append({
                "step": step_num,
                "agents": {a: list(base_env.agent_positions[a]) for a in base_env.agents},
                "new_discovered": new_disc,
            })

    if should_render:
        _write_render_queue({
            "grid_size": grid_size,
            "grid": initial_grid,
            "initial_discovered": initial_discovered,
            "frames": frames,
        }, _ROOT_DIR)

    log.i("Simulation finished")
    return {
        'reward_found': base_env.rewards_collected / base_env.num_rewards,
        'steps_taken': step_counter,
        'tiles_discovered': base_env.getNumTilesDiscovered(),
        'analysis_score': 50 * int(base_env.reward_found) - step_counter + 0.1 * base_env.getNumTilesDiscovered(),
        'Steps_to_find_reward_if_found': base_env.reward_all_found_step,
        'TilesDiscoveredPerStep': base_env.getNumTilesDiscovered() / step_counter if step_counter > 0 else 0,
        'hazard_terminations': base_env.hazard_terminations,
        'has_hazards': base_env.has_hazards,
    }

def runSimulations(simulations = 100, timeStepsRan = 0, num_drones=4, force_map=None, vision_range=3):
    model_path = os.path.join(mappo_dir, "shared_mappo_model")
    vecnorm_path = os.path.join(mappo_dir, "vecnormalize.pkl")

    # Read simulation render setting from training_config.json
    try:
        _cfg_path = os.path.join(mappo_dir, '..', 'training_config.json')
        with open(_cfg_path) as _f:
            _cfg = _json.load(_f)
        sim_render_every = int(_cfg.get('sim_render_every', 0))
    except Exception:
        sim_render_every = 0
    
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
            maxCycles = 512
        else:
            map_name = "map_45x45"
            maxCycles = 1024
        
        n_workers = min(simulations, 6)
        render_flags = _compute_render_flags(simulations, sim_render_every, n_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_run_episode, model_path, vecnorm_path, map_data, maxCycles, num_drones, vision_range, render_flags[i])
                       for i in range(simulations)]
            analysisList = [f.result() for f in futures]
        
        reward_found_pct = sum([a['reward_found'] for a in analysisList])/len(analysisList)
        avg_steps = sum([a['steps_taken'] for a in analysisList])/len(analysisList)
        avg_tiles = sum([a['tiles_discovered'] for a in analysisList])/len(analysisList)
        avg_tiles_per_step = sum([a['TilesDiscoveredPerStep'] for a in analysisList])/len(analysisList)
        reward_found_list = [a for a in analysisList if a['reward_found'] >= 1.0]
        steps_list = [a['Steps_to_find_reward_if_found'] for a in reward_found_list
                      if a['Steps_to_find_reward_if_found'] is not None]
        avg_steps_to_reward = sum(steps_list) / len(steps_list) if steps_list else 0
        has_hazards = analysisList[0]['has_hazards']

        log.i(f"% of rewards discovered: {reward_found_pct:.2%}")
        log.i(f"Average steps taken: {avg_steps:.2f}")
        log.i(f"Average tiles discovered: {avg_tiles:.2f}")
        log.i(f"Average tiles discovered per step taken: {avg_tiles_per_step:.4f}")
        log.i(f"Average steps taken to find reward: {avg_steps_to_reward:.2f}")

        if has_hazards:
            avg_terminated = sum([a['hazard_terminations'] for a in analysisList])/len(analysisList)
            last_col_header = 'Avg Drones Terminated'
            last_col_value = f'{avg_terminated:.4f}'
            log.i(f"Average drones terminated by hazard: {avg_terminated:.4f}")
        else:
            avg_score = sum([a['analysis_score'] for a in analysisList])/len(analysisList)
            last_col_header = 'Avg Score'
            last_col_value = f'{avg_score:.4f}'
            log.i(f"Average analysis score: {avg_score:.4f}")

        csv_filename = os.path.join(mappo_dir, f"{map_name}_analysis_Results.csv")
        file_exists = os.path.exists(csv_filename)

        # Read existing data and remove duplicate timesteps to prevent plotting issues
        existing_rows = []
        if file_exists:
            with open(csv_filename, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if row and row[0] != str(timeStepsRan):  # Keep all rows except those matching current timestep
                        existing_rows.append(row)
        else:
            header = ['TimeSteps', 'Reward Found %', 'Avg Steps', 'Avg Tiles', 'Avg Tiles Per Step', 'Avg Steps to Reward', last_col_header]

        # Write file with deduplicated data
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in existing_rows:
                writer.writerow(row)
            # Write the new data
            writer.writerow([timeStepsRan, f'{reward_found_pct*100:.2f}', f'{avg_steps:.2f}', f'{avg_tiles:.2f}', f'{avg_tiles_per_step:.4f}', f'{avg_steps_to_reward:.2f}', last_col_value])
        
        log.i(f"Results saved to {csv_filename} (duplicates removed)")

if __name__  == "__main__":
    runSimulations(50, 0, 4)