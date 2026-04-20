import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from single_agent_wrapper import SingleAgentWrapper
from Environment import GridWorldEnvironment
from concurrent.futures import ProcessPoolExecutor
import time
import csv
import json as _json
from Maps import map_15x15, map_30x30, map_45x45
from LoggerConfig import log

# Get IPPO directory path
ippo_dir = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(ippo_dir)

def _write_render_queue(data: dict, root_dir: str) -> None:
    """Atomically write render frame data so the UI can pick it up."""
    target = os.path.join(root_dir, "render_queue.json")
    tmp = target + ".new"
    with open(tmp, "w") as f:
        _json.dump(data, f)
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


def _run_episode(model_paths, map_data, maxCycles, num_drones, vision_range, should_render=False):
    """Run a single simulation episode in an isolated process."""
    models = {agent: PPO.load(path, device="cpu") for agent, path in model_paths.items()}
    env = GridWorldEnvironment(
        mapPreset=map_data, maxCycles=maxCycles, visionRange=vision_range,
        use_map_memory=True, num_drones=num_drones,
    )
    obs, _ = env.reset()

    # --- frame collection setup ---
    if should_render:
        initial_grid = env.grid.tolist()
        grid_size = env.grid_size
        known: set = set()
        initial_discovered = []
        for r in range(grid_size):
            for c in range(grid_size):
                if env.discovered[r, c]:
                    known.add((r, c))
                    initial_discovered.append([r, c])
        frames = [{
            "step": 0,
            "agents": {a: list(env.agent_positions[a]) for a in env.agents},
            "new_discovered": initial_discovered,
        }]
        step_num = 0

    while len(env.agents) > 0:
        actions = {}
        for a in env.agents:
            action, _ = models[a].predict(obs[a], deterministic=True)
            actions[a] = int(action)
        obs, _, _, _, _ = env.step(actions)

        if should_render:
            step_num += 1
            new_disc = []
            for r in range(grid_size):
                for c in range(grid_size):
                    if env.discovered[r, c] and (r, c) not in known:
                        known.add((r, c))
                        new_disc.append([r, c])
            frames.append({
                "step": step_num,
                "agents": {a: list(env.agent_positions[a]) for a in env.agents},
                "new_discovered": new_disc,
            })

    env.close()

    if should_render:
        _write_render_queue({
            "grid_size": grid_size,
            "grid": initial_grid,
            "initial_discovered": initial_discovered,
            "frames": frames,
        }, _ROOT_DIR)

    log.i("Simulation finished")
    return env.getEpisodeAnalysis()

def runSimulations(simulations = 100, timeStepsRan = 0, num_drones=4, vision_range=3):
    # Read simulation render setting from training_config.json
    try:
        _cfg_path = os.path.join(ippo_dir, '..', 'training_config.json')
        with open(_cfg_path) as _f:
            _cfg = _json.load(_f)
        sim_render_every = int(_cfg.get('sim_render_every', 0))
    except Exception:
        sim_render_every = 0

    for map_data in [map_15x15, map_30x30, map_45x45]:
        
        # Determine map name and match maxCycles used during training (train_IPPO.py)
        if map_data is map_15x15:
            map_name = "map_15x15"
            maxCycles = 256
        elif map_data is map_30x30:
            map_name = "map_30x30"
            maxCycles = 512
        else:
            map_name = "map_45x45"
            maxCycles = 1024

        # Temp env to enumerate agent names for model loading
        env = GridWorldEnvironment(mapPreset=map_data, maxCycles=maxCycles, visionRange=vision_range, use_map_memory=True, num_drones=num_drones)
        agents = env.possible_agents

        model_paths = {agent: os.path.join(ippo_dir, f"{agent}_ppo_model") for agent in agents}

        n_workers = min(simulations, 6)
        render_flags = _compute_render_flags(simulations, sim_render_every, n_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_run_episode, model_paths, map_data, maxCycles, num_drones, vision_range, render_flags[i])
                       for i in range(simulations)]
            analysisList = [f.result() for f in futures]

        log.i("All simulations completed")

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
        log.i(f"Average steps taken to find reward (only counting episodes where reward was found): {avg_steps_to_reward:.2f}")

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

        # Write to CSV - append mode
        csv_filename = os.path.join(ippo_dir, f"{map_name}_analysis_Results.csv")
        file_exists = False
        try:
            with open(csv_filename, 'r'):
                file_exists = True
        except FileNotFoundError:
            pass

        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['TimeSteps', 'Reward Found %', 'Avg Steps', 'Avg Tiles', 'Avg Tiles Per Step', 'Avg Steps to Reward', last_col_header])
            writer.writerow([timeStepsRan, f'{reward_found_pct*100:.2f}', f'{avg_steps:.2f}', f'{avg_tiles:.2f}', f'{avg_tiles_per_step:.4f}', f'{avg_steps_to_reward:.2f}', last_col_value])

        log.i(f"Results appended to {csv_filename}")


if __name__  == "__main__":
    simulations = 50
    timeStepsRan = 0
    num_drones = 4  # Set number of drones here
    for i in range(2):
        runSimulations(simulations, timeStepsRan, num_drones)
        timeStepsRan += 500_000