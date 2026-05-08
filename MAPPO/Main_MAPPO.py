import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LoggerConfig import log
from Environment import GridWorldEnvironment
from train_MAPPO import trainAgents
from simulate_MAPPO import runSimulations

import json
import time

if __name__ == '__main__':
    # TRAINING CONFIGURATION
    timesteps_per_iteration = 50_000
    total_iterations = 202
    num_drones = 8
    vision_range = 3
    force_specific_map = None  # Set to map_15x15, map_30x30, map_45x45, or None for curriculum

    # Override defaults with UI config if present
    _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training_config.json")
    _cfg_path = os.path.normpath(_cfg_path)
    if os.path.exists(_cfg_path):
        with open(_cfg_path) as _f:
            _cfg = json.load(_f)
        num_drones = _cfg.get("num_drones", num_drones)
        vision_range = _cfg.get("vision_range", vision_range)

    # Get MAPPO directory path
    mappo_dir = os.path.dirname(os.path.abspath(__file__))

    # Load or initialize progress tracker
    progress_file = os.path.join(mappo_dir, "training_progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        start_iteration = progress['iterations_completed']
        total_timesteps_so_far = progress['total_timesteps']
        elapsed_seconds_base = progress.get('elapsed_seconds', 0)
        log.i(f"Resuming from iteration {start_iteration + 1}, total timesteps: {total_timesteps_so_far}")
    else:
        start_iteration = 0
        total_timesteps_so_far = 0
        elapsed_seconds_base = 0
        log.i("Starting fresh training")

    session_start_time = time.time()

    for iteration in range(start_iteration, total_iterations):
        current_timesteps = total_timesteps_so_far + timesteps_per_iteration
        log.i(f"\n=== Iteration {iteration + 1}/{total_iterations} ===")

        # Train shared MAPPO policy
        trainAgents(timesteps_per_iteration, num_drones=num_drones, cumulativeTimestepsSoFar=total_timesteps_so_far, total_training_timesteps=timesteps_per_iteration * total_iterations, force_map=force_specific_map, vision_range=vision_range)

        # Simulate and analyze
        runSimulations(simulations=100, timeStepsRan=current_timesteps, num_drones=num_drones, force_map=force_specific_map, vision_range=vision_range)

        # Save progress
        total_timesteps_so_far = current_timesteps
        elapsed_seconds_total = elapsed_seconds_base + (time.time() - session_start_time)
        with open(progress_file, 'w') as f:
            json.dump({'iterations_completed': iteration + 1, 'total_timesteps': total_timesteps_so_far, 'elapsed_seconds': elapsed_seconds_total}, f)

        log.i(f"Completed iteration {iteration + 1}")