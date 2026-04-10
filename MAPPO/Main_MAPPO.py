import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LoggerConfig import log
from Environment import GridWorldEnvironment
from train_MAPPO import trainAgents
from simulate_MAPPO import runSimulations
from PlotAnalysis import plotAnalysisData
from Maps import map_15x15, map_30x30, map_45x45
import json

# TRAINING CONFIGURATION
timesteps_per_iteration = 50_000
total_iterations = 200
num_drones = 8
force_specific_map = None  # Set to map_15x15, map_30x30, map_45x45, or None for curriculum

# Get MAPPO directory path
mappo_dir = os.path.dirname(os.path.abspath(__file__))

# Load or initialize progress tracker
progress_file = os.path.join(mappo_dir, "training_progress.json")
if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    start_iteration = progress['iterations_completed']
    total_timesteps_so_far = progress['total_timesteps']
    log.i(f"Resuming from iteration {start_iteration + 1}, total timesteps: {total_timesteps_so_far}")
else:
    start_iteration = 0
    total_timesteps_so_far = 0
    log.i("Starting fresh training")

for iteration in range(start_iteration, total_iterations):
    current_timesteps = total_timesteps_so_far + timesteps_per_iteration
    log.i(f"\n=== Iteration {iteration + 1}/{total_iterations} ===")
    
    # Train shared MAPPO policy
    trainAgents(timesteps_per_iteration, num_drones=num_drones, cumulativeTimestepsSoFar=total_timesteps_so_far, force_map=force_specific_map)
    
    # Simulate and analyze
    runSimulations(simulations=100, timeStepsRan=current_timesteps, num_drones=num_drones, force_map=force_specific_map)
    
    # Plot analysis data
    if force_specific_map is not None:
        if force_specific_map is map_15x15:
            plotAnalysisData(os.path.join(mappo_dir, "map_15x15_analysis_Results.csv"))
        elif force_specific_map is map_30x30:
            plotAnalysisData(os.path.join(mappo_dir, "map_30x30_analysis_Results.csv"))
        else:
            plotAnalysisData(os.path.join(mappo_dir, "map_45x45_analysis_Results.csv"))
    else:
        plotAnalysisData(os.path.join(mappo_dir, "map_15x15_analysis_Results.csv"))
        plotAnalysisData(os.path.join(mappo_dir, "map_30x30_analysis_Results.csv"))
        plotAnalysisData(os.path.join(mappo_dir, "map_45x45_analysis_Results.csv"))
    
    # Save progress
    total_timesteps_so_far = current_timesteps
    with open(progress_file, 'w') as f:
        json.dump({'iterations_completed': iteration + 1, 'total_timesteps': total_timesteps_so_far}, f)
    
    log.i(f"Completed iteration {iteration + 1}")