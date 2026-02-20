import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LoggerConfig import log
from Environment import GridWorldEnvironment
from IPPO.train_IPPO import trainAgents
from IPPO.simulate_IPPO import runSimulations
from PlotAnalysis import plotAnalysisData
import json

timesteps_per_iteration = 50_000
total_iterations = 1000

# Get IPPO directory path
ippo_dir = os.path.dirname(os.path.abspath(__file__))

# Load or initialize progress tracker
progress_file = os.path.join(ippo_dir, "training_progress.json")
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
    
    # Train
    trainAgents(timesteps_per_iteration, parallel=False)
    
    # Simulate and analyze
    runSimulations(simulations=100, timeStepsRan=current_timesteps)
    
    # Plot analysis data for all maps
    plotAnalysisData(os.path.join(ippo_dir, "map_15x15_analysis_Results.csv"))
    plotAnalysisData(os.path.join(ippo_dir, "map_30x30_analysis_Results.csv"))
    plotAnalysisData(os.path.join(ippo_dir, "map_45x45_analysis_Results.csv"))
    
    # Save progress
    total_timesteps_so_far = current_timesteps
    with open(progress_file, 'w') as f:
        json.dump({'iterations_completed': iteration + 1, 'total_timesteps': total_timesteps_so_far}, f)
    
    log.i(f"Completed iteration {iteration + 1}")