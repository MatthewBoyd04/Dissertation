import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LoggerConfig import log
from Environment import GridWorldEnvironment
from IPPO.train_IPPO import trainAgents
from IPPO.simulate_IPPO import runSimulations
import json

timesteps_per_iteration = 100_000
total_iterations = 20

# Load or initialize progress tracker
progress_file = "training_progress.json"
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
    trainAgents(timesteps_per_iteration)
    
    # Simulate and analyze
    runSimulations(simulations=100, timeStepsRan=current_timesteps)
    
    # Save progress
    total_timesteps_so_far = current_timesteps
    with open(progress_file, 'w') as f:
        json.dump({'iterations_completed': iteration + 1, 'total_timesteps': total_timesteps_so_far}, f)
    
    log.i(f"Completed iteration {iteration + 1}")