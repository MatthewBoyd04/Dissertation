"""
Script to reset training by deleting old models and statistics.
Run this before starting fresh training with the fixed MAPPO implementation.
"""
import os
import shutil

mappo_dir = os.path.dirname(os.path.abspath(__file__))

files_to_delete = [
    "shared_mappo_model.zip",
    "vecnormalize.pkl",
    "training_progress.json",
    "map_15x15_analysis_Results.csv",
    "map_30x30_analysis_Results.csv",
    "map_45x45_analysis_Results.csv"
]

print("Resetting MAPPO training...")
print("-" * 50)

for filename in files_to_delete:
    filepath = os.path.join(mappo_dir, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"✓ Deleted: {filename}")
    else:
        print(f"  Skipped: {filename} (not found)")

# Optionally archive old tensorboard logs
tensorboard_dir = os.path.join(mappo_dir, "mappo_tensorboard")
if os.path.exists(tensorboard_dir):
    archive_dir = os.path.join(mappo_dir, "_droneDataArchive", "old_tensorboard_logs")
    os.makedirs(archive_dir, exist_ok=True)
    
    # Count existing archives
    existing = len([d for d in os.listdir(archive_dir) if d.startswith("run_")])
    new_archive = os.path.join(archive_dir, f"run_{existing + 1}")
    
    shutil.move(tensorboard_dir, new_archive)
    print(f"✓ Archived tensorboard logs to: {new_archive}")

print("-" * 50)
print("Reset complete! Ready for fresh training.")
print("\nTo start training, run: python Main_MAPPO.py")
