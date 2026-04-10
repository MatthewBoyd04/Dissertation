#!/usr/bin/env python3
"""
Joint PPO Training Data Cleanup Script

This script safely removes all Joint PPO training data including:
- Training progress file (training_progress.json)
- Trained model files (joint_ppo_model_*.zip)
- Vector normalization files (joint_ppo_vec_normalize_*.pkl)
- TensorBoard logs (joint_ppo_tensorboard/)
- Analysis results CSV files (map_*_analysis_Results.csv)

Usage: python clear_joint_ppo_data.py
"""

import os
import shutil
import glob
import sys

def get_training_data_files():
    """Get all Joint PPO training data files to be deleted"""
    files_to_delete = []

    # Training progress
    if os.path.exists('training_progress.json'):
        files_to_delete.append('training_progress.json')

    # Model files
    model_files = glob.glob('joint_ppo_model_*.zip')
    files_to_delete.extend(model_files)

    # Vector normalization files
    vec_norm_files = glob.glob('joint_ppo_vec_normalize_*.pkl')
    files_to_delete.extend(vec_norm_files)

    # TensorBoard directory
    if os.path.exists('joint_ppo_tensorboard'):
        files_to_delete.append('joint_ppo_tensorboard/')

    # Analysis results
    analysis_files = glob.glob('map_*_analysis_Results.csv')
    files_to_delete.extend(analysis_files)

    return files_to_delete

def confirm_deletion(files_to_delete):
    """Ask user to confirm deletion"""
    print("\nJOINT PPO TRAINING DATA CLEANUP")
    print("=" * 50)
    print("\nThe following files/directories will be PERMANENTLY DELETED:")
    print()

    for file in files_to_delete:
        if file.endswith('/'):
            print(f"  [DIR] {file} (directory)")
        else:
            print(f"  [FILE] {file}")

    print()
    print("WARNING: This action CANNOT be undone!")
    print()

    while True:
        response = input("Are you sure you want to delete all Joint PPO training data? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Please enter 'yes' or 'no'")

def delete_files(files_to_delete):
    """Delete the specified files and directories"""
    deleted_count = 0
    errors = []

    for file_path in files_to_delete:
        try:
            if file_path.endswith('/'):
                # Remove directory
                dir_path = file_path.rstrip('/')
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    print(f"Deleted directory: {dir_path}")
                    deleted_count += 1
            else:
                # Remove file
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                    deleted_count += 1
        except Exception as e:
            error_msg = f"Failed to delete {file_path}: {str(e)}"
            print(error_msg)
            errors.append(error_msg)

    return deleted_count, errors

def main():
    """Main cleanup function"""
    print("Scanning for Joint PPO training data...")

    files_to_delete = get_training_data_files()

    if not files_to_delete:
        print("\nNo Joint PPO training data found to delete.")
        return

    # Confirm deletion
    if not confirm_deletion(files_to_delete):
        print("\nCleanup cancelled by user.")
        return

    # Perform deletion
    print("\nDeleting files...")
    print("-" * 30)

    deleted_count, errors = delete_files(files_to_delete)

    print("-" * 30)
    print(f"\nSuccessfully deleted {deleted_count} items.")

    if errors:
        print(f"\n{len(errors)} errors occurred:")
        for error in errors:
            print(f"   {error}")
    else:
        print("\nAll Joint PPO training data has been cleared!")
        print("\nYou can now start fresh training with:")
        print("   python train_JOINT_PPO.py")

if __name__ == "__main__":
    main()