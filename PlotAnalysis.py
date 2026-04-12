import pandas as pd
import matplotlib.pyplot as plt
import os

# Store figure references for each CSV file
_plot_figures = {}

def plotAnalysisData(csv_filename, ma_window=10):
    """
    Plots all metrics from a CSV analysis file.
    Updates existing plot if it exists, creates new one otherwise.

    Args:
        csv_filename: Path to the CSV file to plot
        ma_window: Window size for moving average (default: 10)
    """
    if not os.path.exists(csv_filename):
        print(f"File {csv_filename} not found")
        return

    # Read CSV
    df = pd.read_csv(csv_filename)

    # "Avg Steps to Reward" is 0 when no reward was found that round — treat as missing
    # so the plot shows gaps rather than misleading 0-steps data points.
    df['Avg Steps to Reward'] = df['Avg Steps to Reward'].replace(0, float('nan'))

    # Calculate moving averages (skip NaN via min_periods)
    df['MA_Reward'] = df['Reward Found %'].rolling(window=ma_window, min_periods=1).mean()
    df['MA_Steps'] = df['Avg Steps'].rolling(window=ma_window, min_periods=1).mean()
    df['MA_Tiles'] = df['Avg Tiles'].rolling(window=ma_window, min_periods=1).mean()
    df['MA_TilesPerStep'] = df['Avg Tiles Per Step'].rolling(window=ma_window, min_periods=1).mean()
    df['MA_StepsToReward'] = df['Avg Steps to Reward'].rolling(window=ma_window, min_periods=1).mean()
    if 'Avg Score' in df.columns:
        last_col = 'Avg Score'
        last_col_label = 'Average Analysis Score'
        last_col_ylabel = 'Score'
        last_col_color = 'brown'
    else:
        last_col = 'Avg Drones Terminated'
        last_col_label = 'Avg Drones Terminated by Hazard'
        last_col_ylabel = 'Drones'
        last_col_color = 'darkred'
    df['MA_LastCol'] = df[last_col].rolling(window=ma_window, min_periods=1).mean()
    
    # Get or create figure for this CSV file
    if csv_filename not in _plot_figures:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Training Progress - {os.path.basename(csv_filename)}', fontsize=16)
        _plot_figures[csv_filename] = (fig, axes)
        plt.ion()  # Turn on interactive mode
    else:
        fig, axes = _plot_figures[csv_filename]
        # Clear all axes
        for ax in axes.flat:
            ax.clear()
    
    # Plot each metric with moving average
    axes[0, 0].plot(df['TimeSteps'], df['Reward Found %'], marker='o', alpha=0.3, label='Raw')
    axes[0, 0].plot(df['TimeSteps'], df['MA_Reward'], linewidth=2, label=f'MA({ma_window})')
    axes[0, 0].set_title('Reward Found %')
    axes[0, 0].set_xlabel('TimeSteps')
    axes[0, 0].set_ylabel('%')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(df['TimeSteps'], df['Avg Steps'], marker='o', color='orange', alpha=0.3, label='Raw')
    axes[0, 1].plot(df['TimeSteps'], df['MA_Steps'], color='orange', linewidth=2, label=f'MA({ma_window})')
    axes[0, 1].set_title('Average Steps Taken')
    axes[0, 1].set_xlabel('TimeSteps')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(df['TimeSteps'], df['Avg Tiles'], marker='o', color='green', alpha=0.3, label='Raw')
    axes[0, 2].plot(df['TimeSteps'], df['MA_Tiles'], color='green', linewidth=2, label=f'MA({ma_window})')
    axes[0, 2].set_title('Average Tiles Discovered')
    axes[0, 2].set_xlabel('TimeSteps')
    axes[0, 2].set_ylabel('Tiles')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    axes[1, 0].plot(df['TimeSteps'], df['Avg Tiles Per Step'], marker='o', color='red', alpha=0.3, label='Raw')
    axes[1, 0].plot(df['TimeSteps'], df['MA_TilesPerStep'], color='red', linewidth=2, label=f'MA({ma_window})')
    axes[1, 0].set_title('Avg Tiles Per Step')
    axes[1, 0].set_xlabel('TimeSteps')
    axes[1, 0].set_ylabel('Tiles/Step')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Drop NaN rows for "Steps to Reward" — gaps mean reward wasn't found that round
    str_valid = df[['TimeSteps', 'Avg Steps to Reward', 'MA_StepsToReward']].dropna()
    axes[1, 1].plot(str_valid['TimeSteps'], str_valid['Avg Steps to Reward'], marker='o', color='purple', alpha=0.3, label='Raw (reward found)')
    axes[1, 1].plot(str_valid['TimeSteps'], str_valid['MA_StepsToReward'], color='purple', linewidth=2, label=f'MA({ma_window})')
    axes[1, 1].set_title('Avg Steps to Reward\n(gaps = reward not found)')
    axes[1, 1].set_xlabel('TimeSteps')
    axes[1, 1].set_ylabel('Steps')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    axes[1, 2].plot(df['TimeSteps'], df[last_col], marker='o', color=last_col_color, alpha=0.3, label='Raw')
    axes[1, 2].plot(df['TimeSteps'], df['MA_LastCol'], color=last_col_color, linewidth=2, label=f'MA({ma_window})')
    axes[1, 2].set_title(last_col_label)
    axes[1, 2].set_xlabel('TimeSteps')
    axes[1, 2].set_ylabel(last_col_ylabel)
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)

if __name__ == "__main__":
    x = -1
    while x not in ["1", "2", "3"]:
        x = input("Press (1) for the latest IPPO results, (2) for the latest MAPPO results, or (3) for the latest JOINT_PPO results...")
        if x == "1":
            plotAnalysisData("IPPO/map_15x15_analysis_Results.csv", 50)
            plotAnalysisData("IPPO/map_30x30_analysis_Results.csv", 50)
            plotAnalysisData("IPPO/map_45x45_analysis_Results.csv", 50)
        elif x == "2":
            plotAnalysisData("MAPPO/map_15x15_analysis_Results.csv", 50)
            plotAnalysisData("MAPPO/map_30x30_analysis_Results.csv", 50)
            plotAnalysisData("MAPPO/map_45x45_analysis_Results.csv", 50)
        elif x == "3":
            plotAnalysisData("JOINT_PPO/map_15x15_analysis_Results.csv", 50)
            plotAnalysisData("JOINT_PPO/map_30x30_analysis_Results.csv", 50)
            plotAnalysisData("JOINT_PPO/map_45x45_analysis_Results.csv", 50)
        else:
            print("Invalid input, please try again.")
    input("Press Enter to close...")
