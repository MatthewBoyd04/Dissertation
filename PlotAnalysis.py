import pandas as pd
import matplotlib.pyplot as plt
import os

# Store figure references for each CSV file
_plot_figures = {}

def plotAnalysisData(csv_filename):
    """
    Plots all metrics from a CSV analysis file.
    Updates existing plot if it exists, creates new one otherwise.
    
    Args:
        csv_filename: Path to the CSV file to plot
    """
    if not os.path.exists(csv_filename):
        print(f"File {csv_filename} not found")
        return
    
    # Read CSV
    df = pd.read_csv(csv_filename)
    
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
    
    # Plot each metric
    axes[0, 0].plot(df['TimeSteps'], df['Reward Found %'], marker='o')
    axes[0, 0].set_title('Reward Found %')
    axes[0, 0].set_xlabel('TimeSteps')
    axes[0, 0].set_ylabel('%')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(df['TimeSteps'], df['Avg Steps'], marker='o', color='orange')
    axes[0, 1].set_title('Average Steps Taken')
    axes[0, 1].set_xlabel('TimeSteps')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(df['TimeSteps'], df['Avg Tiles'], marker='o', color='green')
    axes[0, 2].set_title('Average Tiles Discovered')
    axes[0, 2].set_xlabel('TimeSteps')
    axes[0, 2].set_ylabel('Tiles')
    axes[0, 2].grid(True)
    
    axes[1, 0].plot(df['TimeSteps'], df['Avg Tiles Per Step'], marker='o', color='red')
    axes[1, 0].set_title('Avg Tiles Per Step')
    axes[1, 0].set_xlabel('TimeSteps')
    axes[1, 0].set_ylabel('Tiles/Step')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(df['TimeSteps'], df['Avg Steps to Reward'], marker='o', color='purple')
    axes[1, 1].set_title('Avg Steps to Reward')
    axes[1, 1].set_xlabel('TimeSteps')
    axes[1, 1].set_ylabel('Steps')
    axes[1, 1].grid(True)
    
    axes[1, 2].plot(df['TimeSteps'], df['Avg Score'], marker='o', color='brown')
    axes[1, 2].set_title('Average Analysis Score')
    axes[1, 2].set_xlabel('TimeSteps')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)

if __name__ == "__main__":
    # Test with example file
    plotAnalysisData("map_30x30_analysis_Results.csv")
