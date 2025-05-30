"""Used for plotting PER buffer information for debugging"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def plot_priority_stats():
    # Find all priority files in output directory
    files = glob.glob('./out/*_priorities.npy')
    if not files:
        print("No priority files found in ./out directory")
        return
    
    plt.rcParams.update({'font.size': 12})
    
    # Create figure with 4 subplots
    _, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(6, 7))

    # Plot each file's data
    for file in files:

        # Extract filename for labeling
        base_name = os.path.basename(file).replace('_priorities.npy', '')
        
        # Load data and prepare timesteps
        data = np.load(file)
        steps = np.array([(i + 1) for i in range(len(data))]) * 1e4

        # Extract priority metrics
        stored_max = data[:, 0]
        stored_min = data[:, 1]
        max_is_weight = data[:, 2]
        actual_max = data[:, 3]
        actual_min = data[:, 4]
        actual_mean = data[:, 5]
        beta = data[:, 6]
        
        # Plot data points
        # ax1.plot(steps, stored_max, '--', label=f'{base_name}_stored_max')
        ax1.plot(steps, actual_max, '-', label=f'{base_name}_max_priority')
        
        # ax2.plot(steps, stored_min, '--', label=f'{base_name}_stored_min')
        ax2.plot(steps, actual_min, '-', label=f'{base_name}_min_priority')
        
        ax3.plot(steps, actual_mean, label=f'{base_name}_mean_priority')

        ax4.plot(steps, max_is_weight, label=f'{base_name}_max_is_weight')

        ax5.plot(steps, beta, label=f'{base_name}_beta')
    
    ax1.set_title("Prioritised Experience Replay Buffer Diagnostics",fontsize=14, fontweight="bold")
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Priority Value')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_ylabel('Priority Value')
    ax2.set_xlabel('Timesteps')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    ax3.set_ylabel('Priority Value')
    ax3.set_xlabel('Timesteps')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)

    ax4.set_ylabel('IS Weight Value')
    ax4.set_xlabel('Timesteps')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)

    ax5.set_ylabel('Beta value')
    ax5.set_xlabel('Timesteps')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_priority_stats()