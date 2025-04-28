import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def plot_performance():
    files = glob.glob('./out/*_bench.npy')

    if not files:
        print("No bench files found in ./out directory")
        return
    
    arrays = []
    filenames = []
    for file in files:
        try:
            data = np.load(file)
            if data.ndim == 2:  # Only keep 2D arrays
                arrays.append(data)
                filenames.append(os.path.basename(file))
            else:
                print(data.ndim)
                print(f"Skipping {file} - not 1D (shape: {data.shape})")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not arrays:
        print("No 1D arrays found")
        return
    
    # Create single figure
    plt.figure(figsize=(10, 6))
    
    # Plot all 1D arrays on same graph
    for arr, filename in zip(arrays, filenames):
        timestep = np.array([(i + 1) for i in range(len(arr))])
        mean = np.array([obs[0] for obs in arr])
        sd = np.array([obs[1] for obs in arr])

        upper = mean + sd
        lower = mean - sd

        plt.plot(timestep, mean, label=filename)
        plt.fill_between(
            timestep,
            lower,
            upper,
            alpha=0.15,
            linewidth=0,
            color=plt.gca().lines[-1].get_color()
        )
    
    # Add plot decorations
    plt.title("Performance")
    plt.xlabel("Timestep * 1e4")
    plt.ylabel("Mean Episode Reward")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside plot
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_performance()