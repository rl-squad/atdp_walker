import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def plot_1d_arrays():
    # Find all .npy files in current directory
    npy_files = glob.glob('*.npy')
    
    if not npy_files:
        print("No .npy files found in current directory")
        return
    
    # Load only 1D arrays
    arrays = []
    filenames = []
    for file in npy_files:
        try:
            data = np.load(file)
            if data.ndim == 1:  # Only keep 1D arrays
                arrays.append(data)
                filenames.append(os.path.basename(file))
            else:
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
        plt.plot(arr, label=filename)
    
    # Add plot decorations
    plt.title("1D Arrays Comparison")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside plot
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_1d_arrays()