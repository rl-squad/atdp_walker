import pandas as pd
import matplotlib.pyplot as plt


class TrainingComparator:
    def __init__(self, log_paths: dict):
        """
        log_paths: dict of {label: path_to_csv}
        Example:
        {
            "Stable-Baselines3": "results/results_stable/training_log.csv",
            "CleanRL": "results/results_cleanrl/training_log.csv"
        }
        """
        self.log_paths = log_paths
        self.data = self.load_data()

    def load_data(self):
        """Load CSV logs for each method into a dictionary of DataFrames."""
        data = {}
        for label, path in self.log_paths.items():
            try:
                df = pd.read_csv(path)
                data[label] = df
            except FileNotFoundError:
                print(f"[Warning] File not found: {path}")
        return data

    def plot_rewards(self):
        """Plot episode rewards over training for each method."""
        plt.figure(figsize=(10, 5))
        for label, df in self.data.items():
            plt.plot(df["Episode"], df["Reward"], label=label)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode Reward Comparison")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_loss(self):
        """If 'Loss' column exists, compare it."""
        plt.figure(figsize=(10, 5))
        plotted = False
        for label, df in self.data.items():
            if "Loss" in df.columns:
                plt.plot(df["Episode"], df["Loss"], label=label)
                plotted = True
        if plotted:
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.title("Loss Comparison (if available)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("No loss data found in any log.")


if __name__ == "__main__":
    logs = {
        "Stable-Baselines3": "results/results_stable/training_log.csv",
        "CleanRL": "results/results_cleanrl/training_log.csv"
    }

    comparator = TrainingComparator(logs)
    comparator.plot_rewards()
    comparator.plot_loss()
