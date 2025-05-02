import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def plot_td3_ablation():
    # find all experiment benchmark files
    files = glob.glob('./out/td3_ablation_*_*_bench.npy')
    random_walker = glob.glob('./out/random_walker_bench.npy')
    
    if not files:
        print("No benchmark files found in ./out")
        return

    # group files by experiment
    experiments = {}
    for file in files:
        basename = os.path.basename(file)
        parts = basename.split('_')
        experiment_id = parts[2]
        if experiment_id not in experiments:
            experiments[experiment_id] = []
        experiments[experiment_id].append(file)

    plt.rcParams.update({'font.size': 12})

    plt.figure(figsize=(12, 8))
    colors = plt.cm.Dark2(np.linspace(0, 1, len(experiments)))
    
    # plot random agent
    timesteps = np.arange(300) * 1e4
    random_walker_data = np.load(random_walker[0])
    plt.plot(
        np.arange(len(random_walker_data)) * 1e4,
        random_walker_data[:, 0],
        linestyle="dashed",
        label="Random agent",
        color="black"
    )

    # process each experiment in sorted order
    for experiment_id, exp_files in sorted(experiments.items()):

        all_agents_data = []
        for file in exp_files:
            data = np.load(file)
            all_agents_data.append(data)

        # stack all agents
        stacked_data = np.stack(all_agents_data)
        # extract the means, shape (mean, sd)
        means = stacked_data[:, :, 0]

        # mean eps reward across 10 agents
        combined_mean = np.mean(means, axis=0)
        # standard error of the mean
        sem = scipy.stats.sem(means, axis=0)
        ci = 1.96 * sem

        timesteps = np.arange(len(combined_mean)) * 1e4 # benchmark every 10k steps
        label = f"Exp {experiment_id}: " + get_ablation_label(int(experiment_id))
        plt.plot(timesteps, combined_mean, label=label, color=colors[int(experiment_id)])
        plt.fill_between(
            timesteps,
            combined_mean - ci,
            combined_mean + ci,
            alpha=0.2,
            color=colors[int(experiment_id)],
            linewidth=0
        )

    plt.title("TD3 Ablation Study (Mean ± 95% CI across 10 agents)", fontweight="bold")
    plt.xlabel("Training Steps")
    plt.ylabel("Mean Episode Reward")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 000 0 DDPG
    # 001 1 target_policy_smoothing
    # 010 2 delayed_policy_updates
    # 011 3 target_policy_smoothing + delayed_policy_updates
    # 100 4 double_clipped_Q
    # 101 5 double_clipped_Q + target_policy_smoothing
    # 110 6 double_clipped_Q delayed_policy_updates
    # 111 7 TD3

def get_ablation_label(exp_id):
    if exp_id == 7:
        return "TD3"
    components = ["DDPG"]
    if exp_id & 4: components.append("double_clipped_Q")
    if exp_id & 2: components.append("delayed_policy_updates")
    if exp_id & 1: components.append("target_policy_smoothing")
    return " + \n".join(components)

if __name__ == "__main__":
    plot_td3_ablation()