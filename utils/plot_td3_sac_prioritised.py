import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from collections import defaultdict

def plot_td3_sac_prio():

    # reusing TD3 + prio data from td3_noisy
    td3_prio_files = glob.glob('./out/td3_noisy_prio_0_*_bench.npy')
    # reusing TD3 data from td3_ablation
    td3_baseline_files = glob.glob('./out/td3_ablation_7_*_bench.npy')
    random_walker = glob.glob('./out/random_walker_bench.npy')
    sac_prio_files = glob.glob('./out/sac_prioritised_*_*_bench.npy')

    # group files by experiment
    experiments = defaultdict(list)

    plt.rcParams.update({'font.size': 13})

    # 0 TD3
    # 1 TD3 + prioritised_xp_replay
    # 2 SAC
    # 3 SAC + prioritised_xp_replay

    for file in td3_baseline_files:
        experiment_id = 0
        experiments[experiment_id].append(file)
    
    for file in td3_prio_files:
        basename = os.path.basename(file)
        parts = basename.split('_')
        experiment_id = int(parts[3]) + 1
        experiments[experiment_id].append(file)
    
    for file in sac_prio_files:
        basename = os.path.basename(file)
        parts = basename.split('_')
        experiment_id = int(parts[2]) + 2

        # Swap experiment ids since sac_0 was prioritised and sac_1 was uniform
        if experiment_id == 3:
            experiment_id -= 1
        elif experiment_id == 2:
            experiment_id += 1
        
        experiments[experiment_id].append(file)
    
    plt.figure(figsize=(12, 8))

    colors = ["#4E79A7", "#59A14F" , "#F28E2B" , "#E15759"]
    # colors = ["#E69F00", "#CC79A7", "#009E73", "#56B4E9"]
    
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
        # extract the means of each agent, shape (mean, sd)
        means = stacked_data[:, :, 0]

        # mean eps reward across 10 agents
        combined_mean = np.mean(means, axis=0)
        # standard error of the mean
        sem = scipy.stats.sem(means, axis=0)
        ci = 1.96 * sem

        timesteps = np.arange(len(combined_mean)) * 1e4 # benchmark every 10k steps
        label = f"Exp {experiment_id}: " + get_label(experiment_id)
        plt.plot(timesteps, combined_mean, label=label, color=colors[experiment_id])
        plt.fill_between(
            timesteps,
            combined_mean - ci,
            combined_mean + ci,
            alpha=0.2,
            color=colors[experiment_id],
            linewidth=0
        )

    plt.title("SAC and TD3 with Uniform Replay or Prioritised Experience " \
              "Replay (Mean ± 95% CI across 10 agents)", fontweight="bold")
    plt.xlabel("Training Steps")
    plt.ylabel("Mean Episode Reward")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def get_label(exp_id):
    components = ["TD3" if exp_id < 2 else "SAC"]
    if exp_id % 2 == 1: components.append("prioritised_xp_replay")
    return " + \n".join(components)

if __name__ == "__main__":
    plot_td3_sac_prio()