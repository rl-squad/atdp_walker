from algorithms.td3_noisy_prioritised import NoisyPrioritisedTD3
import os
import multiprocessing as mp

base = "td3_noisy_prio_"

def experiment(i, j):
    os.environ["OUT"] = base + str(i) + "_" + str(j)

    td3_noisy_prio = NoisyPrioritisedTD3(
        seed=j,
        prioritised_xp_replay=(i != 1),
        debug_per=(i != 1),
        noisy_net=(i != 0),
    )
    
    td3_noisy_prio.train_batch(num_steps=3000000, benchmark=True)

def main():
    seeds = range(10)
    processes = [mp.Process(target=experiment, args=(2, s)) for s in seeds]

    for p in processes:
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")

    main()