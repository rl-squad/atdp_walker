from algorithms.td3_noisy_prioritised import NoisyPrioritisedTD3
import os

def main():

    for i in range(3):

        base = "td3_noisy_prio_"
        
        # 10 agents per test
        for j in range(10):
            
            os.environ["OUT"] = base + str(i) + "_" + str(j)

            td3_noisy_prio = NoisyPrioritisedTD3(
                prioritised_xp_replay=(i != 1),
                debug_per=(i != 1),
                noisy_net=(i != 0),
            )
            
            td3_noisy_prio.train_batch(num_steps=3000000, benchmark=True)

            # print(os.getenv("OUT"), i != 1, i != 0)

            # 0 TD3 + prioritised_xp_replay
            # 1 TD3 + noisy_policy_network
            # 2 TD3 + prioritised_xp_replay + noisy_policy_network

if __name__ == "__main__":
    main()