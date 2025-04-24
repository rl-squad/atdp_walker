from algorithms.td3_ablation import TD3_Ablation
import os

def main():

    # 2 ** 3 tests
    for i in range(8):

        base = "td3_ablation_"
        
        # 10 agents per test
        for j in range(10):
            
            os.environ["OUT"] = base + str(i) + "_" + str(j)

            td3_ablation = TD3_Ablation(
                double_clipped_Q=bool(i & 4),
                delayed_policy_updates=bool(i & 2),
                target_policy_smoothing=bool(i & 1),
            )
            
            td3_ablation.train_batch(num_steps=6000000, benchmark=True)

            # print(os.getenv("OUT"), bool(i & 4), bool(i & 2), bool(i & 1))

            # 000 0 DDPG
            # 001 1 target_policy_smoothing
            # 010 2 delayed_policy_updates
            # 011 3 target_policy_smoothing + delayed_policy_updates
            # 100 4 double_clipped_Q
            # 101 5 double_clipped_Q + target_policy_smoothing
            # 110 6 double_clipped_Q delayed_policy_updates
            # 111 7 TD3

if __name__ == "__main__":
    main()