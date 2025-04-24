from algorithms.td3_ablation import TD3_Ablation

def main():
    td3 = TD3_Ablation(
        double_clipped_Q=False,
        delayed_policy_updates=False,
        target_policy_smoothing=False,
    )
    td3.train_batch(num_steps=1200000, benchmark=True)

if __name__ == "__main__":
    main()