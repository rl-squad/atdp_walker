from algorithms.td3_per import TD3PER
import os

def main():

    base = "td3per_final_"
    
    for i in range(3):

        os.environ["OUT"] = base + str(i)

        td3_per = TD3PER(
            device="cpu", # see which runs faster depending on machine: "cpu", "cuda" or "mps"
            seed=i
        )

        td3_per.train_batch(
            log_save_policy_weights=True,
            num_steps=6000000,
            benchmark=True
        )

if __name__ == "__main__":
    main()