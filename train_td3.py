from algorithms.td3 import TD3

def main():
    td3 = TD3(prioritised_experience_replay=True, update_every=50)
    td3.train_batch(num_steps=1200000, benchmark=True)

if __name__ == "__main__":
    main()