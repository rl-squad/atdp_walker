from algorithms.td3 import TD3

def main():
    td3 = TD3(update_every=50)
    td3.train_batch(num_envs=8)

if __name__ == "__main__":
    main()