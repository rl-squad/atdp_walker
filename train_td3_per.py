from algorithms.td3_per import TD3PER

def main():
    td3_per = TD3PER()
    td3_per.train_batch(num_steps=1200000, benchmark=True)

if __name__ == "__main__":
    main()