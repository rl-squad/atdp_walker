from algorithms.ddpg import DDPG

def main():
    ddpg = DDPG(update_every=50)
    ddpg.train_batch(batch_size=8)

if __name__ == "__main__":
    main()