from algorithms.ddpg_per import DDPGPER

def main():
    ddpg_per = DDPGPER(update_every=50)
    ddpg_per.train_batch(num_steps=1200000, benchmark=True)

if __name__ == "__main__":
    main()