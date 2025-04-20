from algorithms.ddpg import DDPG

def main():
    ddpg = DDPG(debug_per=True, prioritised_experience_replay=True, update_every=50)
    ddpg.train_batch(num_steps=1200000, benchmark=True)

if __name__ == "__main__":
    main()