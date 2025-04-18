from algorithms.noisy_ddpg import NoisyDDPG

def main():
    ddpg = NoisyDDPG(update_every=50, device="cpu")
    ddpg.train_batch(num_envs=8, benchmark=True)

if __name__ == "__main__":
    main()