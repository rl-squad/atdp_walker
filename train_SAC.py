from algorithms.sac import SAC

def main():
    sac = SAC(update_every=100, device="cpu")  
    sac.train(num_episodes=5000, benchmark=True)

if __name__ == "__main__":
    main()
