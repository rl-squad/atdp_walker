from algorithms.ddpg import DDPG

def main():
    ddpg = DDPG(update_every=50)
    ddpg.train(benchmark=True)

if __name__ == "__main__":
    main()