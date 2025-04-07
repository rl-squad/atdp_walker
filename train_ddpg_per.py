from algorithms.ddpg_per import DDPGPER

def main():
    ddpg_per = DDPGPER(update_every=50)
    ddpg_per.train()

if __name__ == "__main__":
    main()