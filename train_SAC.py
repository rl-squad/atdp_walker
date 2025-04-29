from algorithms.sac import SAC

def main():
    sac = SAC()  
    sac.train_batch(num_steps=3000000, benchmark=True)

if __name__ == "__main__":
    main()
