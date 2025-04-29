from algorithms.sac_per import SACPER
import os

def main():

    base = "sac_prioritised_"

    # SAC with and without PER optimisation
    for i in range(2):
        
        # 10 agents per test
        for j in range(10):

            os.environ["OUT"] = base + str(i) + "_" + str(j)

            sac_per = SACPER(
                device="cpu", # see which runs faster depending on machine: "cpu", "cuda" or "mps"
                prioritised_xp_replay=bool(i==0),
                seed=j,
            )

            sac_per.train_batch(
                num_steps=3000000,
                benchmark=True
            )

            # print(os.getenv("OUT"), bool(i==0))
            # i=0 SAC with Prioritised Experience Replay
            # i=1 SAC with Uniform Replay

if __name__ == "__main__":
    main()
