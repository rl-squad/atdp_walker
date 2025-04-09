from algorithms.sac import SAC


# Instantiate and train the SAC agent
sac = SAC()
sac.train(num_episodes=5000)
sac.save_policy("./out/sac_policy.pth")