from algorithms.td3_rvm import TD3RVM

td3 = TD3RVM()
td3.train(num_episodes=5000)