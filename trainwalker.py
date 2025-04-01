import gymnasium as gym
from stable_baselines3 import SAC

env = gym.make(
    "Walker2d-v5",
    #xml_file = ".xml",
    forward_reward_weight=0.5,
    ctrl_cost_weight=1e-2,
    healthy_reward=2,
    terminate_when_unhealthy=True,
    healthy_z_range=(1, 2),
    healthy_angle_range=(-0.5, 0.5),
    reset_noise_scale=1e-3,
    exclude_current_positions_from_observation=True
    )

model = SAC(
    "MlpPolicy",
    env, 
    verbose=1,
    ent_coef='auto_0.1' 
    )

model.learn(total_timesteps=1_000_000)

model.save("sac_walker2d_v2")

env.close()
