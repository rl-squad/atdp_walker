#!/usr/bin/env python
import os
import argparse
import numpy as np
import torch
import gymnasium as gym
import shap
import pickle

# Import your local modules
from algorithms.common import PolicyNetwork, DEFAULT_DEVICE, ACTION_DIM, STATE_DIM

def load_policy(policy_path):
    policy = PolicyNetwork().to(DEFAULT_DEVICE)
    policy.load_state_dict(torch.load(policy_path, map_location=DEFAULT_DEVICE))
    policy.eval()
    print(f"Loaded policy from {policy_path}. Expected input dim: {policy.fc1.in_features}")
    return policy

def init_env():
    env = gym.make("Walker2d-v5", render_mode="rgb_array")
    return env

def collect_states_on_policy(env, policy, num_samples=500):
    states = []
    observation, _ = env.reset()
    for _ in range(num_samples):
        states.append(observation)
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(DEFAULT_DEVICE)
        with torch.no_grad():
            action = policy(obs_tensor).cpu().numpy()[0]
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset()
    return np.array(states)

def wrapped_policy(x, policy):
    # Ensure that x is converted to a torch tensor on the proper device
    x_tensor = torch.tensor(x, dtype=torch.float32).to(DEFAULT_DEVICE)
    with torch.no_grad():
        out = policy(x_tensor)
    return out.detach().cpu().numpy()

def run_shap_computation(policy_path, num_bg_samples, num_explain_samples, output_file):
    policy = load_policy(policy_path)
    env = init_env()

    background_data = collect_states_on_policy(env, policy, num_samples=num_bg_samples)

    background_summary = shap.kmeans(background_data, 50)
    background_summary_array = background_summary.data

    # Create a wrapped function so inputs are properly converted
    wrapped_fn = lambda x: wrapped_policy(x, policy)

    explainer = shap.Explainer(wrapped_fn, background_summary_array)

    states_to_explain = collect_states_on_policy(env, policy, num_samples=num_explain_samples)

    shap_values = explainer(states_to_explain)

    # Save the SHAP Explanation object along with states and metadata.
    output_data = {
        "shap_values": shap_values,  # This is the Explanation object
        "states_to_explain": states_to_explain,
        "feature_names": [
            "torso_height", "torso_angle", "left_thigh_angle", "left_knee_angle", "left_ankle_angle",
            "right_thigh_angle", "right_knee_angle", "right_ankle_angle", "left_thigh_velocity", 
            "left_knee_velocity", "left_ankle_velocity", "right_thigh_velocity", "right_knee_velocity",
            "right_ankle_velocity", "horizontal_velocity", "vertical_velocity", "angular_velocity"
        ] if STATE_DIM == 17 else [f"feature_{i}" for i in range(STATE_DIM)]
    }
    with open(output_file, "wb") as f:
        pickle.dump(output_data, f)
    print(f"SHAP computation complete and saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Compute SHAP values for a policy in Walker2d")
    parser.add_argument("--policy_path", type=str, default="ddpg_batch.pth",
                        help="Path to the policy checkpoint file")
    parser.add_argument("--num_bg", type=int, default=200,
                        help="Number of background samples")
    parser.add_argument("--num_explain", type=int, default=50,
                        help="Number of on-policy samples to explain")
    parser.add_argument("--output_file", type=str, default="shap_output.pkl",
                        help="File where computed SHAP results will be stored")
    args = parser.parse_args()

    run_shap_computation(args.policy_path, args.num_bg, args.num_explain, args.output_file)

if __name__ == "__main__":
    main()
