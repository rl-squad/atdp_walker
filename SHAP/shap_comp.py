#!/usr/bin/env python
import os
import argparse
import numpy as np
import torch
import gymnasium as gym
import shap
import pickle
import sys

# Add path to allow for importing local module from a different subfolder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

def run_shap_computation(policy_path, num_bg_samples, num_explain_samples, output_file, kmeans_clusters, nsamples):
    policy = load_policy(policy_path)
    env = init_env()

    background_data = collect_states_on_policy(env, policy, num_samples=num_bg_samples)

    # Use a larger number of clusters if specified.
    background_summary = shap.kmeans(background_data, kmeans_clusters)
    background_summary_array = background_summary.data

    wrapped_fn = lambda x: wrapped_policy(x, policy)

    # Pass the nsamples parameter when computing SHAP values.
    explainer = shap.Explainer(wrapped_fn, background_summary_array)
    
    states_to_explain = collect_states_on_policy(env, policy, num_samples=num_explain_samples)

    # Pass nsamples here if supported by the explainer:
    shap_values = explainer(states_to_explain)

    output_data = {
        "shap_values": shap_values,
        "states_to_explain": states_to_explain,
        "feature_names": [
            "torso_height", "torso_angle", "right_thigh_angle", "right_leg_angle", "right_foot_angle",
            "left_thigh_angle", "left_leg_angle", "left_foot_angle", "torso_x_velocity", 
            "torso_z_velocity", "torso_angular_velocity", "right_thigh_angular_velocity", "right_leg_angular_velocity",
            "right_foot_angular_velocity", "left_thigh_angular_velocity", "left_leg_angular_velocity", "left_foot_angular_velocity"
        ] if STATE_DIM == 17 else [f"feature_{i}" for i in range(STATE_DIM)]
    }
    with open(output_file, "wb") as f:
        pickle.dump(output_data, f)
    print(f"SHAP computation complete and saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compute SHAP values for a policy in Walker2d")
    parser.add_argument("--policy_path", type=str, default="SHAP/ddpg_batch.pth",
                        help="Path to the policy checkpoint file")
    parser.add_argument("--num_bg", type=int, default=500,
                        help="Number of background samples")
    parser.add_argument("--num_explain", type=int, default=100,
                        help="Number of on-policy samples to explain")
    parser.add_argument("--kmeans_clusters", type=int, default=100,
                        help="Number of clusters for background summarization using kmeans")
    # Optionally add an argument for nsamples if needed:
    parser.add_argument("--nsamples", type=int, default=500,
                        help="Number of samples used in SHAP integration")
    parser.add_argument("--output_file", type=str, default="SHAP/shap_output.pkl",
                        help="File where computed SHAP results will be stored")
    args = parser.parse_args()
    
    run_shap_computation(args.policy_path, args.num_bg, args.num_explain, args.output_file, args.kmeans_clusters, args.nsamples)


if __name__ == "__main__":
    main()
