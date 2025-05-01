#!/usr/bin/env python
import os
import random
import numpy as np
import torch
import gymnasium as gym
import shap
import pickle
import sys

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
policy_path       = "SHAP/ddpg_batch.pth"
num_bg_samples    = 3000                    # How many on-policy states for background
num_explain_samples = 300                   # How many on-policy states to explain
kmeans_clusters   = 800                     # Number of clusters for k-means summarization
nsamples          = 1500                    # Integration samples for SHAP (kernel/permutation)
algorithm         = "kernel"                # "kernel", "permutation", "exact", "gradient", etc.
seed              = 42                      
output_file       = "SHAP/shap_output.pkl"
# -----------------------------------------------------------------------------

# Add path to allow importing local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from algorithms.common import PolicyNetwork, DEFAULT_DEVICE, ACTION_DIM, STATE_DIM


def load_policy(path):
    policy = PolicyNetwork().to(DEFAULT_DEVICE)
    policy.load_state_dict(torch.load(path, map_location=DEFAULT_DEVICE))
    policy.eval()
    print(f"Loaded policy from {path}. Input dim: {policy.fc1.in_features}")
    return policy


def init_env(seed=None):
    env = gym.make("Walker2d-v5", render_mode="rgb_array")
    if seed is not None:
        env.reset(seed=seed)
    return env


def collect_states_on_policy(env, policy, num_samples):
    states = []
    obs, _ = env.reset()
    for _ in range(num_samples):
        states.append(obs)
        tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEFAULT_DEVICE)
        with torch.no_grad():
            act = policy(tensor).cpu().numpy()[0]
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            obs, _ = env.reset()
    return np.array(states)


def wrapped_policy(x, policy):
    xt = torch.tensor(x, dtype=torch.float32).to(DEFAULT_DEVICE)
    with torch.no_grad():
        out = policy(xt)
    return out.cpu().numpy()


def run_shap_computation():
    # reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # load
    policy = load_policy(policy_path)
    env    = init_env(seed)

    # collect data
    bg_data = collect_states_on_policy(env, policy, num_bg_samples)
    expl_data = collect_states_on_policy(env, policy, num_explain_samples)

    # feature names
    if STATE_DIM == 17:
        feature_names = [
            "torso_height", "torso_angle", "right_thigh_angle", "right_leg_angle", "right_foot_angle",
            "left_thigh_angle", "left_leg_angle", "left_foot_angle", "torso_x_velocity", 
            "torso_z_velocity", "torso_angular_velocity", "right_thigh_angular_velocity", "right_leg_angular_velocity",
            "right_foot_angular_velocity", "left_thigh_angular_velocity", "left_leg_angular_velocity", "left_foot_angular_velocity"
        ]
    else:
        feature_names = [f"feature_{i}" for i in range(STATE_DIM)]

    # k-means background summary
    print(f"Summarizing {len(bg_data)} states into {kmeans_clusters} clusters...")
    bg_summary = shap.kmeans(bg_data, kmeans_clusters).data

    # wrapped fn
    fn = lambda x: wrapped_policy(x, policy)

    # compute
    if algorithm.lower() == "kernel":
        print(f"Using KernelExplainer (nsamples={nsamples})")
        ke = shap.KernelExplainer(fn, bg_summary)
        raw = ke.shap_values(expl_data, nsamples=nsamples)
        base = ke.expected_value
        expl = shap.Explanation(values=raw,
                                base_values=base,
                                data=expl_data,
                                feature_names=feature_names)
    else:
        print(f"Using SHAP Explainer algorithm='{algorithm}' (nsamples={nsamples})")
        expl = shap.Explainer(fn, bg_summary, algorithm=algorithm)(expl_data, nsamples=nsamples)

    # save
    with open(output_file, 'wb') as f:
        pickle.dump({
            'shap_values': expl,
            'states_to_explain': expl_data,
            'feature_names': feature_names
        }, f)
    print(f"Saved SHAP results to {output_file}")


if __name__ == "__main__":
    run_shap_computation()
