"""
Diagnostic script to check if NCA reservoir responds to CartPole observations.

Usage:
    python diagnose_reservoir.py --logdir logs/train_nca_conserve/20260222-130839
"""

import os
import inspect
import numpy as np
import tensorflow as tf
from critical_nca import CriticalNCA
import utils
from evaluate_criticality import apply_conservation
import argparse

try:
    import gymnasium as gym
    GYM_VERSION = "gymnasium"
except ImportError:
    import gym
    GYM_VERSION = "gym"


def get_nca(args, ckpt=""):
    print("Loading NCA from: " + args.log_dir)
    valid_keys = set(inspect.signature(CriticalNCA.__init__).parameters.keys())
    valid_keys.discard("self")
    model_kwargs = {k: v for k, v in args.nca_model.items() if k in valid_keys}
    nca = CriticalNCA(**model_kwargs)
    
    ckpt_filename = ""
    if ckpt == "":
        checkpoint_filename = "checkpoint"
        with open(os.path.join(args.log_dir, checkpoint_filename), "r") as f:
            first_line = f.readline()
            start_idx = first_line.find(": ")
            ckpt_filename = first_line[start_idx+3:-2]
    else:
        ckpt_filename = os.path.basename(ckpt)
    
    nca.load_weights(os.path.join(args.log_dir, ckpt_filename))
    print(f"Model loaded: {ckpt_filename}")
    return nca


def _maybe_numpy(x):
    return x.numpy() if isinstance(x, tf.Tensor) else x


def encode_observation(observation, width, channel_n, encoding="continuous"):
    x = np.zeros((1, width, channel_n), dtype=np.float32)
    normalized_obs = np.array([
        np.clip(observation[0] / 4.8, -1, 1),
        np.clip(observation[1] / 2.0, -1, 1),
        np.clip(observation[2] / 0.418, -1, 1),
        np.clip(observation[3] / 3.0, -1, 1)
    ])
    
    if encoding == "thermometer":
        # Thermometer encoding: convert each obs to binary spatial pattern
        # 4 obs × 20 bits each, placed at positions 5-24, 30-49, 55-74, 80-99
        n_bits = 20
        starts = [5, 30, 55, 80]
        for i, obs_val in enumerate(normalized_obs):
            mapped = (obs_val + 1.0) / 2.0  # [-1,1] -> [0,1]
            n_active = int(round(mapped * n_bits))
            for b in range(n_bits):
                pos = starts[i] + b
                if pos < width:
                    x[0, pos, 0] = 1.0 if b < n_active else 0.0
    else:
        # Original continuous encoding with Gaussian window
        for i, obs_val in enumerate(normalized_obs):
            start_pos = 10 + i * 20
            for offset in range(-2, 3):
                pos = (start_pos + offset) % width
                weight = np.exp(-0.5 * offset**2)
                x[0, pos, 0] = obs_val * weight
    return x


def get_nca_features(nca, observation, width, timesteps, args=None, encoding="continuous"):
    x = encode_observation(observation, width, nca.channel_n, encoding=encoding)
    for t in range(timesteps):
        x = nca(x).numpy()
        if args is not None and hasattr(args, 'conserve') and args.conserve:
            x = _maybe_numpy(apply_conservation(x, args))
    return x.flatten()


def diagnose_reservoir(args, width=100, timesteps=5, encoding="continuous"):
    """Check if reservoir responds meaningfully to different observations."""
    nca = get_nca(args)
    
    print(f"\n=== Reservoir Diagnostics ===")
    print(f"Width: {width}, Timesteps: {timesteps}, Channels: {nca.channel_n}, Encoding: {encoding}")
    
    # Test observations
    test_obs = [
        np.array([0.0, 0.0, 0.0, 0.0]),      # Centered
        np.array([0.0, 0.0, 0.2, 0.0]),      # Pole tilted right
        np.array([0.0, 0.0, -0.2, 0.0]),     # Pole tilted left
        np.array([1.0, 0.0, 0.0, 0.0]),      # Cart right
        np.array([-1.0, 0.0, 0.0, 0.0]),     # Cart left
    ]
    
    labels = ["Centered", "Pole right", "Pole left", "Cart right", "Cart left"]
    
    features_list = []
    for obs, label in zip(test_obs, labels):
        features = get_nca_features(nca, obs, width, timesteps, args, encoding=encoding)
        features_list.append(features)
        print(f"\n{label}:")
        print(f"  Feature mean: {np.mean(features):.6f}")
        print(f"  Feature std:  {np.std(features):.6f}")
        print(f"  Feature min:  {np.min(features):.6f}")
        print(f"  Feature max:  {np.max(features):.6f}")
        print(f"  Non-zero:     {np.count_nonzero(features)}/{len(features)}")
    
    # Check feature diversity
    print(f"\n=== Feature Diversity ===")
    for i in range(len(features_list)):
        for j in range(i+1, len(features_list)):
            diff = np.linalg.norm(features_list[i] - features_list[j])
            corr = np.corrcoef(features_list[i], features_list[j])[0,1]
            print(f"{labels[i]} vs {labels[j]}:")
            print(f"  L2 distance: {diff:.6f}")
            print(f"  Correlation: {corr:.6f}")
    
    # Check if features are mostly zeros
    all_features = np.array(features_list)
    nonzero_ratio = np.count_nonzero(all_features) / all_features.size
    print(f"\n=== Overall Statistics ===")
    print(f"Non-zero ratio: {nonzero_ratio:.4f}")
    print(f"Feature variance: {np.var(all_features):.6f}")
    
    if nonzero_ratio < 0.01:
        print("\n⚠️  WARNING: Reservoir is mostly zeros! NCA may not be processing inputs.")
    elif np.var(all_features) < 1e-6:
        print("\n⚠️  WARNING: Very low feature variance! Reservoir may not be discriminative.")
    else:
        print("\n✓ Reservoir appears to be responding to inputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose NCA reservoir")
    parser.add_argument("--logdir", required=True, help="Path to NCA checkpoint directory")
    parser.add_argument("--width", type=int, default=100, help="Reservoir width (default: 100)")
    parser.add_argument("--timesteps", type=int, default=5, help="NCA timesteps (default: 5)")
    parser.add_argument("--encoding", default="continuous", choices=["continuous", "thermometer"],
                        help="Input encoding: continuous (original) or thermometer (binary)")
    p_args = parser.parse_args()
    
    if p_args.logdir:
        args_filename = os.path.join(p_args.logdir, "args.json")
        argsio = utils.ArgsIO(args_filename)
        args = argsio.load_json()
        args.log_dir = p_args.logdir
        
        diagnose_reservoir(args, width=p_args.width, timesteps=p_args.timesteps, encoding=p_args.encoding)
    else:
        print("Add --logdir [path/to/checkpoint]")
