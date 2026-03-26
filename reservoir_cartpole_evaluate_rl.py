"""
CartPole Evaluation with RL-trained Policy/Q-network.

Evaluates NCA reservoir with REINFORCE policy or Q-learning network.

Usage:
    # Evaluate REINFORCE policy
    python reservoir_cartpole_evaluate_rl.py \
        --logdir logs/train_nca_conserve/20260222-130839 \
        --policy cartpole_policy_conserving_reinforce.pkl \
        --num_runs 10 \
        --timesteps 5 \
        --record_gif
    
    # Evaluate Q-learning network
    python reservoir_cartpole_evaluate_rl.py \
        --logdir logs/train_nca_conserve/20260222-130839 \
        --policy cartpole_qnetwork_conserving_qlearning.pkl \
        --num_runs 10 \
        --timesteps 5 \
        --record_gif
"""

import os
import inspect
import numpy as np
import tensorflow as tf
from critical_nca import CriticalNCA
import utils
from evaluate_criticality import apply_conservation
import argparse
import pickle

try:
    import gymnasium as gym
    GYM_VERSION = "gymnasium"
except ImportError:
    import gym
    GYM_VERSION = "gym"

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False


class LinearQReadout:
    """
    Q(s,a) = W_out · x
    where x = reservoir state
    Output: 2 Q-values (one for each action)
    """
    def __init__(self, reservoir_size, n_actions=2):
        self.W = np.zeros((n_actions, reservoir_size), dtype=np.float32)
        
    def q_values(self, reservoir_state):
        return self.W @ reservoir_state
        
    def select_action(self, reservoir_state, epsilon=0.0):
        """Greedy action selection (no exploration during evaluation)."""
        q_vals = self.q_values(reservoir_state)
        return np.argmax(q_vals)


def get_nca(args, ckpt=""):
    """Load trained NCA from checkpoint directory."""
    print("Loading NCA from: " + args.log_dir)

    valid_keys = set(inspect.signature(CriticalNCA.__init__).parameters.keys())
    valid_keys.discard("self")
    model_kwargs = {k: v for k, v in args.nca_model.items() if k in valid_keys}

    nca = CriticalNCA(**model_kwargs)
    nca.dmodel.summary()

    ckpt_filename = ""
    if ckpt == "":
        checkpoint_filename = "checkpoint"
        with open(os.path.join(args.log_dir, checkpoint_filename), "r") as f:
            first_line = f.readline()
            start_idx = first_line.find(": ")
            ckpt_filename = first_line[start_idx+3:-2]
    else:
        ckpt_filename = os.path.basename(ckpt)

    print("Testing model with lowest training loss...")
    nca.load_weights(os.path.join(args.log_dir, ckpt_filename))
    print(f"Model loaded: {ckpt_filename}")

    return nca


def load_policy(policy_file):
    """Load trained policy or Q-network."""
    print(f"Loading policy/Q-network from: {policy_file}")
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)
    print("Policy/Q-network loaded")
    return policy


def _maybe_numpy(x):
    return x.numpy() if isinstance(x, tf.Tensor) else x


def _to_numpy(x):
    """Convert to numpy array if TF tensor."""
    return x.numpy() if isinstance(x, tf.Tensor) else np.asarray(x)


def setup_cartpole_env(record_gif=False):
    if record_gif and GYM_VERSION == "gymnasium":
        env = gym.make('CartPole-v1', render_mode='rgb_array')
    else:
        if GYM_VERSION == "gymnasium":
            env = gym.make('CartPole-v1')
        else:
            env = gym.make('CartPole-v1')
    return env


def encode_observation(observation, width, channel_n, encoding="continuous"):
    """Improved spatial encoding with multiple encoding options."""
    x = np.zeros((1, width, channel_n), dtype=np.float32)
    
    normalized_obs = np.array([
        np.clip(observation[0] / 4.8, -1, 1),
        np.clip(observation[1] / 2.0, -1, 1),
        np.clip(observation[2] / 0.418, -1, 1),
        np.clip(observation[3] / 3.0, -1, 1)
    ])
    
    if encoding == "thermometer":
        n_bits = 20
        starts = [5, 30, 55, 80]
        for i, obs_val in enumerate(normalized_obs):
            mapped = (obs_val + 1.0) / 2.0
            n_active = int(round(mapped * n_bits))
            for b in range(n_bits):
                pos = starts[i] + b
                if pos < width:
                    x[0, pos, 0] = 1.0 if b < n_active else 0.0
    elif encoding == "onehot":
        n_bits = 20
        starts = [5, 30, 55, 80]
        for i, obs_val in enumerate(normalized_obs):
            mapped = (obs_val + 1.0) / 2.0
            active_bin = int(round(mapped * (n_bits - 1)))
            active_bin = np.clip(active_bin, 0, n_bits - 1)
            pos = starts[i] + active_bin
            if pos < width:
                x[0, pos, 0] = 1.0
    else:  # continuous
        for i, obs_val in enumerate(normalized_obs):
            start_pos = 10 + i * 20
            for offset in range(-2, 3):
                pos = (start_pos + offset) % width
                weight = np.exp(-0.5 * offset**2)
                x[0, pos, 0] = obs_val * weight
    
    return x


def get_nca_features(nca, observation, width, timesteps, args=None, encoding="continuous"):
    """Process observation through NCA and extract features."""
    x = encode_observation(observation, width, nca.channel_n, encoding=encoding)

    # Keep data as TF tensor inside the NCA loop to avoid TF<->numpy copies.
    x = tf.constant(x)
    for t in range(timesteps):
        x = nca(x)
        if args is not None and hasattr(args, 'conserve') and args.conserve:
            x = apply_conservation(x, args)

    features = _to_numpy(x).flatten()
    return features


def get_action_from_policy(policy, features):
    """Get action from policy (works for both REINFORCE and Q-learning)."""
    if hasattr(policy, 'select_action'):
        # REINFORCE policy or Q-network with epsilon=0 (greedy)
        if hasattr(policy, 'q_values'):
            # Q-network: greedy action
            return policy.select_action(features, epsilon=0.0)
        else:
            # REINFORCE policy: sample from distribution
            return policy.select_action(features)
    else:
        raise ValueError("Unknown policy type")


def save_gif(frames, filename, fps=30):
    """Save frames as GIF file."""
    if not IMAGEIO_AVAILABLE:
        print("Warning: imageio not available, cannot save GIF")
        return
    
    if frames and len(frames) > 0:
        imageio.mimsave(filename, frames, fps=fps)
        print(f"GIF saved: {filename}")


def evaluate_episode(env, nca, policy, width, timesteps, args=None, record_gif=False, encoding="continuous"):
    """Run single CartPole episode with NCA reservoir and trained policy."""
    if GYM_VERSION == "gymnasium":
        observation, info = env.reset()
    else:
        observation = env.reset()
    
    done = False
    total_reward = 0
    episode_length = 0
    frames = [] if record_gif else None
    
    while not done:
        if record_gif:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        # Get NCA features
        features = get_nca_features(nca, observation, width, timesteps, args, encoding=encoding)
        
        # Get action from policy
        action = get_action_from_policy(policy, features)
        
        # Step environment
        if GYM_VERSION == "gymnasium":
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        else:
            observation, reward, done, info = env.step(action)
        
        total_reward += reward
        episode_length += 1
    
    if record_gif:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
    
    return episode_length, total_reward, frames


def evaluate_nca_cartpole(args, policy, num_runs=10, timesteps=5, width=100,
                         record_best_gif=False, encoding="continuous", policy_tag="policy"):
    """Evaluate NCA with RL policy on CartPole."""
    nca = get_nca(args)
    
    if record_best_gif and GYM_VERSION == "gymnasium":
        env = gym.make('CartPole-v1', render_mode='rgb_array')
    else:
        env = setup_cartpole_env()
    
    print(f"\nEvaluating CartPole with RL policy ({num_runs} episodes, {timesteps} timesteps/action, {width} cells, {encoding} encoding)...")
    
    episode_lengths = []
    total_rewards = []
    best_episode_idx = -1
    best_episode_length = 0
    best_frames = None
    
    for i in range(num_runs):
        record_this = record_best_gif and IMAGEIO_AVAILABLE
        episode_length, total_reward, frames = evaluate_episode(
            env, nca, policy, width, timesteps, args, record_gif=record_this, encoding=encoding
        )
        episode_lengths.append(episode_length)
        total_rewards.append(total_reward)
        print(f"Episode {i+1}/{num_runs}: length={episode_length}, reward={total_reward:.1f}")
        
        if record_best_gif and episode_length > best_episode_length:
            best_episode_length = episode_length
            best_episode_idx = i
            best_frames = frames
    
    env.close()
    
    if record_best_gif and best_frames is not None:
        label = os.path.basename(args.log_dir)
        gif_filename = f"cartpole_rl_best_{label}_{encoding}_t{timesteps}_{policy_tag}.gif"
        save_gif(best_frames, gif_filename, fps=30)
        print(f"Best episode: #{best_episode_idx+1} with length {best_episode_length}")
    
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    results = {
        'episode_lengths': episode_lengths,
        'total_rewards': total_rewards,
        'mean_length': mean_length,
        'std_length': std_length,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'num_runs': num_runs,
        'timesteps': timesteps,
        'width': width,
        'encoding': encoding,
        'policy_tag': policy_tag
    }
    
    print(f"\nResults:")
    print(f"  Mean episode length: {mean_length:.1f} ± {std_length:.1f}")
    print(f"  Mean total reward:   {mean_reward:.1f} ± {std_reward:.1f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NCA on CartPole with RL policy")
    parser.add_argument("--logdir", required=True, help="Path to NCA checkpoint directory")
    parser.add_argument("--policy", required=True, help="Path to trained policy/Q-network (.pkl)")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of episodes (default: 10)")
    parser.add_argument("--timesteps", type=int, default=5, help="NCA timesteps per action (default: 5)")
    parser.add_argument("--width", type=int, default=100, help="Reservoir width (default: 100)")
    parser.add_argument("--save_results", action="store_true", help="Save results to CSV")
    parser.add_argument("--record_gif", action="store_true", help="Record GIF of best episode")
    parser.add_argument("--encoding", default="continuous", choices=["continuous", "thermometer", "onehot"],
                        help="Input encoding: continuous, thermometer, or onehot (default: continuous)")
    p_args = parser.parse_args()

    if p_args.logdir and p_args.policy:
        # Load args from checkpoint
        args_filename = os.path.join(p_args.logdir, "args.json")
        argsio = utils.ArgsIO(args_filename)
        args = argsio.load_json()
        args.log_dir = p_args.logdir
        
        # Load policy
        policy = load_policy(p_args.policy)
        policy_tag = os.path.splitext(os.path.basename(p_args.policy))[0]
        
        # Run evaluation
        results = evaluate_nca_cartpole(
            args,
            policy,
            num_runs=p_args.num_runs,
            timesteps=p_args.timesteps,
            width=p_args.width,
            record_best_gif=p_args.record_gif,
            encoding=p_args.encoding,
            policy_tag=policy_tag
        )
        
        # Save results
        if p_args.save_results:
            import pandas as pd
            label = os.path.basename(p_args.logdir)
            results_df = pd.DataFrame({
                'episode': range(1, p_args.num_runs + 1),
                'length': results['episode_lengths'],
                'reward': results['total_rewards']
            })
            csv_filename = (
                f"cartpole_rl_results_{label}_{p_args.encoding}_t{p_args.timesteps}_{policy_tag}.csv"
            )
            results_df.to_csv(csv_filename, index=False)
            print(f"\nResults saved to: {csv_filename}")
    else:
        print("Add --logdir [path/to/checkpoint] and --policy [path/to/policy.pkl]")
