"""
CartPole Training with Q-Learning.

Trains a linear Q-value approximator on NCA reservoir states.

Usage:
    python reservoir_cartpole_train_qlearning.py \
        --logdir logs/train_nca_conserve/20260222-130839 \
        --num_episodes 500 \
        --timesteps 5 \
        --label conserving_qlearning
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
import time

try:
    import gymnasium as gym
    GYM_VERSION = "gymnasium"
except ImportError:
    import gym
    GYM_VERSION = "gym"


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
        
    def select_action(self, reservoir_state, epsilon=0.1):
        """Epsilon-greedy action selection."""
        if np.random.random() < epsilon:
            return np.random.randint(len(self.W))
        else:
            q_vals = self.q_values(reservoir_state)
            return np.argmax(q_vals)
        
    def update(self, res_state, action, reward, next_res_state, done, gamma=0.99, lr=0.001):
        """Q-learning update."""
        # Current Q-value
        q_current = self.W[action] @ res_state
        
        # Target Q-value
        if done:
            q_target = reward
        else:
            q_next_max = np.max(self.W @ next_res_state)
            q_target = reward + gamma * q_next_max
        
        # TD error
        td_error = q_target - q_current
        
        # Update weights
        self.W[action] += lr * td_error * res_state


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
    print("Model weights loaded successfully.")

    return nca


def _maybe_numpy(x):
    return x.numpy() if isinstance(x, tf.Tensor) else x


def _to_numpy(x):
    """Convert to numpy array if TF tensor."""
    return x.numpy() if isinstance(x, tf.Tensor) else np.asarray(x)


def setup_cartpole_env():
    if GYM_VERSION == "gymnasium":
        env = gym.make('CartPole-v1')
    else:
        env = gym.make('CartPole-v1')
    return env


def encode_observation(observation, width, channel_n, encoding="continuous"):
    """Improved spatial encoding."""
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
    else:
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
    
    # Keep as TF tensor throughout the loop — avoid per-step TF↔numpy copies
    x = tf.constant(x)
    for t in range(timesteps):
        x = nca(x)
        if args is not None and hasattr(args, 'conserve') and args.conserve:
            x = apply_conservation(x, args)
    
    features = _to_numpy(x).flatten()
    return features


def train_qlearning(args, q_network, num_episodes=500, timesteps=5, width=100, 
                    gamma=0.99, lr=0.001, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                    encoding="continuous"):
    """Train Q-network using Q-learning."""
    nca = get_nca(args)
    env = setup_cartpole_env()
    
    print(f"\nTraining Q-Learning...")
    print(f"Episodes: {num_episodes}, Timesteps: {timesteps}, LR: {lr}, Gamma: {gamma}")
    print(f"Epsilon: {epsilon_start} → {epsilon_end} (decay: {epsilon_decay}), Encoding: {encoding}")
    
    episode_rewards = []
    episode_lengths = []
    epsilon = epsilon_start
    
    print("Starting training loop...")
    train_start = time.time()
    for episode in range(num_episodes):
        if GYM_VERSION == "gymnasium":
            observation, info = env.reset()
        else:
            observation = env.reset()
        
        done = False
        episode_reward = 0
        episode_length = 0
        
        # Get initial reservoir state
        res_state = get_nca_features(nca, observation, width, timesteps, args, encoding=encoding)
        
        while not done:
            # Select action
            action = q_network.select_action(res_state, epsilon=epsilon)
            
            # Step environment
            if GYM_VERSION == "gymnasium":
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            else:
                observation, reward, done, info = env.step(action)
            
            # Get next reservoir state
            next_res_state = get_nca_features(nca, observation, width, timesteps, args, encoding=encoding)
            
            # Update Q-network
            q_network.update(res_state, action, reward, next_res_state, done, gamma=gamma, lr=lr)
            
            res_state = next_res_state
            episode_reward += reward
            episode_length += 1
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Timing for first episode
        if episode == 0:
            ep1_time = time.time() - train_start
            est_total = ep1_time * num_episodes / 60
            print(f"Episode 1: {episode_length} steps, {ep1_time:.1f}s "
                  f"(est. total: {est_total:.0f} min)")
        
        if (episode + 1) % 50 == 0:
            recent_reward = np.mean(episode_rewards[-50:])
            recent_length = np.mean(episode_lengths[-50:])
            elapsed = time.time() - train_start
            eps_per_sec = (episode + 1) / elapsed
            remaining = (num_episodes - episode - 1) / eps_per_sec
            print(f"Episode {episode+1}/{num_episodes}: "
                  f"Avg reward: {recent_reward:.1f}, "
                  f"Avg length: {recent_length:.1f}, "
                  f"Eps: {epsilon:.3f}, "
                  f"Elapsed: {elapsed/60:.1f}min, "
                  f"ETA: {remaining/60:.1f}min")
    
    env.close()
    
    total_time = time.time() - train_start
    print(f"\nTraining complete! ({total_time/60:.1f} min)")
    print(f"Final 50 episodes avg reward: {np.mean(episode_rewards[-50:]):.1f}")
    print(f"Final 50 episodes avg length: {np.mean(episode_lengths[-50:]):.1f}")
    
    return q_network, episode_rewards, episode_lengths


if __name__ == "__main__":
    # Limit TF/BLAS threads to avoid contention on multi-core machines
    import os as _os
    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS']:
        _os.environ.setdefault(var, '1')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    parser = argparse.ArgumentParser(description="Train CartPole Q-network")
    parser.add_argument("--logdir", required=True, help="Path to NCA checkpoint directory")
    parser.add_argument("--num_episodes", type=int, default=500, help="Number of training episodes (default: 500)")
    parser.add_argument("--timesteps", type=int, default=5, help="NCA timesteps per action (default: 5)")
    parser.add_argument("--width", type=int, default=100, help="Reservoir width (default: 100)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial epsilon (default: 1.0)")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Final epsilon (default: 0.01)")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Epsilon decay (default: 0.995)")
    parser.add_argument("--label", required=True, help="Label for output files")
    parser.add_argument("--encoding", default="continuous", choices=["continuous", "thermometer", "onehot"],
                        help="Input encoding: continuous (original), thermometer (binary fill), or onehot (single bin)")
    p_args = parser.parse_args()
    
    if p_args.logdir:
        # Load args from checkpoint
        args_filename = os.path.join(p_args.logdir, "args.json")
        argsio = utils.ArgsIO(args_filename)
        args = argsio.load_json()
        args.log_dir = p_args.logdir
        
        # Initialize Q-network
        reservoir_size = p_args.width * 5  # width * channels
        q_network = LinearQReadout(reservoir_size, n_actions=2)
        
        # Train
        q_network, rewards, lengths = train_qlearning(
            args, q_network,
            num_episodes=p_args.num_episodes,
            timesteps=p_args.timesteps,
            width=p_args.width,
            gamma=p_args.gamma,
            lr=p_args.lr,
            epsilon_start=p_args.epsilon_start,
            epsilon_end=p_args.epsilon_end,
            epsilon_decay=p_args.epsilon_decay,
            encoding=p_args.encoding
        )
        
        # Save Q-network
        qnet_filename = f"cartpole_qnetwork_{p_args.label}.pkl"
        with open(qnet_filename, 'wb') as f:
            pickle.dump(q_network, f)
        print(f"\nQ-network saved to: {qnet_filename}")
        
        # Save training history
        history_filename = f"cartpole_training_{p_args.label}.npz"
        np.savez(history_filename, rewards=rewards, lengths=lengths)
        print(f"Training history saved to: {history_filename}")
    else:
        print("Add --logdir [path/to/checkpoint]")
