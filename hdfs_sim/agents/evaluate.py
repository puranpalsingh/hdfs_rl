import argparse
import yaml
import numpy as np
import os
from pathlib import Path

from stable_baselines3 import PPO

from hdfs_sim.env.hdfs_env import HDFSReplicationEnv
from hdfs_sim.env.reward import RewardConfig
from hdfs_sim.env.failure import FailureModel
from hdfs_sim.baselines.static_baseline import StaticBaseline


def _build_env(cluster_cfg, episode_cfg, reward_config_obj, failure_model):
    """Create a plain HDFSReplicationEnv."""
    return HDFSReplicationEnv(
        num_racks=cluster_cfg.get("num_racks", 3),
        nodes_per_rack=cluster_cfg.get("nodes_per_rack", 4),
        node_capacity_gb=cluster_cfg.get("node_capacity_gb", 100.0),
        num_files=cluster_cfg.get("num_files", 20),
        file_size_range=tuple(cluster_cfg.get("file_size_range", [1.0, 10.0])),
        max_replication=cluster_cfg.get("max_replication", 5),
        max_steps=episode_cfg.get("max_steps", 100),
        time_step_hours=episode_cfg.get("time_step_hours", 1.0),
        reward_config=reward_config_obj,
        failure_model=failure_model,
    )


from collections import defaultdict

def _run_episodes(agent, env, num_episodes: int) -> dict:
    """Run evaluation episodes using the standard gymnasium step() API."""
    rewards = []
    data_loss_counts = []
    rf_counts = defaultdict(int)
    total_actions = 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        max_data_loss = 0
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            
            # Record RF distribution (action is [0, 3], actual RF is [2, 5])
            for a in action:
                rf_counts[int(a) + 2] += 1
                total_actions += 1

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Track max data loss in the episode
            summary = info.get("cluster_summary", {})
            current_loss = summary.get("files_with_data_loss", 0)
            if current_loss > max_data_loss:
                max_data_loss = current_loss

        rewards.append(total_reward)
        data_loss_counts.append(max_data_loss)
        
    return {
        "rewards": rewards,
        "data_loss": data_loss_counts,
        "rf_dist": {k: (v / max(1, total_actions)) * 100 for k, v in rf_counts.items()}
    }


def _print_results(name: str, stats: dict) -> None:
    rewards = stats["rewards"]
    mean = np.mean(rewards)
    std  = np.std(rewards)
    mn   = np.min(rewards)
    mx   = np.max(rewards)
    print(f"{name} -> Mean: {mean:.2f} +/- {std:.2f}  (min: {mn:.2f}, max: {mx:.2f})")
    
    data_loss = stats["data_loss"]
    print(f"  Avg Data Loss (files): {np.mean(data_loss):.2f} +/- {np.std(data_loss):.2f}")
    
    rf_dist = stats["rf_dist"]
    print(f"  RF Distribution:")
    for rf in sorted(rf_dist.keys()):
        print(f"    RF={rf}: {rf_dist[rf]:.1f}%")



def evaluate(
    model_path: str,
    config_path: str,
    num_episodes: int,
    baseline_rep_factor: int = 3,
):
    """Evaluate a trained PPO agent against a static baseline.

    Args:
        model_path: Path to the trained model (.zip).
        config_path: Path to cluster config.
        num_episodes: Number of episodes to run for evaluation.
        baseline_rep_factor: Replication factor for the static baseline.
    """
    # ------------------------------------------------------------------ #
    # Load configuration
    # ------------------------------------------------------------------ #
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    cluster_cfg = config.get("cluster", {})
    episode_cfg = config.get("episode", {})
    reward_cfg  = config.get("reward", {})
    failure_cfg = config.get("failure", {})

    reward_config_obj = RewardConfig(**reward_cfg)

    # ------------------------------------------------------------------ #
    # Load PPO model
    # ------------------------------------------------------------------ #
    try:
        model = PPO.load(model_path, device="cpu")
        print(f"Loaded RL model from {model_path}")
    except Exception as e:
        print(f"Could not load RL model: {e}")
        return

    # ------------------------------------------------------------------ #
    # Evaluate Static Baseline
    # ------------------------------------------------------------------ #
    env = _build_env(cluster_cfg, episode_cfg, reward_config_obj, FailureModel(**failure_cfg))
    baseline = StaticBaseline(num_files=env.num_files, replication_factor=baseline_rep_factor)

    print(f"\nEvaluating Static Baseline (rep={baseline_rep_factor})...")
    baseline_rewards = _run_episodes(baseline, env, num_episodes)
    _print_results("Static Baseline", baseline_rewards)

    # ------------------------------------------------------------------ #
    # Evaluate PPO Agent
    # ------------------------------------------------------------------ #
    print(f"\nEvaluating PPO Agent...")
    env2 = _build_env(cluster_cfg, episode_cfg, reward_config_obj, FailureModel(**failure_cfg))
    ppo_rewards = _run_episodes(model, env2, num_episodes)
    _print_results("PPO Agent", ppo_rewards)

    # ------------------------------------------------------------------ #
    # Summary comparison
    # ------------------------------------------------------------------ #
    b_mean = np.mean(baseline_rewards["rewards"])
    p_mean = np.mean(ppo_rewards["rewards"])
    delta  = p_mean - b_mean
    print(f"\n{'='*50}")
    print(f"  Delta (PPO - Baseline): {delta:+.2f}")
    if delta > 0:
        print(f"  ✓ PPO outperforms baseline by {abs(delta):.2f} points")
    else:
        print(f"  ✗ Baseline still leads by {abs(delta):.2f} points")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL agent vs Baseline")
    parser.add_argument("--model",        type=str, required=True,
                        help="Path to trained PPO model (.zip)")
    parser.add_argument("--config",       type=str, default="configs/cluster_config.yaml",
                        help="Path to config")
    parser.add_argument("--episodes",     type=int, default=20,
                        help="Number of episodes to evaluate (default: 20)")
    parser.add_argument("--baseline_rep", type=int, default=3,
                        help="Static baseline replication factor")

    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            config_path,
        )

    evaluate(args.model, config_path, args.episodes, args.baseline_rep)