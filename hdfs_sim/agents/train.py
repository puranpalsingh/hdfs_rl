import os
import argparse
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import get_linear_fn

from hdfs_sim.env.hdfs_env import HDFSReplicationEnv
from hdfs_sim.env.reward import RewardConfig
from hdfs_sim.env.failure import FailureModel


def _make_env_factory(cluster_cfg, episode_cfg, reward_config_obj, failure_model, log_dir=None, seed=None):
    """Return a callable that creates a single HDFSReplicationEnv instance.

    Using a factory (closure) is the standard pattern for DummyVecEnv / SubprocVecEnv
    so each parallel worker gets its own independent environment and RNG state.
    """
    def _init():
        env = HDFSReplicationEnv(
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
            seed=seed,
        )
        if log_dir is not None:
            env = Monitor(env, log_dir)
        return env
    return _init


def train(config_path: str, total_timesteps: int, output_dir: str, n_envs: int = 4):
    """Train a PPO agent on the HDFS Replication Simulator.

    Key improvements over v1:
    - n_envs parallel training environments -> more diverse rollout data per update.
    - Separate, independent eval environment for EvalCallback (fixes the v1 bug
      where EvalCallback shared the training env's RNG state, causing the best_model
      to overfit to specific seeds and perform worse on held-out evaluation).
    - VecNormalize on both train and eval envs -> tames the high-variance reward
      signal (gamma=50 data-loss penalty), stabilising PPO gradient estimates.
    - Higher entropy coefficient -> prevents premature convergence in the large
      MultiDiscrete action space (6^20 combinations).

    Args:
        config_path: Path to the cluster configuration YAML file.
        total_timesteps: Number of timesteps to train for.
        output_dir: Directory to save the trained model and logs.
        n_envs: Number of parallel training environments.
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

    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Build training VecEnv
    # Each parallel env gets its own FailureModel (independent RNG) and
    # a Monitor wrapper so SB3 can correctly track episode rewards/lengths.
    # We intentionally skip VecNormalize: the reward function in reward.py
    # is already normalised to [0,1] components (the large raw values seen
    # in ep_rew_mean are the sum over 100 steps, not per-step).  Applying
    # VecNormalize on top flattened the reward gradient and caused clip_
    # fraction to stick at ~0.57 across all iterations (PPO couldn't
    # distinguish good from bad actions in the normalised space).
    # ------------------------------------------------------------------ #
    def _make_failure_model():
        return FailureModel(**failure_cfg)

    train_env_fns = [
        _make_env_factory(cluster_cfg, episode_cfg, reward_config_obj,
                          _make_failure_model(), log_dir=log_dir, seed=i)
        for i in range(n_envs)
    ]
    train_env = DummyVecEnv(train_env_fns)

    # ------------------------------------------------------------------ #
    # Build a SEPARATE eval VecEnv  <- the key fix vs. v1
    # Uses a Monitor wrapper and a fixed seed so EvalCallback always
    # evaluates on the same distribution, independent of training RNG.
    # ------------------------------------------------------------------ #
    eval_log_dir = os.path.join(output_dir, "eval_monitor")
    os.makedirs(eval_log_dir, exist_ok=True)
    eval_env = DummyVecEnv([
        _make_env_factory(cluster_cfg, episode_cfg, reward_config_obj,
                          _make_failure_model(), log_dir=eval_log_dir, seed=42)
    ])

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // n_envs, 1),  # every ~10k env-steps
        save_path=os.path.join(output_dir, "checkpoints"),
        name_prefix="hdfs_ppo_model",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(output_dir, "best_model"),
        log_path=os.path.join(output_dir, "eval_logs"),
        eval_freq=max(10_000 // n_envs, 1),  # every ~10k env-steps
        n_eval_episodes=20,                   # more episodes -> lower variance estimate
        deterministic=True,
        render=False,
    )

    # ------------------------------------------------------------------ #
    # PPO hyperparameters (tuned for this environment)
    # ------------------------------------------------------------------ #
    # n_steps=2048 per env x n_envs=4 -> 8192 transitions per update.
    # This gives ~80 full episodes per update (each ep=100 steps),
    # enough for the gradient to average out the data-loss reward spikes.
    #
    # clip_range=0.1: tighter than the default 0.2 — the v1 run and the
    # first phase3 attempt both showed clip_fraction stuck at ~0.57, meaning
    # the policy was taking steps that were too large every iteration.
    # Halving clip_range forces smaller, more stable policy updates.
    #
    # learning_rate: linear decay from 1e-4 to 0 gives larger steps early
    # (exploration) and fine-grained adjustments near convergence.
    #
    # ent_coef=0.01: entropy bonus prevents the policy from collapsing to
    # "always replicate=3" across all 20 files without exploring.
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=os.path.join(output_dir, "tensorboard"),
        device="cpu",
        # Rollout
        n_steps=2048,
        # Optimisation
        batch_size=256,
        n_epochs=5,
        learning_rate=get_linear_fn(1e-4, 1e-6, 1.0),  # 1e-4 -> 1e-6 over training
        # Value function & advantage
        gamma=0.99,
        gae_lambda=0.95,
        # Clipping — tighter than default to curb the high clip_fraction
        clip_range=0.1,
        clip_range_vf=0.1,
        # Exploration
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        # Policy network: wider to handle the large obs/action space
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    print(f"\nStarting training for {total_timesteps} timesteps "
          f"across {n_envs} parallel environments...")
    print(f"  Effective steps per update: {n_envs * 1024}")
    print(f"  Eval env: independent (separate RNG) -- 20 episodes per eval\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=False,
    )

    # ------------------------------------------------------------------ #
    # Save final model
    # ------------------------------------------------------------------ #
    final_model_path = os.path.join(output_dir, "final_model")
    model.save(final_model_path)
    print(f"\nModel saved -> {final_model_path}.zip")

    train_env.close()
    eval_env.close()
    print("\nTraining complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for HDFS replication")
    parser.add_argument("--config", type=str, default="configs/cluster_config.yaml",
                        help="Path to cluster config file")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps (default: 500k)")
    parser.add_argument("--output", type=str, default="ppo_agent_phase3",
                        help="Output directory for models and logs")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="Number of parallel training environments (default: 4)")

    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            config_path,
        )

    train(config_path, args.timesteps, args.output, args.n_envs)
