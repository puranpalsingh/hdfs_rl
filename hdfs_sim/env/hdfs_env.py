"""
Gymnasium environment for HDFS replication factor optimization.

This is the core RL environment. An agent observes the cluster state
(node health, load, rack topology, file replication, access patterns)
and decides a target replication factor for each file. The environment
applies the actions using rack-aware placement, simulates time passing
(with optional failures in Phase 2), and returns a reward that balances
storage cost against data availability.

Design rationale:
- MultiDiscrete action space: each file gets an independent replication
  decision (0 to max_replication). This mirrors how a real HDFS NameNode
  works — it decides "how many replicas" and a separate placement policy
  decides "where." We don't ask the RL agent to do placement.
- Flat Box observation: SB3's MlpPolicy works best with flat vectors.
  Dict/Tuple spaces have compatibility issues and add complexity without
  clear benefit for this problem size.
- Episode = 100 timesteps: long enough for multiple failure/recovery cycles,
  short enough for fast training.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hdfs_sim.env.cluster import Cluster
from hdfs_sim.env.reward import RewardConfig, compute_reward


class HDFSReplicationEnv(gym.Env):
    """Gymnasium environment simulating an HDFS-like replication cluster.

    Observation Space:
        Box of shape (3*num_nodes + 4*num_files,) with values in [0, 1].
        Per node: [health, load_fraction, rack_id_normalized]
        Per file: [alive_replicas_norm, access_frequency, size_norm, desired_rep_norm]

    Action Space:
        MultiDiscrete([max_replication+1] * num_files).
        Each element is the target replication factor for that file.

    Reward:
        Composite reward balancing storage cost, unavailability, data loss,
        and access frequency coverage. See reward.py for details.

    Episode termination:
        After max_steps timesteps (default 100). No early termination —
        the agent should learn to maintain the cluster across the full episode.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        num_racks: int = 3,
        nodes_per_rack: int = 4,
        node_capacity_gb: float = 100.0,
        num_files: int = 20,
        file_size_range: tuple[float, float] = (1.0, 10.0),
        max_replication: int = 5,
        max_steps: int = 100,
        time_step_hours: float = 1.0,
        reward_config: Optional[RewardConfig] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        failure_model: Any = None,
    ):
        """Initialize the HDFS replication environment.

        Args:
            num_racks: Number of racks in the cluster.
            nodes_per_rack: Number of nodes per rack.
            node_capacity_gb: Storage capacity per node in GB.
            num_files: Number of files in the cluster.
            file_size_range: (min, max) file size in GB.
            max_replication: Maximum allowed replication factor.
            max_steps: Number of timesteps per episode.
            time_step_hours: Simulated hours per timestep.
            reward_config: Reward function weights.
            render_mode: Gymnasium render mode.
            seed: Random seed for reproducibility.
            failure_model: Optional failure model (Phase 2 integration).
        """
        super().__init__()

        self.num_racks = num_racks
        self.nodes_per_rack = nodes_per_rack
        self.num_nodes = num_racks * nodes_per_rack
        self.num_files = num_files
        self.max_replication = max_replication
        self.max_steps = max_steps
        self.time_step_hours = time_step_hours
        self.reward_config = reward_config or RewardConfig()
        self.render_mode = render_mode
        self.failure_model = failure_model

        # Create the cluster
        self._init_rng = np.random.default_rng(seed)
        self.cluster = Cluster(
            num_racks=num_racks,
            nodes_per_rack=nodes_per_rack,
            node_capacity_gb=node_capacity_gb,
            num_files=num_files,
            file_size_range=file_size_range,
            max_replication=max_replication,
            rng=np.random.default_rng(self._init_rng.integers(0, 2**31)),
        )

        # ---- Define spaces ----
        obs_dim = self.cluster.observation_dim
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Each file gets a replication decision.
        # To prevent catastrophic data loss traps during exploration, we restrict 
        # the minimum RF to 2. The RL agent chooses 0 to max_replication-2.
        # This is mapped to RF = action + 2.
        # e.g. for max_replication=5, choices are 0,1,2,3 -> RF 2,3,4,5.
        self.action_space = spaces.MultiDiscrete(
            [max_replication - 1] * num_files
        )

        # Episode tracking
        self.current_step = 0
        self.simulated_time_hours = 0.0
        self._episode_reward_history: list[float] = []

        # Store cluster config for re-creation on reset
        self._cluster_kwargs = {
            "num_racks": num_racks,
            "nodes_per_rack": nodes_per_rack,
            "node_capacity_gb": node_capacity_gb,
            "num_files": num_files,
            "file_size_range": file_size_range,
            "max_replication": max_replication,
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to a clean initial state.

        Args:
            seed: Random seed for this episode.
            options: Additional options (unused).

        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed)

        # Create a new RNG from the seed Gymnasium provides
        if seed is not None:
            self._init_rng = np.random.default_rng(seed)

        episode_rng = np.random.default_rng(
            self._init_rng.integers(0, 2**31)
        )

        # Reset the cluster with fresh randomness
        self.cluster.rng = episode_rng
        self.cluster.reset(rng=episode_rng)

        # Reset failure model if present
        if self.failure_model is not None:
            self.failure_model.reset(rng=np.random.default_rng(
                episode_rng.integers(0, 2**31)
            ))

        # Reset episode tracking
        self.current_step = 0
        self.simulated_time_hours = 0.0
        self._episode_reward_history = []

        # Initial placement: start with default replication=3 for all files
        # This gives the agent a baseline to improve upon
        initial_actions = np.full(self.num_files, 3, dtype=np.int64)
        self.cluster.apply_replication_actions(initial_actions)

        obs = self.cluster.get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one timestep of the environment.

        Flow:
        1. Apply failure/recovery events (Phase 2; no-op in Phase 1).
        2. Optionally shift access patterns (small random walk).
        3. Apply the agent's replication decisions.
        4. Compute reward.
        5. Check termination.

        Args:
            action: Array of shape (num_files,) with target replication
                    factors per file.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        self.current_step += 1
        self.simulated_time_hours += self.time_step_hours

        # Step 1: Apply failures (Phase 2 integration point)
        failure_events = []
        if self.failure_model is not None:
            failure_events = self.failure_model.step(
                self.cluster, self.time_step_hours
            )

        # Step 2: Small random walk on access frequencies
        self._drift_access_patterns()

        # Step 3: Clean up dead-node replicas from file replica sets,
        # then apply the agent's replication decisions
        self._clean_dead_replicas()
        
        # Map agent action [0, 1, 2, 3] to actual RF [2, 3, 4, 5]
        actual_action = action + 2
        placement_stats = self.cluster.apply_replication_actions(actual_action)

        # Step 4: Compute reward
        reward_breakdown = compute_reward(self.cluster, self.reward_config)
        reward = float(reward_breakdown.total_reward)
        self._episode_reward_history.append(reward)

        # Step 5: Termination
        terminated = False  # No early termination by design
        truncated = self.current_step >= self.max_steps

        obs = self.cluster.get_observation()
        info = self._get_info()
        info["reward_breakdown"] = dataclasses.asdict(reward_breakdown)  # type: ignore[name-defined]
        info["placement_stats"] = placement_stats
        info["failure_events"] = len(failure_events)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """Render the current cluster state."""
        summary = self.cluster.get_cluster_summary()
        output_lines = [
            f"\n{'='*60}",
            f"  HDFS Replication Simulator — Step {self.current_step}/{self.max_steps}",
            f"  Simulated time: {self.simulated_time_hours:.1f} hours",
            f"{'='*60}",
            f"  Nodes: {summary['alive_nodes']}/{summary['total_nodes']} alive",
            f"  Storage: {summary['total_storage_used_gb']:.1f} / {summary['total_capacity_gb']:.1f} GB",
            f"  Files: {summary['total_files']} total",
            f"    - Data loss (0 replicas): {summary['files_with_data_loss']}",
            f"    - Under-replicated:       {summary['files_under_replicated']}",
            f"  Total alive replicas: {summary['total_alive_replicas']}",
        ]

        if self._episode_reward_history:
            output_lines.append(
                f"  Last reward: {self._episode_reward_history[-1]:.4f}"
            )
            output_lines.append(
                f"  Avg reward:  {np.mean(self._episode_reward_history):.4f}"
            )

        output_lines.append(f"{'='*60}\n")
        output = "\n".join(output_lines)

        if self.render_mode == "human":
            print(output)
        return output

    def _drift_access_patterns(self) -> None:
        """Apply a small random walk to file access frequencies.

        This simulates real-world access pattern shifts — some files
        become hotter, some cooler. The drift is small (σ=0.02 per step)
        so patterns change slowly, as they do in practice.
        """
        for file in self.cluster.files.values():
            drift = self.cluster.rng.normal(0, 0.02)
            file.access_frequency = float(
                np.clip(file.access_frequency + drift, 0.0, 1.0)
            )

    def _clean_dead_replicas(self) -> None:
        """Remove references to replicas on dead nodes.

        When a node dies, its data is gone. We remove the node from
        each file's replica set so the alive_replicas count is accurate.
        This mirrors HDFS's behavior where the NameNode detects missing
        block reports from dead DataNodes.
        """
        dead_node_ids = {
            nid for nid, node in self.cluster.nodes.items() if not node.is_alive
        }
        if not dead_node_ids:
            return

        for file in self.cluster.files.values():
            lost = file.replica_node_ids & dead_node_ids
            file.replica_node_ids -= dead_node_ids
            # The storage is "freed" on the dead node (it's gone)
            for nid in lost:
                self.cluster.nodes[nid].used_gb = max(
                    0.0, self.cluster.nodes[nid].used_gb - file.size_gb
                )

    def _get_info(self) -> dict:
        """Return auxiliary information for debugging."""
        return {
            "step": self.current_step,
            "simulated_time_hours": self.simulated_time_hours,
            "cluster_summary": self.cluster.get_cluster_summary(),
        }


# Need this import for dataclasses.asdict in step()
import dataclasses  # noqa: E402
