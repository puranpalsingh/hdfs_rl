"""
Reward function for the HDFS Replication Simulator.

The reward balances four competing objectives:
1. Storage cost — penalize over-replication (wasted disk)
2. Unavailability penalty — penalize files with fewer replicas than desired
3. Data loss penalty — severely penalize files with zero alive replicas
4. Access bonus — small positive reward for ensuring hot files are well-replicated

Design rationale:
- In fintech, data loss is catastrophically worse than wasted storage.
  The default weight ratio (50:5:1 for loss:unavailability:storage) reflects
  real operational priorities where a single data-loss event can trigger
  regulatory consequences.
- All components are normalized to [0, 1] so the weight ratios remain
  meaningful regardless of cluster size (3 nodes or 3000 nodes).
- The reward function is deliberately isolated from the environment so it
  can be unit-tested and tuned independently.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hdfs_sim.env.cluster import Cluster


@dataclasses.dataclass
class RewardConfig:
    """Configurable weights for the composite reward function.

    Attributes:
        alpha: Weight for storage cost penalty. Higher = agent is more
               frugal with replicas.
        beta: Weight for unavailability penalty. Higher = agent prioritizes
              maintaining desired replication factor.
        gamma: Weight for data loss penalty. Should be much larger than
               alpha and beta — data loss is catastrophic.
        delta: Weight for access bonus. Small positive incentive for
               ensuring hot files have adequate replication.
        access_threshold: Minimum alive replicas for a file to count as
                          "well-replicated" for the access bonus.
    """

    alpha: float = 1.0    # Storage cost weight
    beta: float = 5.0     # Unavailability weight
    gamma: float = 50.0   # Data loss weight
    delta: float = 0.5    # Access bonus weight
    access_threshold: int = 2  # Min replicas for access bonus


@dataclasses.dataclass
class RewardBreakdown:
    """Detailed breakdown of the reward for logging and debugging.

    Having this as a structured object makes it easy to log each component
    to TensorBoard or Prometheus during training and benchmarking.
    """

    storage_cost: float
    unavailability_penalty: float
    data_loss_penalty: float
    access_bonus: float
    total_reward: float


def compute_reward(
    cluster: Cluster,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Compute the composite reward for the current cluster state.

    Args:
        cluster: The current cluster state after actions have been applied.
        config: Reward weights. Uses defaults if None.

    Returns:
        RewardBreakdown with individual components and total.
    """
    if config is None:
        config = RewardConfig()

    num_files = max(1, cluster.num_files)  # Avoid division by zero

    # ---- 1. Storage cost: penalize over-replication ----
    # Ratio of total alive replicas to the theoretical maximum
    # (every file at max replication). Ranges from 0 to 1.
    total_alive = cluster.total_alive_replicas()
    max_possible = num_files * cluster.max_replication
    storage_cost = total_alive / max(1, max_possible)

    # ---- 2. Unavailability penalty: files with fewer replicas than desired ----
    # Fraction of files that are under-replicated. Ranges from 0 to 1.
    under_replicated = cluster.files_under_replicated()
    unavailability_penalty = len(under_replicated) / num_files

    # ---- 3. Data loss penalty: files with ZERO alive replicas ----
    # This is the catastrophic failure case. Ranges from 0 to 1.
    lost_files = cluster.files_with_zero_replicas()
    data_loss_penalty = len(lost_files) / num_files

    # ---- 4. Access bonus: hot files with adequate replication ----
    # Positive reward for files where access_frequency > 0 AND
    # alive_replicas >= access_threshold. Weighted by access frequency
    # so protecting hotter files matters more.
    access_scores = []
    for file in cluster.files.values():
        alive = cluster.count_alive_replicas_for_file(file.file_id)
        if alive >= config.access_threshold:
            access_scores.append(file.access_frequency)
        else:
            access_scores.append(0.0)

    access_bonus = float(np.mean(access_scores)) if access_scores else 0.0

    # ---- Composite reward ----
    raw_total = (
        -config.alpha * storage_cost
        - config.beta * unavailability_penalty
        - config.gamma * data_loss_penalty
        + config.delta * access_bonus
    )

    # Normalise to [-1, ~0] per step so PPO's value network can fit the
    # return scale without special initialisation.
    # Max penalty per step = alpha + beta + gamma (all components at worst = 1).
    # Dividing by this keeps weight ratios intact while bringing episode
    # returns from ~[-5600, 0] down to ~[-100, 0] (100 steps x [-1, 0]).
    max_penalty = config.alpha + config.beta + config.gamma
    total = raw_total / max(1.0, max_penalty)

    return RewardBreakdown(
        storage_cost=storage_cost,
        unavailability_penalty=unavailability_penalty,
        data_loss_penalty=data_loss_penalty,
        access_bonus=access_bonus,
        total_reward=total,
    )
