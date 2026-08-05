"""Tests for the reward function.

These tests verify that the reward function correctly balances the four
competing objectives and produces sensible values at edge cases:
- All files lost → minimum reward (dominated by data_loss_penalty)
- Perfect replication → reasonable reward (low storage cost, no penalties)
- Over-replication → storage cost increases
"""

import numpy as np
import pytest

from hdfs_sim.env.cluster import Cluster
from hdfs_sim.env.reward import RewardConfig, compute_reward


@pytest.fixture
def cluster():
    """A fresh cluster with deterministic randomness."""
    c = Cluster(
        num_racks=3, nodes_per_rack=4, num_files=10,
        file_size_range=(2.0, 5.0), max_replication=5,
        rng=np.random.default_rng(42),
    )
    c.reset()
    return c


class TestRewardBasicProperties:
    """Test fundamental properties of the reward function."""

    def test_reward_returns_breakdown(self, cluster):
        actions = np.full(cluster.num_files, 3)
        cluster.apply_replication_actions(actions)
        result = compute_reward(cluster)
        assert hasattr(result, "total_reward")
        assert hasattr(result, "storage_cost")
        assert hasattr(result, "unavailability_penalty")
        assert hasattr(result, "data_loss_penalty")
        assert hasattr(result, "access_bonus")

    def test_components_are_normalized(self, cluster):
        """Each component should be in [0, 1]."""
        actions = np.full(cluster.num_files, 3)
        cluster.apply_replication_actions(actions)
        result = compute_reward(cluster)
        assert 0.0 <= result.storage_cost <= 1.0
        assert 0.0 <= result.unavailability_penalty <= 1.0
        assert 0.0 <= result.data_loss_penalty <= 1.0
        assert 0.0 <= result.access_bonus <= 1.0

    def test_reward_is_finite(self, cluster):
        actions = np.full(cluster.num_files, 3)
        cluster.apply_replication_actions(actions)
        result = compute_reward(cluster)
        assert np.isfinite(result.total_reward)


class TestRewardEdgeCases:
    """Test reward at extreme cluster states."""

    def test_all_files_lost_has_worst_reward(self, cluster):
        """When all files have 0 replicas, data_loss_penalty dominates."""
        # Don't place any replicas → all files lost
        result_no_replicas = compute_reward(cluster)

        # Place replicas
        actions = np.full(cluster.num_files, 3)
        cluster.apply_replication_actions(actions)
        result_with_replicas = compute_reward(cluster)

        assert result_no_replicas.total_reward < result_with_replicas.total_reward
        assert result_no_replicas.data_loss_penalty == 1.0  # All files lost

    def test_no_data_loss_with_replicas(self, cluster):
        """With replicas and all nodes alive, data_loss_penalty should be 0."""
        actions = np.full(cluster.num_files, 3)
        cluster.apply_replication_actions(actions)
        result = compute_reward(cluster)
        assert result.data_loss_penalty == 0.0

    def test_over_replication_increases_storage_cost(self, cluster):
        """More replicas → higher storage cost component."""
        actions_low = np.full(cluster.num_files, 2)
        cluster.apply_replication_actions(actions_low)
        result_low = compute_reward(cluster)

        # Reset and over-replicate
        cluster.reset()
        actions_high = np.full(cluster.num_files, 5)
        cluster.apply_replication_actions(actions_high)
        result_high = compute_reward(cluster)

        assert result_high.storage_cost > result_low.storage_cost

    def test_single_replica_is_unavailable_risk(self, cluster):
        """Files with rep=1 are at risk; if desired is higher, they're under-replicated."""
        actions = np.full(cluster.num_files, 1)
        cluster.apply_replication_actions(actions)
        # The files have 1 replica but desired=1, so they're not under-replicated
        result = compute_reward(cluster)
        assert result.unavailability_penalty == 0.0  # desired == actual

    def test_unavailability_when_under_replicated(self, cluster):
        """If we set desired high but can't place, unavailability penalty > 0."""
        # Place only 1 replica per file
        actions = np.full(cluster.num_files, 1)
        cluster.apply_replication_actions(actions)

        # Now manually set desired_replication higher than actual
        for f in cluster.files.values():
            f.desired_replication = 4

        result = compute_reward(cluster)
        assert result.unavailability_penalty > 0.0


class TestRewardConfigWeights:
    """Test that config weights actually affect the reward."""

    def test_higher_gamma_amplifies_data_loss(self, cluster):
        """Higher data loss weight should produce more negative reward when files are lost."""
        # No replicas = total data loss
        config_low = RewardConfig(gamma=10.0)
        config_high = RewardConfig(gamma=100.0)

        result_low = compute_reward(cluster, config_low)
        result_high = compute_reward(cluster, config_high)

        assert result_high.total_reward < result_low.total_reward

    def test_zero_weights_eliminate_components(self, cluster):
        """Setting a weight to 0 should make that component not affect the total."""
        config = RewardConfig(alpha=0.0, beta=0.0, gamma=0.0, delta=0.0)
        result = compute_reward(cluster, config)
        assert result.total_reward == 0.0

    def test_access_bonus_with_replicated_hot_files(self, cluster):
        """Hot files with enough replicas should produce positive access bonus."""
        # Make all files hot
        for f in cluster.files.values():
            f.access_frequency = 0.9

        # Place enough replicas
        actions = np.full(cluster.num_files, 3)
        cluster.apply_replication_actions(actions)

        config = RewardConfig(access_threshold=2)
        result = compute_reward(cluster, config)
        assert result.access_bonus > 0.5  # Hot files are well-replicated
