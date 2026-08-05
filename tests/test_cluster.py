"""Tests for the Cluster data models and rack-aware placement logic.

These tests verify the core invariants that make the simulation realistic:
1. Rack-aware placement: replicas spread across racks.
2. No same-node duplicates: at most one replica of a file per node.
3. Capacity constraints: replicas aren't placed on full nodes.
4. Observation vector: correct shape and value ranges.
"""

import numpy as np
import pytest

from hdfs_sim.env.cluster import Cluster, File, Node, Rack


class TestTopology:
    """Test that cluster topology is built correctly."""

    def test_default_topology_sizes(self):
        cluster = Cluster(num_racks=3, nodes_per_rack=4)
        assert len(cluster.racks) == 3
        assert len(cluster.nodes) == 12
        for rack in cluster.racks.values():
            assert len(rack.node_ids) == 4

    def test_node_ids_are_unique(self):
        cluster = Cluster(num_racks=3, nodes_per_rack=4)
        all_ids = list(cluster.nodes.keys())
        assert len(all_ids) == len(set(all_ids))

    def test_every_node_belongs_to_a_rack(self):
        cluster = Cluster(num_racks=3, nodes_per_rack=4)
        rack_nodes = set()
        for rack in cluster.racks.values():
            rack_nodes.update(rack.node_ids)
        assert rack_nodes == set(cluster.nodes.keys())

    def test_node_rack_id_matches_rack(self):
        cluster = Cluster(num_racks=3, nodes_per_rack=4)
        for rack in cluster.racks.values():
            for nid in rack.node_ids:
                assert cluster.nodes[nid].rack_id == rack.rack_id


class TestFileGeneration:
    """Test that files are generated with valid attributes."""

    def test_correct_number_of_files(self):
        cluster = Cluster(num_files=20, rng=np.random.default_rng(42))
        assert len(cluster.files) == 20

    def test_file_sizes_in_range(self):
        cluster = Cluster(
            file_size_range=(1.0, 10.0), num_files=50,
            rng=np.random.default_rng(42),
        )
        for f in cluster.files.values():
            assert 1.0 <= f.size_gb <= 10.0

    def test_access_frequencies_in_range(self):
        cluster = Cluster(num_files=50, rng=np.random.default_rng(42))
        for f in cluster.files.values():
            assert 0.0 <= f.access_frequency <= 1.0

    def test_files_start_with_no_replicas(self):
        cluster = Cluster(rng=np.random.default_rng(42))
        cluster.reset()
        for f in cluster.files.values():
            assert len(f.replica_node_ids) == 0


class TestRackAwarePlacement:
    """Test the HDFS-style rack-aware replica placement algorithm."""

    def test_single_replica_placed(self):
        cluster = Cluster(
            num_racks=3, nodes_per_rack=4, num_files=1,
            rng=np.random.default_rng(42),
        )
        cluster.reset()
        file = cluster.files[0]
        placed = cluster.place_replicas(file, target_replication=1)
        assert placed == 1
        assert len(file.replica_node_ids) == 1

    def test_two_replicas_cross_rack(self):
        """Second replica MUST be on a different rack from the first."""
        cluster = Cluster(
            num_racks=3, nodes_per_rack=4, num_files=1,
            rng=np.random.default_rng(42),
        )
        cluster.reset()
        file = cluster.files[0]
        cluster.place_replicas(file, target_replication=2)

        # Get rack IDs of the two replicas
        racks = {cluster.nodes[nid].rack_id for nid in file.replica_node_ids}
        assert len(racks) >= 2, "Two replicas should be on different racks"

    def test_three_replicas_placement(self):
        """Three replicas: at least 2 racks, no same-node duplicates."""
        cluster = Cluster(
            num_racks=3, nodes_per_rack=4, num_files=1,
            rng=np.random.default_rng(42),
        )
        cluster.reset()
        file = cluster.files[0]
        cluster.place_replicas(file, target_replication=3)

        assert len(file.replica_node_ids) == 3
        # All on distinct nodes
        assert len(file.replica_node_ids) == 3
        # At least 2 distinct racks
        racks = {cluster.nodes[nid].rack_id for nid in file.replica_node_ids}
        assert len(racks) >= 2

    def test_no_same_node_duplicates(self):
        """A file should never have two replicas on the same node."""
        cluster = Cluster(
            num_racks=3, nodes_per_rack=4, num_files=1,
            max_replication=5, rng=np.random.default_rng(42),
        )
        cluster.reset()
        file = cluster.files[0]
        cluster.place_replicas(file, target_replication=5)

        node_ids = list(file.replica_node_ids)
        assert len(node_ids) == len(set(node_ids))

    def test_capacity_constraint(self):
        """Replicas should not be placed on nodes without enough space."""
        cluster = Cluster(
            num_racks=2, nodes_per_rack=1, node_capacity_gb=5.0,
            num_files=1, file_size_range=(3.0, 3.0), max_replication=5,
            rng=np.random.default_rng(42),
        )
        cluster.reset()
        file = cluster.files[0]
        # Only 2 nodes, each with 5GB. File is 3GB.
        # First replica: node has 5-3=2GB left. Second: other node 5-3=2GB.
        # Third: no node has 3GB free → should fail.
        placed = cluster.place_replicas(file, target_replication=5)
        assert placed == 2  # Only 2 nodes available

    def test_max_replicas_capped_by_available_nodes(self):
        """Can't have more replicas than alive nodes with capacity."""
        cluster = Cluster(
            num_racks=1, nodes_per_rack=3, num_files=1,
            max_replication=5, rng=np.random.default_rng(42),
        )
        cluster.reset()
        file = cluster.files[0]
        placed = cluster.place_replicas(file, target_replication=5)
        assert placed <= 3  # Only 3 nodes total


class TestReplicaRemoval:
    """Test replica removal logic."""

    def test_remove_replicas_reduces_count(self):
        cluster = Cluster(
            num_racks=3, nodes_per_rack=4, num_files=1,
            rng=np.random.default_rng(42),
        )
        cluster.reset()
        file = cluster.files[0]
        cluster.place_replicas(file, target_replication=4)
        assert len(file.replica_node_ids) == 4

        removed = cluster.remove_replicas(file, target_replication=2)
        assert removed == 2
        assert len(file.replica_node_ids) == 2

    def test_remove_frees_storage(self):
        cluster = Cluster(
            num_racks=3, nodes_per_rack=4, num_files=1,
            file_size_range=(5.0, 5.0), rng=np.random.default_rng(42),
        )
        cluster.reset()
        file = cluster.files[0]
        cluster.place_replicas(file, target_replication=3)

        used_before = cluster.total_storage_used_gb()
        cluster.remove_replicas(file, target_replication=1)
        used_after = cluster.total_storage_used_gb()

        assert used_after < used_before
        assert abs(used_after - used_before + 2 * 5.0) < 0.01


class TestApplyActions:
    """Test the apply_replication_actions interface."""

    def test_apply_actions_creates_replicas(self):
        cluster = Cluster(
            num_racks=3, nodes_per_rack=4, num_files=5,
            rng=np.random.default_rng(42),
        )
        cluster.reset()

        actions = np.array([3, 2, 1, 4, 2])
        stats = cluster.apply_replication_actions(actions)
        assert stats["placed"] > 0

        for fid, target in enumerate(actions):
            alive = cluster.count_alive_replicas_for_file(fid)
            assert alive == target

    def test_apply_actions_clamps_to_max(self):
        cluster = Cluster(
            num_racks=3, nodes_per_rack=4, num_files=1,
            max_replication=5, rng=np.random.default_rng(42),
        )
        cluster.reset()

        # Action exceeds max replication — should be clamped
        actions = np.array([10])
        cluster.apply_replication_actions(actions)
        alive = cluster.count_alive_replicas_for_file(0)
        assert alive <= 5


class TestObservationVector:
    """Test observation vector shape and value ranges."""

    def test_observation_shape(self):
        cluster = Cluster(
            num_racks=3, nodes_per_rack=4, num_files=20,
            rng=np.random.default_rng(42),
        )
        obs = cluster.get_observation()
        expected_dim = 3 * 12 + 4 * 20  # 36 + 80 = 116
        assert obs.shape == (expected_dim,)

    def test_observation_values_in_range(self):
        cluster = Cluster(rng=np.random.default_rng(42))
        # Place some replicas so there's actual data
        actions = np.full(cluster.num_files, 3)
        cluster.apply_replication_actions(actions)

        obs = cluster.get_observation()
        assert np.all(obs >= 0.0), f"Min value: {obs.min()}"
        assert np.all(obs <= 1.0), f"Max value: {obs.max()}"

    def test_observation_dtype(self):
        cluster = Cluster(rng=np.random.default_rng(42))
        obs = cluster.get_observation()
        assert obs.dtype == np.float32

    def test_observation_dim_property(self):
        cluster = Cluster(num_racks=2, nodes_per_rack=3, num_files=10)
        assert cluster.observation_dim == 3 * 6 + 4 * 10  # 18 + 40 = 58


class TestClusterMetrics:
    """Test cluster-level metric calculations."""

    def test_no_data_loss_initially(self):
        cluster = Cluster(rng=np.random.default_rng(42))
        cluster.reset()
        actions = np.full(cluster.num_files, 3)
        cluster.apply_replication_actions(actions)
        assert len(cluster.files_with_zero_replicas()) == 0

    def test_data_loss_when_all_nodes_dead(self):
        cluster = Cluster(rng=np.random.default_rng(42))
        cluster.reset()
        actions = np.full(cluster.num_files, 1)
        cluster.apply_replication_actions(actions)

        # Kill all nodes
        for node in cluster.nodes.values():
            node.is_alive = False

        # All files should show zero alive replicas
        for fid in range(cluster.num_files):
            assert cluster.count_alive_replicas_for_file(fid) == 0

    def test_cluster_summary_keys(self):
        cluster = Cluster(rng=np.random.default_rng(42))
        summary = cluster.get_cluster_summary()
        expected_keys = {
            "total_nodes", "alive_nodes", "dead_nodes",
            "total_storage_used_gb", "total_capacity_gb",
            "total_files", "files_with_data_loss",
            "files_under_replicated", "total_alive_replicas",
        }
        assert set(summary.keys()) == expected_keys


class TestReset:
    """Test that reset produces a clean state."""

    def test_reset_clears_replicas(self):
        cluster = Cluster(rng=np.random.default_rng(42))
        actions = np.full(cluster.num_files, 3)
        cluster.apply_replication_actions(actions)
        assert cluster.total_storage_used_gb() > 0

        cluster.reset()
        # After reset, files should have no replicas
        for f in cluster.files.values():
            assert len(f.replica_node_ids) == 0

    def test_reset_revives_nodes(self):
        cluster = Cluster(rng=np.random.default_rng(42))
        cluster.nodes[0].is_alive = False
        cluster.nodes[1].is_alive = False
        cluster.reset()
        for node in cluster.nodes.values():
            assert node.is_alive is True
            assert node.health == 1.0
