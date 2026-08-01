"""
Cluster topology data models for the HDFS Replication Simulator.

Models a rack-aware storage cluster with Nodes grouped into Racks.
Files are stored with configurable replication across nodes, following
HDFS-style cross-rack placement constraints.

Design rationale:
- Racks are modeled explicitly because HDFS's core reliability mechanism is
  cross-rack replication. Without rack awareness, the RL agent can't learn
  that collocating all replicas on one rack is dangerous.
- Nodes have capacity and health attributes to let the agent reason about
  load balancing and failure risk.
- Files track their own replica locations so the reward function can quickly
  compute availability and storage cost.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass
class Node:
    """A single storage node in the cluster.

    Attributes:
        node_id: Unique identifier for this node.
        rack_id: Which rack this node belongs to.
        capacity_gb: Total storage capacity in gigabytes.
        used_gb: Currently used storage in gigabytes.
        is_alive: Whether the node is currently operational.
        health: Health score in [0, 1]. 1.0 = fully healthy.
                In Phase 1 this mirrors is_alive; Phase 2 replaces it
                with a continuous signal from the failure model.
    """

    node_id: int
    rack_id: int
    capacity_gb: float
    used_gb: float = 0.0
    is_alive: bool = True
    health: float = 1.0

    @property
    def free_gb(self) -> float:
        return max(0.0, self.capacity_gb - self.used_gb)

    @property
    def load_fraction(self) -> float:
        """Fraction of capacity in use, clamped to [0, 1]."""
        if self.capacity_gb <= 0:
            return 1.0
        return min(1.0, max(0.0, self.used_gb / self.capacity_gb))


@dataclasses.dataclass
class Rack:
    """A rack containing multiple storage nodes.

    In a real data center, all nodes on a rack share a top-of-rack (ToR)
    switch and often a power circuit. A rack failure takes out all its nodes.
    """

    rack_id: int
    node_ids: list[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class File:
    """A file stored in the simulated HDFS cluster.

    Attributes:
        file_id: Unique identifier.
        size_gb: Size of the file in gigabytes.
        access_frequency: Normalized access frequency in [0, 1].
                          Higher = more frequently read ("hotter").
        desired_replication: The target replication factor set by the agent.
        replica_node_ids: Set of node IDs currently holding a replica.
        is_permanently_lost: True if all replicas were destroyed simultaneously
                             and the file can never be recovered. This prevents
                             the simulation from "magically" creating new replicas
                             from a file that no longer exists anywhere.
    """

    file_id: int
    size_gb: float
    access_frequency: float
    desired_replication: int = 3
    replica_node_ids: set[int] = dataclasses.field(default_factory=set)
    is_permanently_lost: bool = False
    had_replicas_ever: bool = False  # True once at least one replica was placed

    @property
    def alive_replicas(self) -> int:
        """Number of replicas currently tracked (may include dead nodes)."""
        return len(self.replica_node_ids)


class Cluster:
    """Simulated HDFS-like storage cluster with rack-aware topology.

    This is the central data structure that the Gymnasium environment
    manipulates. It owns all nodes, racks, and files, and provides
    methods for:
    - Rack-aware replica placement (mirroring HDFS BlockPlacementPolicyDefault)
    - Replica removal when the agent reduces replication factor
    - Serialization to a flat numpy observation vector
    - Capacity accounting

    The placement algorithm follows HDFS conventions:
    1. First replica: random alive node with capacity.
    2. Second replica: random alive node on a DIFFERENT rack.
    3. Third replica: random alive node on the SAME rack as replica 2
       (but different node).
    4. Replicas 4+: spread across racks with fewest existing replicas.
    """

    def __init__(
        self,
        num_racks: int = 3,
        nodes_per_rack: int = 4,
        node_capacity_gb: float = 100.0,
        num_files: int = 20,
        file_size_range: tuple[float, float] = (1.0, 10.0),
        max_replication: int = 5,
        rng: Optional[np.random.Generator] = None,
    ):
        self.num_racks = num_racks
        self.nodes_per_rack = nodes_per_rack
        self.num_nodes = num_racks * nodes_per_rack
        self.node_capacity_gb = node_capacity_gb
        self.num_files = num_files
        self.file_size_range = file_size_range
        self.max_replication = max_replication
        self.rng = rng or np.random.default_rng()

        # Initialize topology
        self.nodes: dict[int, Node] = {}
        self.racks: dict[int, Rack] = {}
        self.files: dict[int, File] = {}

        self._build_topology()
        self._generate_files()

    def _build_topology(self) -> None:
        """Create racks and nodes with deterministic IDs."""
        node_id = 0
        for rack_id in range(self.num_racks):
            rack = Rack(rack_id=rack_id)
            for _ in range(self.nodes_per_rack):
                node = Node(
                    node_id=node_id,
                    rack_id=rack_id,
                    capacity_gb=self.node_capacity_gb,
                )
                self.nodes[node_id] = node
                rack.node_ids.append(node_id)
                node_id += 1
            self.racks[rack_id] = rack

    def _generate_files(self) -> None:
        """Create files with random sizes and access frequencies."""
        for file_id in range(self.num_files):
            size = self.rng.uniform(*self.file_size_range)
            # Access frequency follows a Zipf-like distribution:
            # most files are cold, few are hot
            access_freq = float(np.clip(self.rng.exponential(0.3), 0.0, 1.0))
            self.files[file_id] = File(
                file_id=file_id,
                size_gb=round(size, 2),
                access_frequency=round(access_freq, 4),
            )

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        """Reset the cluster to a clean initial state.

        All nodes become alive and healthy. Files get new random attributes
        and no replicas (the agent must place them on the first step).
        """
        if rng is not None:
            self.rng = rng

        # Reset all nodes
        for node in self.nodes.values():
            node.used_gb = 0.0
            node.is_alive = True
            node.health = 1.0

        # Regenerate files with fresh random attributes.
        # This also clears any is_permanently_lost flags from the previous
        # episode, giving the agent a clean slate each episode.
        self.files.clear()
        self._generate_files()

    # ------------------------------------------------------------------
    # Replica placement — mirrors HDFS BlockPlacementPolicyDefault
    # ------------------------------------------------------------------

    def get_alive_nodes(self) -> list[int]:
        """Return IDs of all alive nodes."""
        return [nid for nid, n in self.nodes.items() if n.is_alive]

    def get_alive_nodes_in_rack(self, rack_id: int) -> list[int]:
        """Return IDs of alive nodes in a specific rack."""
        return [
            nid
            for nid in self.racks[rack_id].node_ids
            if self.nodes[nid].is_alive
        ]

    def get_nodes_with_capacity(self, size_gb: float) -> list[int]:
        """Return IDs of alive nodes that can fit a file of the given size."""
        return [
            nid
            for nid in self.get_alive_nodes()
            if self.nodes[nid].free_gb >= size_gb
        ]

    def place_replicas(self, file: File, target_replication: int) -> int:
        """Add replicas to reach the target replication factor.

        Uses HDFS-style rack-aware placement:
        1. First replica: random alive node with capacity.
        2. Second replica: different rack from replica 1.
        3. Third replica: same rack as replica 2, different node.
        4. Replica 4+: spread across racks with fewest replicas.

        Returns the number of NEW replicas actually placed (may be less
        than requested if capacity is insufficient).
        """
        placed = 0
        current_count = self._count_alive_replicas(file)

        while current_count < target_replication:
            node_id = self._pick_placement_node(file)
            if node_id is None:
                break  # No eligible node found

            file.replica_node_ids.add(node_id)
            file.had_replicas_ever = True  # Mark that this file has been replicated
            self.nodes[node_id].used_gb += file.size_gb
            placed += 1
            current_count += 1

        file.desired_replication = target_replication
        return placed

    def remove_replicas(self, file: File, target_replication: int) -> int:
        """Remove replicas to reach the target replication factor.

        Removes replicas preferring nodes on racks that already have
        multiple replicas (to improve cross-rack distribution) and
        nodes with highest load.

        Returns the number of replicas removed.
        """
        removed = 0
        current_count = self._count_alive_replicas(file)

        while current_count > target_replication and file.replica_node_ids:
            # Pick the best replica to remove
            node_id = self._pick_removal_node(file)
            if node_id is None:
                break

            file.replica_node_ids.discard(node_id)
            self.nodes[node_id].used_gb = max(
                0.0, self.nodes[node_id].used_gb - file.size_gb
            )
            removed += 1
            current_count -= 1

        file.desired_replication = target_replication
        return removed

    def apply_replication_actions(self, actions: np.ndarray) -> dict:
        """Apply a vector of target replication factors for all files.

        Args:
            actions: Array of shape (num_files,) with target replication
                     factors, values in [0, max_replication].

        Returns:
            Dict with placement statistics.
        """
        stats = {"placed": 0, "removed": 0, "failed_placements": 0,
                 "permanently_lost": 0}

        for file_id, target_rep in enumerate(actions):
            if file_id not in self.files:
                continue

            file = self.files[file_id]

            # --- BUG FIX: Magical Restoration Guard ---
            # Only flag permanent loss AFTER at least one replica was ever
            # successfully placed. Without this guard, fresh files at episode
            # start (which have empty replica_node_ids but desired_replication=3
            # as default) would be incorrectly marked as permanently lost before
            # the agent ever got a chance to replicate them.
            if (file.had_replicas_ever
                    and not file.is_permanently_lost
                    and self._count_alive_replicas(file) == 0
                    and len(file.replica_node_ids) == 0):
                file.is_permanently_lost = True

            if file.is_permanently_lost:
                stats["permanently_lost"] += 1
                continue  # Cannot restore — skip placement entirely.

            target_rep = int(np.clip(target_rep, 0, self.max_replication))
            current = self._count_alive_replicas(file)

            if target_rep > current:
                placed = self.place_replicas(file, target_rep)
                stats["placed"] += placed
                if placed < (target_rep - current):
                    stats["failed_placements"] += (target_rep - current) - placed
            elif target_rep < current:
                removed = self.remove_replicas(file, target_rep)
                stats["removed"] += removed

        return stats

    def _count_alive_replicas(self, file: File) -> int:
        """Count replicas on nodes that are currently alive."""
        return sum(
            1
            for nid in file.replica_node_ids
            if nid in self.nodes and self.nodes[nid].is_alive
        )

    def _pick_placement_node(self, file: File) -> Optional[int]:
        """Pick the next node for replica placement using rack-aware policy.

        Follows HDFS BlockPlacementPolicyDefault:
        - Replica 1: any alive node with capacity
        - Replica 2: node on a different rack from all existing replicas
        - Replica 3: node on the same rack as replica 2 (different node)
        - Replica 4+: rack with fewest existing replicas
        """
        eligible = self.get_nodes_with_capacity(file.size_gb)
        # Exclude nodes already holding a replica (HDFS: max 1 replica per node)
        eligible = [nid for nid in eligible if nid not in file.replica_node_ids]

        if not eligible:
            return None

        alive_replica_nodes = [
            nid for nid in file.replica_node_ids if self.nodes[nid].is_alive
        ]
        num_alive = len(alive_replica_nodes)

        if num_alive == 0:
            # First replica: any eligible node
            return int(self.rng.choice(eligible))

        # Determine which racks already have replicas
        racks_with_replicas = set()
        for nid in alive_replica_nodes:
            racks_with_replicas.add(self.nodes[nid].rack_id)

        if num_alive == 1:
            # Second replica: must be on a different rack
            cross_rack = [
                nid
                for nid in eligible
                if self.nodes[nid].rack_id not in racks_with_replicas
            ]
            if cross_rack:
                return int(self.rng.choice(cross_rack))
            # Fall back to any eligible node if cross-rack impossible
            return int(self.rng.choice(eligible))

        if num_alive == 2:
            # Third replica: prefer same rack as the second replica
            # (which is the one on a different rack from the first)
            # Find the "second replica rack" — the rack that doesn't
            # contain the first replica
            rack_ids_list = [self.nodes[nid].rack_id for nid in alive_replica_nodes]
            if len(set(rack_ids_list)) >= 2:
                # Pick the rack of the most recently added replica
                second_rack = rack_ids_list[-1]
                same_rack = [
                    nid
                    for nid in eligible
                    if self.nodes[nid].rack_id == second_rack
                ]
                if same_rack:
                    return int(self.rng.choice(same_rack))

            # Fall back: any eligible node
            return int(self.rng.choice(eligible))

        # Replica 4+: pick from rack with fewest existing replicas
        rack_replica_counts: dict[int, int] = {}
        for rack_id in range(self.num_racks):
            rack_replica_counts[rack_id] = sum(
                1
                for nid in alive_replica_nodes
                if self.nodes[nid].rack_id == rack_id
            )

        # Filter eligible by rack, prefer racks with lowest count
        min_count = min(rack_replica_counts[self.nodes[nid].rack_id] for nid in eligible)
        best_rack_nodes = [
            nid
            for nid in eligible
            if rack_replica_counts[self.nodes[nid].rack_id] == min_count
        ]
        return int(self.rng.choice(best_rack_nodes))

    def _pick_removal_node(self, file: File) -> Optional[int]:
        """Pick the best replica to remove.

        Prefers removing from racks that have multiple replicas of this file
        (to maintain cross-rack distribution), and among those, prefers
        the node with highest load.
        """
        alive_replicas = [
            nid
            for nid in file.replica_node_ids
            if nid in self.nodes and self.nodes[nid].is_alive
        ]
        if not alive_replicas:
            # Remove a dead replica reference
            if file.replica_node_ids:
                return file.replica_node_ids.pop()
            return None

        # Count replicas per rack for this file
        rack_counts: dict[int, list[int]] = {}
        for nid in alive_replicas:
            rid = self.nodes[nid].rack_id
            rack_counts.setdefault(rid, []).append(nid)

        # Prefer removing from racks with multiple replicas
        max_rack_count = max(len(nids) for nids in rack_counts.values())
        candidates = []
        for rid, nids in rack_counts.items():
            if len(nids) == max_rack_count:
                candidates.extend(nids)

        # Among candidates, pick the one with highest load (free up busy nodes)
        candidates.sort(key=lambda nid: self.nodes[nid].load_fraction, reverse=True)
        return candidates[0]

    # ------------------------------------------------------------------
    # Observation vector construction
    # ------------------------------------------------------------------

    def get_observation(self) -> np.ndarray:
        """Serialize cluster state to a flat numpy observation vector.

        Layout (all values normalized to [0, 1]):
        - Per node (3 features × num_nodes):
            [health, load_fraction, rack_id_normalized]
        - Per file (4 features × num_files):
            [alive_replicas_normalized, access_frequency, size_normalized, desired_rep_normalized]

        Returns:
            np.ndarray of shape (3*num_nodes + 4*num_files,) with dtype float32
        """
        obs = np.zeros(self.observation_dim, dtype=np.float32)

        # Node features
        idx = 0
        for node_id in range(self.num_nodes):
            node = self.nodes[node_id]
            obs[idx] = node.health if node.is_alive else 0.0
            obs[idx + 1] = node.load_fraction
            obs[idx + 2] = node.rack_id / max(1, self.num_racks - 1)
            idx += 3

        # File features
        max_size = max((f.size_gb for f in self.files.values()), default=1.0)
        for file_id in range(self.num_files):
            file = self.files[file_id]
            alive = self._count_alive_replicas(file)
            obs[idx] = alive / max(1, self.max_replication)
            obs[idx + 1] = file.access_frequency
            obs[idx + 2] = file.size_gb / max(1.0, max_size)
            obs[idx + 3] = file.desired_replication / max(1, self.max_replication)
            idx += 4

        return obs

    @property
    def observation_dim(self) -> int:
        """Total dimension of the observation vector."""
        return 3 * self.num_nodes + 4 * self.num_files

    # ------------------------------------------------------------------
    # Cluster-level metrics
    # ------------------------------------------------------------------

    def total_storage_used_gb(self) -> float:
        """Total storage consumed by all replicas across all nodes."""
        return sum(node.used_gb for node in self.nodes.values())

    def total_alive_replicas(self) -> int:
        """Total number of alive replicas across all files."""
        return sum(self._count_alive_replicas(f) for f in self.files.values())

    def files_with_zero_replicas(self) -> list[int]:
        """Return file IDs that have no alive replicas (data loss).

        This includes both files that are currently unreachable because all
        their host nodes are temporarily down AND files that are permanently
        lost (is_permanently_lost=True) because all replicas were simultaneously
        destroyed with no source copy remaining to re-replicate from.
        """
        return [
            fid
            for fid, f in self.files.items()
            if f.is_permanently_lost or self._count_alive_replicas(f) == 0
        ]

    def files_under_replicated(self) -> list[int]:
        """Return file IDs with fewer alive replicas than desired."""
        return [
            fid
            for fid, f in self.files.items()
            if self._count_alive_replicas(f) < f.desired_replication
        ]

    def get_cluster_summary(self) -> dict:
        """Return a human-readable summary of the cluster state."""
        alive_nodes = len(self.get_alive_nodes())
        return {
            "total_nodes": self.num_nodes,
            "alive_nodes": alive_nodes,
            "dead_nodes": self.num_nodes - alive_nodes,
            "total_storage_used_gb": round(self.total_storage_used_gb(), 2),
            "total_capacity_gb": round(
                sum(n.capacity_gb for n in self.nodes.values()), 2
            ),
            "total_files": self.num_files,
            "files_with_data_loss": len(self.files_with_zero_replicas()),
            "files_under_replicated": len(self.files_under_replicated()),
            "total_alive_replicas": self.total_alive_replicas(),
        }

    def count_alive_replicas_for_file(self, file_id: int) -> int:
        """Public interface for counting alive replicas of a file."""
        if file_id not in self.files:
            return 0
        return self._count_alive_replicas(self.files[file_id])
