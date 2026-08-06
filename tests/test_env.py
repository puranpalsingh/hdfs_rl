"""Tests for the HDFSReplicationEnv Gymnasium environment.

These tests verify:
1. The environment passes Gymnasium's env_checker (API compliance)
2. Reset and step produce valid observations
3. Episodes terminate correctly
4. Rewards are computed and finite
5. The environment is deterministic when seeded
"""

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from hdfs_sim.env.hdfs_env import HDFSReplicationEnv


@pytest.fixture
def env():
    """Create a fresh environment instance."""
    return HDFSReplicationEnv(seed=42)


class TestEnvChecker:
    """Run Gymnasium's built-in environment validator."""

    def test_check_env_passes(self):
        """The environment must pass Gymnasium's strict API checks.

        This validates observation/action space types, reset/step signatures,
        return types, and value ranges. If this test fails, SB3 won't work.
        """
        env = HDFSReplicationEnv(seed=42)
        # check_env raises an exception if anything is wrong
        check_env(env, warn=True)


class TestReset:
    """Test environment reset behavior."""

    def test_reset_returns_observation_and_info(self, env):
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_reset_observation_shape(self, env):
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape

    def test_reset_observation_in_bounds(self, env):
        obs, _ = env.reset()
        assert np.all(obs >= env.observation_space.low)
        assert np.all(obs <= env.observation_space.high)

    def test_reset_with_seed_is_deterministic(self):
        env1 = HDFSReplicationEnv(seed=123)
        env2 = HDFSReplicationEnv(seed=123)
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_info_has_required_keys(self, env):
        _, info = env.reset()
        assert "step" in info
        assert "simulated_time_hours" in info
        assert "cluster_summary" in info


class TestStep:
    """Test environment step behavior."""

    def test_step_returns_five_values(self, env):
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result

    def test_step_observation_shape(self, env):
        env.reset()
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert obs.shape == env.observation_space.shape

    def test_step_observation_in_bounds(self, env):
        env.reset()
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert np.all(obs >= env.observation_space.low)
        assert np.all(obs <= env.observation_space.high)

    def test_step_reward_is_finite(self, env):
        env.reset()
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        assert np.isfinite(reward)

    def test_step_returns_booleans_for_termination(self, env):
        env.reset()
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_info_has_reward_breakdown(self, env):
        env.reset()
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert "reward_breakdown" in info
        breakdown = info["reward_breakdown"]
        assert "storage_cost" in breakdown
        assert "data_loss_penalty" in breakdown

    def test_step_count_increments(self, env):
        env.reset()
        for i in range(5):
            action = env.action_space.sample()
            _, _, _, _, info = env.step(action)
            assert info["step"] == i + 1


class TestEpisodeTermination:
    """Test that episodes terminate correctly after max_steps."""

    def test_episode_truncates_at_max_steps(self):
        env = HDFSReplicationEnv(max_steps=10, seed=42)
        env.reset()

        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if step < 9:
                assert not truncated, f"Truncated early at step {step}"
                assert not terminated
            else:
                assert truncated, "Should be truncated at max_steps"
                assert not terminated  # We never set terminated

    def test_no_early_termination(self):
        """The environment should never terminate early."""
        env = HDFSReplicationEnv(max_steps=50, seed=42)
        env.reset()

        for _ in range(49):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            assert not terminated


class TestActionSpace:
    """Test action space properties."""

    def test_action_space_type(self, env):
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)

    def test_action_space_size(self, env):
        assert len(env.action_space.nvec) == env.num_files

    def test_action_space_range(self, env):
        # Each file's action should range from 0 to max_replication-2
        for n in env.action_space.nvec:
            assert n == env.max_replication - 1

    def test_sampled_actions_are_valid(self, env):
        for _ in range(100):
            action = env.action_space.sample()
            assert env.action_space.contains(action)


class TestObservationSpace:
    """Test observation space properties."""

    def test_observation_space_type(self, env):
        assert isinstance(env.observation_space, gym.spaces.Box)

    def test_observation_space_dimension(self, env):
        expected = 3 * env.num_nodes + 4 * env.num_files
        assert env.observation_space.shape == (expected,)

    def test_observation_space_bounds(self, env):
        assert np.all(env.observation_space.low == 0.0)
        assert np.all(env.observation_space.high == 1.0)


class TestReplicationBehavior:
    """Test that the environment actually applies replication decisions."""

    def test_replication_factor_3_creates_replicas(self):
        env = HDFSReplicationEnv(
            num_files=5, max_replication=5, seed=42,
        )
        env.reset()

        # Set all files to replication=3 (action 1 maps to RF 3)
        action = np.full(5, 1, dtype=np.int64)
        env.step(action)

        for fid in range(5):
            alive = env.cluster.count_alive_replicas_for_file(fid)
            assert alive == 3, f"File {fid} has {alive} replicas, expected 3"

    def test_reducing_replication_removes_replicas(self):
        env = HDFSReplicationEnv(
            num_files=5, max_replication=5, seed=42,
        )
        env.reset()  # reset places 3 replicas per file

        # Reduce to 2 (action 0 maps to RF 2)
        action = np.full(5, 0, dtype=np.int64)
        env.step(action)

        for fid in range(5):
            alive = env.cluster.count_alive_replicas_for_file(fid)
            assert alive == 2, f"File {fid} has {alive} replicas, expected 2"


class TestRender:
    """Test render output."""

    def test_render_ansi_returns_string(self):
        env = HDFSReplicationEnv(render_mode="ansi", seed=42)
        env.reset()
        output = env.render()
        assert isinstance(output, str)
        assert "HDFS Replication Simulator" in output
