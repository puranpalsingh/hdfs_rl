import numpy as np

class StaticBaseline:
    """A static baseline agent that always requests a constant replication factor."""

    def __init__(self, num_files: int, replication_factor: int = 3):
        """Initialize the static baseline.

        Args:
            num_files: Number of files in the environment.
            replication_factor: The constant replication factor to request.
        """
        self.num_files = num_files
        self.replication_factor = replication_factor

    def predict(self, observation: np.ndarray, state=None, episode_start=None, deterministic=True):
        """Predict the next action based on the observation.
        Matches the stable-baselines3 predict() signature for easy swapping.

        Args:
            observation: The environment observation (ignored by static baseline).
            state: The hidden state (ignored).
            episode_start: Whether the episode has started (ignored).
            deterministic: Whether to use deterministic actions (ignored).

        Returns:
            Tuple of (action, state).
        """
        action = np.full(self.num_files, self.replication_factor, dtype=np.int64)
        return action, None