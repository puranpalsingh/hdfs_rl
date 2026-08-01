from hdfs import InsecureClient
from typing import Dict, Any, List

class HDFSBridge:
    """A bridge to a real HDFS cluster to sync state and apply replication factors."""
    
    def __init__(self, webhdfs_url: str, user: str = "hdfs"):
        """Initialize the HDFS Bridge.
        
        Args:
            webhdfs_url: URL to the WebHDFS endpoint (e.g., 'http://namenode:50070')
            user: User to authenticate as.
        """
        self.client = InsecureClient(webhdfs_url, user=user)
        
    def get_file_status(self, hdfs_path: str) -> Dict[str, Any]:
        """Get the status of a specific file/directory.
        
        Args:
            hdfs_path: Path in HDFS.
            
        Returns:
            Dictionary containing file status (size, replication, etc).
        """
        return self.client.status(hdfs_path)
    
    def set_replication(self, hdfs_path: str, replication: int) -> bool:
        """Set the replication factor for a specific file.
        
        Args:
            hdfs_path: Path to the file in HDFS.
            replication: Target replication factor.
            
        Returns:
            True if successful.
        """
        return self.client.set_replication(hdfs_path, replication)
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Fetch general cluster information (if available).
        
        Note: The standard `hdfs` library focuses on WebHDFS REST API for file operations.
        Getting deep cluster metrics (like per-node health) might require hitting JMX endpoints
        on the NameNode (e.g., http://namenode:50070/jmx).
        """
        # Placeholder for JMX metrics fetching
        # Usually requires requests.get(f"{self.webhdfs_url}/jmx")
        return {"status": "Not fully implemented yet (requires JMX)"}
        
    def sync_decisions(self, file_paths: List[str], actions: List[int]):
        """Apply a batch of replication decisions from the RL agent to HDFS.
        
        Args:
            file_paths: List of HDFS file paths corresponding to the actions.
            actions: List of replication factors output by the agent.
        """
        if len(file_paths) != len(actions):
            raise ValueError("file_paths and actions must have the same length.")
            
        for path, rep in zip(file_paths, actions):
            try:
                self.set_replication(path, rep)
                print(f"Set replication of {path} to {rep}")
            except Exception as e:
                print(f"Failed to set replication for {path}: {e}")

