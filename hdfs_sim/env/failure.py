import numpy as np

class FailureModel:
    """Simulates hardware failures and recoveries in the cluster."""
    
    def __init__(self, node_mtbf_hours=200.0, node_mttr_hours=2.0, 
                 rack_mtbf_hours=2000.0, rack_mttr_hours=4.0):
        self.node_mtbf = node_mtbf_hours
        self.node_mttr = node_mttr_hours
        self.rack_mtbf = rack_mtbf_hours
        self.rack_mttr = rack_mttr_hours
        self.rng = np.random.default_rng()
        
        # Track recovery times for failed components
        self.node_recovery_times = {} # node_id -> time_remaining
        self.rack_recovery_times = {} # rack_id -> time_remaining

    def reset(self, rng=None):
        if rng is not None:
            self.rng = rng
        self.node_recovery_times.clear()
        self.rack_recovery_times.clear()

    def step(self, cluster, time_step_hours):
        failure_events = []
        
        # 1. Update recovery times
        for rid in list(self.rack_recovery_times.keys()):
            self.rack_recovery_times[rid] -= time_step_hours
            if self.rack_recovery_times[rid] <= 0:
                del self.rack_recovery_times[rid]
                
        for nid in list(self.node_recovery_times.keys()):
            self.node_recovery_times[nid] -= time_step_hours
            if self.node_recovery_times[nid] <= 0:
                del self.node_recovery_times[nid]
                
        # 2. Simulate new failures (Exponential distribution for time to failure)
        node_fail_prob = 1.0 - np.exp(-time_step_hours / self.node_mtbf) if self.node_mtbf > 0 else 0
        for nid, node in cluster.nodes.items():
            if nid not in self.node_recovery_times and self.rng.random() < node_fail_prob:
                self.node_recovery_times[nid] = self.rng.exponential(self.node_mttr)
                failure_events.append(f"Node {nid} failed")
                
        rack_fail_prob = 1.0 - np.exp(-time_step_hours / self.rack_mtbf) if self.rack_mtbf > 0 else 0
        for rid in cluster.racks.keys():
            if rid not in self.rack_recovery_times and self.rng.random() < rack_fail_prob:
                self.rack_recovery_times[rid] = self.rng.exponential(self.rack_mttr)
                failure_events.append(f"Rack {rid} failed")
                
        # 3. Apply state to cluster nodes
        for nid, node in cluster.nodes.items():
            # A node is dead if it is personally failed OR its rack is failed
            if nid in self.node_recovery_times or node.rack_id in self.rack_recovery_times:
                node.is_alive = False
                node.health = 0.0
            else:
                node.is_alive = True
                node.health = 1.0
                
        return failure_events
