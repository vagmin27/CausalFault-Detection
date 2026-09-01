"""
Edge Node Simulation Model.

Models physical resource capacity (CPU, Memory, Network), latency, packet loss,
and correlated system state evolution under workload and injected faults.
Supports persistent physical mitigation state following recovery action execution.
"""

import math
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class EdgeNode:
    """
    Simulated Edge Node managing resource utilization, quality of service,
    and persistent recovery state modifications.
    """

    def __init__(
        self,
        node_id: str,
        cpu_capacity: float = 100.0,
        memory_capacity: float = 100.0,
        network_capacity: float = 100.0,
        base_latency: float = 15.0,  # ms
    ):
        self.node_id = node_id
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.network_capacity = network_capacity
        self.base_latency = base_latency

        self.active: bool = True

        # State metrics
        self.workload: float = 20.0        # requests / sec
        self.cpu_utilization: float = 25.0    # %
        self.memory_utilization: float = 30.0 # %
        self.network_utilization: float = 20.0# %
        self.latency: float = base_latency    # ms
        self.packet_loss: float = 0.5         # %
        self.throughput: float = 50.0         # Mbps

        # Base fault offsets
        self.fault_cpu_offset: float = 0.0
        self.fault_mem_offset: float = 0.0
        self.fault_net_offset: float = 0.0
        self.fault_lat_offset: float = 0.0
        self.fault_loss_offset: float = 0.0
        self.node_failed: bool = False

        # Persistent Recovery Mitigation Factors (1.0 = unmitigated, < 1.0 = recovered)
        self.workload_mitigation: float = 1.0
        self.cpu_mitigation: float = 1.0
        self.mem_mitigation: float = 1.0
        self.net_mitigation: float = 1.0
        self.lat_mitigation: float = 1.0

    def update_metrics(self, current_workload: float):
        """
        Update physical system state using realistic causal equations:
        Workload ↑ => CPU utilization ↑
        CPU utilization ↑ => Latency ↑
        Network congestion ↑ => Packet loss ↑ => Latency ↑
        """
        if not self.active or self.node_failed:
            self.cpu_utilization = 0.0
            self.memory_utilization = 0.0
            self.network_utilization = 0.0
            self.latency = 999.0  # Offline / unreachable
            self.packet_loss = 100.0
            self.throughput = 0.0
            self.workload = 0.0
            return

        # Apply persistent workload offloading mitigation
        self.workload = current_workload * self.workload_mitigation

        # Apply active fault offsets scaled by persistent recovery mitigations
        eff_cpu_offset = self.fault_cpu_offset * self.cpu_mitigation
        eff_mem_offset = self.fault_mem_offset * self.mem_mitigation
        eff_net_offset = self.fault_net_offset * self.net_mitigation
        eff_lat_offset = self.fault_lat_offset * self.lat_mitigation
        eff_loss_offset = self.fault_loss_offset * self.net_mitigation

        # 1. CPU Utilization
        raw_cpu = 15.0 + (self.workload / self.cpu_capacity) * 50.0 + eff_cpu_offset
        self.cpu_utilization = max(0.0, min(100.0, raw_cpu))

        # 2. Memory Utilization
        raw_mem = 20.0 + (self.workload / self.memory_capacity) * 40.0 + eff_mem_offset
        self.memory_utilization = max(0.0, min(100.0, raw_mem))

        # 3. Network Utilization
        raw_net = 15.0 + (self.workload / self.network_capacity) * 45.0 + eff_net_offset
        self.network_utilization = max(0.0, min(100.0, raw_net))

        # 4. Throughput (Mbps)
        self.throughput = max(0.0, self.workload * 2.5 * (1.0 - self.network_utilization / 150.0))

        # 5. Packet Loss
        congestion_loss = max(0.0, (self.network_utilization - 75.0) * 0.8) if self.network_utilization > 75.0 else 0.0
        raw_loss = congestion_loss + eff_loss_offset
        self.packet_loss = max(0.0, min(100.0, raw_loss))

        # 6. Processing Latency
        cpu_delay = math.pow(self.cpu_utilization / 100.0, 2.0) * 120.0
        loss_delay = self.packet_loss * 5.0
        raw_lat = self.base_latency + cpu_delay + loss_delay + eff_lat_offset
        self.latency = max(1.0, raw_lat)

    def apply_fault(
        self,
        fault_type_str: str,
        severity: float = 1.0,
    ):
        """Apply base fault offsets."""
        if fault_type_str == "CPU_OVERLOAD":
            self.fault_cpu_offset = 60.0 * severity
        elif fault_type_str == "MEMORY_OVERLOAD":
            self.fault_mem_offset = 65.0 * severity
        elif fault_type_str == "NETWORK_CONGESTION":
            self.fault_net_offset = 60.0 * severity
            self.fault_loss_offset = 15.0 * severity
        elif fault_type_str == "HIGH_LATENCY":
            self.fault_lat_offset = 150.0 * severity
        elif fault_type_str == "PACKET_LOSS":
            self.fault_loss_offset = 35.0 * severity
        elif fault_type_str == "EDGE_NODE_FAILURE":
            self.node_failed = True

    def clear_faults(self):
        """Reset fault offsets and recovery mitigation factors to baseline."""
        self.fault_cpu_offset = 0.0
        self.fault_mem_offset = 0.0
        self.fault_net_offset = 0.0
        self.fault_lat_offset = 0.0
        self.fault_loss_offset = 0.0
        self.node_failed = False

        # Reset mitigations
        self.workload_mitigation = 1.0
        self.cpu_mitigation = 1.0
        self.mem_mitigation = 1.0
        self.net_mitigation = 1.0
        self.lat_mitigation = 1.0

    def apply_recovery_action(self, action: str) -> bool:
        """
        Execute adaptive recovery action establishing persistent mitigation factors.
        Returns True if action was executed successfully.
        """
        logger.info(f"[Node {self.node_id}] Applying persistent recovery action: {action}")

        if action == "WORKLOAD_REDISTRIBUTION":
            # Offload 60% workload, reduce CPU fault impact by 70%
            self.workload_mitigation = 0.4
            self.cpu_mitigation = 0.3

        elif action == "TRAFFIC_REROUTING":
            # Reroute traffic, reduce network & packet loss impact by 80%
            self.net_mitigation = 0.2
            self.lat_mitigation = 0.3

        elif action == "TASK_MIGRATION":
            # Migrate memory/CPU tasks off node
            self.mem_mitigation = 0.2
            self.cpu_mitigation = 0.4
            self.workload_mitigation = 0.5

        elif action == "EDGE_NODE_FAILOVER":
            # Restart or failover node
            self.node_failed = False
            self.clear_faults()
            self.active = True

        elif action == "RESOURCE_REBALANCING":
            # Dynamically allocate extra virtual CPU/RAM capacity
            self.cpu_mitigation = 0.5
            self.mem_mitigation = 0.5
            self.lat_mitigation = 0.4

        # Re-evaluate metrics immediately after recovery
        self.update_metrics(self.workload / max(1e-5, self.workload_mitigation))
        return True

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "active": self.active,
            "cpu_utilization": round(self.cpu_utilization, 2),
            "memory_utilization": round(self.memory_utilization, 2),
            "network_utilization": round(self.network_utilization, 2),
            "latency": round(self.latency, 2),
            "packet_loss": round(self.packet_loss, 2),
            "throughput": round(self.throughput, 2),
            "workload": round(self.workload, 2),
        }
