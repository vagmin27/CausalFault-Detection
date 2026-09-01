"""
SimPy Edge-IoT Simulation Engine.

Orchestrates IoT devices, Edge nodes, workload variation, fault injection,
and physical telemetry sampling within a discrete-event SimPy environment.
"""

import random
import logging
from typing import List, Dict, Generator, Optional
# pyrefly: ignore [missing-import]
import simpy

from .devices import IoTDevice, DeviceType
from .edge_nodes import EdgeNode
from .workload import WorkloadGenerator
from data.telemetry import TelemetryRecord

logger = logging.getLogger(__name__)


class EdgeSimulation:
    """
    SimPy-driven Edge-IoT System Simulation.
    """

    def __init__(
        self,
        num_nodes: int = 3,
        num_devices_per_node: int = 4,
        step_size: float = 1.0,
        seed: int = 42,
    ):
        self.seed = seed
        random.seed(seed)
        self.env = simpy.Environment()
        self.step_size = step_size

        # Create Edge Nodes
        self.nodes: Dict[str, EdgeNode] = {}
        for i in range(1, num_nodes + 1):
            node_id = f"Edge_Node_{i}"
            self.nodes[node_id] = EdgeNode(node_id=node_id, base_latency=12.0 + i * 2.0)

        # Create IoT Devices
        self.devices: List[IoTDevice] = []
        device_types = [DeviceType.SENSOR, DeviceType.CAMERA, DeviceType.ACTUATOR, DeviceType.SMART_GATEWAY]
        dev_counter = 1
        for node_id in self.nodes.keys():
            for j in range(num_devices_per_node):
                dev_type = device_types[j % len(device_types)]
                dev = IoTDevice(
                    device_id=f"Dev_{dev_counter}",
                    device_type=dev_type,
                    edge_node_id=node_id,
                    base_rate=8.0 + random.uniform(0, 4),
                )
                self.devices.append(dev)
                dev_counter += 1

        self.workload_gen = WorkloadGenerator(self.devices)
        self.current_step: int = 0
        self.active_fault_records: List[Dict] = []

    def register_active_fault(self, node_id: str, fault_type_str: str, severity: float):
        """Register active fault injection on target edge node."""
        if node_id in self.nodes:
            self.nodes[node_id].apply_fault(fault_type_str, severity)
            self.active_fault_records.append({
                "node_id": node_id,
                "fault_type": fault_type_str,
                "severity": severity,
                "time": self.env.now,
            })

    def clear_faults_for_node(self, node_id: str):
        """Clear active fault injection for a node."""
        if node_id in self.nodes:
            self.nodes[node_id].clear_faults()
            self.active_fault_records = [
                f for f in self.active_fault_records if f["node_id"] != node_id
            ]

    def step(self) -> List[TelemetryRecord]:
        """
        Advance simulation by one tick step size and sample TelemetryRecords
        from all active Edge Nodes.
        """
        sim_time = self.env.now
        records = []

        for node_id, node in self.nodes.items():
            # Calculate current workload demand
            workload = self.workload_gen.get_aggregated_workload_for_node(node_id, sim_time)
            node.update_metrics(workload)

            # Determine fault label and active fault type
            is_fault = 1 if (node.fault_cpu_offset > 0 or node.fault_mem_offset > 0 or
                            node.fault_net_offset > 0 or node.fault_lat_offset > 0 or
                            node.fault_loss_offset > 0 or node.node_failed) else 0

            active_f_type = "NONE"
            if is_fault:
                matching_f = [f for f in self.active_fault_records if f["node_id"] == node_id]
                if matching_f:
                    active_f_type = matching_f[-1]["fault_type"]

            # Select sample device attached to node
            node_devs = [d for d in self.devices if d.edge_node_id == node_id]
            dev_id = node_devs[0].device_id if node_devs else f"Dev_{node_id}"

            rec = TelemetryRecord(
                timestamp=round(sim_time, 2),
                device_id=dev_id,
                edge_node_id=node_id,
                cpu_utilization=round(node.cpu_utilization, 2),
                memory_utilization=round(node.memory_utilization, 2),
                network_utilization=round(node.network_utilization, 2),
                latency=round(node.latency, 2),
                packet_loss=round(node.packet_loss, 2),
                throughput=round(node.throughput, 2),
                workload=round(node.workload, 2),
                fault_label=is_fault,
                fault_type=active_f_type,
                original_label="SIMULATED_FAULT" if is_fault else "NORMAL",
            )
            records.append(rec)

        # Advance simpy environment time
        self.env.run(until=self.env.now + self.step_size)
        self.current_step += 1
        return records

    def apply_recovery(self, node_id: str, action: str) -> bool:
        """Apply recovery action to target edge node."""
        if node_id in self.nodes:
            res = self.nodes[node_id].apply_recovery_action(action)
            # Remove cleared fault record
            self.active_fault_records = [
                f for f in self.active_fault_records if f["node_id"] != node_id
            ]
            return res
        return False
