# Workload Generation and Physical Metric Correlation Engine.
#
# Simulates dynamic, time-varying workload patterns (diurnal cycles, bursty spikes)
# with explicit physical dependencies (Workload -> CPU -> Latency -> Packet Loss).

import math
import random
from typing import List, Dict
from .devices import IoTDevice


class WorkloadGenerator:
    # Generates dynamic aggregated workload for Edge Nodes from attached IoT devices.

    def __init__(self, devices: List[IoTDevice], base_period: float = 100.0):
        self.devices = devices
        self.base_period = base_period

    def get_aggregated_workload_for_node(self, node_id: str, sim_time: float) -> float:
        # Sum generated workload from all IoT devices associated with target node_id
        # including periodic diurnal wave variation.
        node_devices = [d for d in self.devices if d.edge_node_id == node_id]
        if not node_devices:
            return 10.0  # default base background workload

        total_workload = 0.0
        for dev in node_devices:
            total_workload += dev.generate_workload(sim_time)

        # Diurnal sine wave cycle factor (min 0.7x, max 1.3x base load)
        diurnal_factor = 1.0 + 0.3 * math.sin(2.0 * math.pi * sim_time / self.base_period)
        return total_workload * diurnal_factor

