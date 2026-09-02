# IoT Devices Simulation Model.
#
# Defines diverse IoT device types and their workload generation characteristics.

from enum import Enum
import random
from typing import Dict, Any


class DeviceType(Enum):
    SENSOR = "SENSOR"            # Low throughput, periodic sensor readings
    CAMERA = "CAMERA"            # High bandwidth video stream
    ACTUATOR = "ACTUATOR"        # Control signals, latency-sensitive
    SMART_GATEWAY = "SMART_GATEWAY"  # Aggregated IoT gateway traffic


class IoTDevice:
    # Simulated IoT device producing data traffic directed to an Edge Node.

    def __init__(
        self,
        device_id: str,
        device_type: DeviceType,
        edge_node_id: str,
        base_rate: float = 10.0,
    ):
        self.device_id = device_id
        self.device_type = device_type
        self.edge_node_id = edge_node_id
        self.base_rate = base_rate  # requests / sec or packet rate

    def generate_workload(self, sim_time: float) -> float:
        # Generate active workload (requests/sec) at given simulation timestamp.
        # Modulates workload based on device type.
        noise = random.uniform(0.9, 1.1)


        if self.device_type == DeviceType.SENSOR:
            # Low rate periodic stream
            return self.base_rate * noise

        elif self.device_type == DeviceType.CAMERA:
            # High bandwidth video frames with occasional keyframe spikes
            spike = 2.5 if random.random() < 0.1 else 1.0
            return self.base_rate * 5.0 * spike * noise

        elif self.device_type == DeviceType.ACTUATOR:
            # Bursty control command traffic
            burst = 3.0 if random.random() < 0.05 else 1.0
            return self.base_rate * 0.5 * burst * noise

        elif self.device_type == DeviceType.SMART_GATEWAY:
            # Aggregated stream
            return self.base_rate * 3.0 * noise

        return self.base_rate * noise

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_type": self.device_type.value,
            "edge_node_id": self.edge_node_id,
            "base_rate": self.base_rate,
        }
