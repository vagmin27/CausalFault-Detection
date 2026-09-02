# Fault Types and Fault Injection Specifications.
#
# Defines standard fault categories (CPU overload, Memory pressure, Network congestion,
# High latency, Packet loss, Node crash failure) and recorded fault metadata.

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any


class FaultType(Enum):
    CPU_OVERLOAD = "CPU_OVERLOAD"
    MEMORY_OVERLOAD = "MEMORY_OVERLOAD"
    NETWORK_CONGESTION = "NETWORK_CONGESTION"
    HIGH_LATENCY = "HIGH_LATENCY"
    PACKET_LOSS = "PACKET_LOSS"
    EDGE_NODE_FAILURE = "EDGE_NODE_FAILURE"

    @classmethod
    def from_string(cls, name: str) -> "FaultType":
        for ft in cls:
            if ft.value.upper() == name.upper():
                return ft
        return cls.CPU_OVERLOAD


@dataclass
class FaultRecord:
    # Metadata recording an injected fault event.
    fault_type: FaultType
    target_node_id: str
    start_time: float
    duration: float
    severity: float = 1.0  # 0.0 (mild) to 1.0 (extreme)
    active: bool = False

    def is_active_at(self, current_time: float) -> bool:
        # Check if fault is active at given timestamp.
        return self.start_time <= current_time < (self.start_time + self.duration)


    def to_dict(self) -> Dict[str, Any]:
        return {
            "fault_type": self.fault_type.value,
            "target_node_id": self.target_node_id,
            "start_time": self.start_time,
            "duration": self.duration,
            "severity": self.severity,
            "active": self.active,
        }
