# Bounded Online State Management for Causal Fault-Tolerance Pipeline.
#
# Ensures strict memory limits, prevents memory leaks during streaming,
# and maintains an isolated simulated edge state for safe recovery modeling.

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


class NodeHealthStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    ISOLATED = "ISOLATED"
    MIGRATING = "MIGRATING"


@dataclass
class SimulatedEdgeNode:
    # Represents the simulated operational state of an edge computing node.
    node_id: str
    status: NodeHealthStatus = NodeHealthStatus.HEALTHY
    active_tasks: int = 10
    cpu_capacity_cores: int = 4
    allocated_memory_mb: float = 2048.0
    rate_limited: bool = False
    isolated: bool = False
    consecutive_faults: int = 0
    last_recovery_action: str = "NONE"
    last_recovery_timestamp: float = 0.0
    cooldown_until_pos: int = 0


class BoundedTelemetryBuffer:
    # Fixed-capacity circular buffer storing recent observable telemetry vectors.

    def __init__(self, max_capacity: int = 100):
        self.max_capacity = max_capacity
        self.features_buffer: deque = deque(maxlen=max_capacity)
        self.timestamps: deque = deque(maxlen=max_capacity)
        self.positions: deque = deque(maxlen=max_capacity)

    def append(self, position: int, timestamp: float, feature_vec: np.ndarray) -> None:
        self.positions.append(position)
        self.timestamps.append(timestamp)
        self.features_buffer.append(feature_vec.copy())

    def to_matrix(self) -> np.ndarray:
        if not self.features_buffer:
            return np.empty((0, 0), dtype=np.float64)
        return np.array(self.features_buffer)

    def __len__(self) -> int:
        return len(self.features_buffer)

    def clear(self) -> None:
        self.features_buffer.clear()
        self.timestamps.clear()
        self.positions.clear()


class CausalFTState:
    # Coordinating state manager for the Causal FT pipeline.
    # Maintains bounded memory, recent anomaly history, and simulated node topology.

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.telemetry_buffer = BoundedTelemetryBuffer(max_capacity=window_size)
        self.anomaly_scores: deque = deque(maxlen=window_size)
        self.detected_faults: deque = deque(maxlen=window_size)
        self.executed_actions: deque = deque(maxlen=window_size)

        # Simulated Edge Topology (isolated from physical historical data)
        self.nodes: Dict[str, SimulatedEdgeNode] = {
            "192.168.0.128": SimulatedEdgeNode(node_id="192.168.0.128"),
            "192.168.0.101": SimulatedEdgeNode(node_id="192.168.0.101"),
            "edge_backup_node": SimulatedEdgeNode(node_id="edge_backup_node"),
        }

    def get_or_create_node(self, node_id: str) -> SimulatedEdgeNode:
        if node_id not in self.nodes:
            self.nodes[node_id] = SimulatedEdgeNode(node_id=node_id)
        return self.nodes[node_id]

    def record_observation(
        self,
        position: int,
        timestamp: float,
        feature_vec: np.ndarray,
        anomaly_score: float,
        is_anomaly: bool,
    ) -> None:
        self.telemetry_buffer.append(position, timestamp, feature_vec)
        self.anomaly_scores.append((position, timestamp, anomaly_score))
        if is_anomaly:
            self.detected_faults.append((position, timestamp, anomaly_score))

    def reset(self) -> None:
        self.telemetry_buffer.clear()
        self.anomaly_scores.clear()
        self.detected_faults.clear()
        self.executed_actions.clear()
        for node in self.nodes.values():
            node.status = NodeHealthStatus.HEALTHY
            node.active_tasks = 10
            node.rate_limited = False
            node.isolated = False
            node.consecutive_faults = 0
            node.last_recovery_action = "NONE"
            node.last_recovery_timestamp = 0.0
            node.cooldown_until_pos = 0
