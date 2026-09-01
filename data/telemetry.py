"""
Common Telemetry Record and Data Source Interface.

This module defines the unified TelemetryRecord class and DataSource interface.
All simulation data and real-world dataset feeds are converted into TelemetryRecord
instances so downstream components (Detector, Causal Analyzer, Recovery Manager, Evaluator)
remain completely decoupled from data source schemas.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Generator


@dataclass
class TelemetryRecord:
    """
    Standardized Telemetry Record representation.
    
    Fields marked Optional may be None if a specific dataset does not collect
    or provide that telemetry dimension.
    """
    timestamp: float
    device_id: str
    edge_node_id: str
    cpu_utilization: Optional[float] = None       # percentage (0.0 to 100.0)
    memory_utilization: Optional[float] = None    # percentage (0.0 to 100.0)
    network_utilization: Optional[float] = None   # percentage (0.0 to 100.0)
    latency: Optional[float] = None              # in milliseconds (ms)
    packet_loss: Optional[float] = None          # percentage (0.0 to 100.0)
    throughput: Optional[float] = None           # Mbps
    workload: Optional[float] = None             # requests / sec
    fault_label: int = 0                         # 0: Normal, 1: Fault / Anomaly
    fault_type: Optional[str] = "NONE"           # e.g., "CPU_OVERLOAD", "NONE", or raw attack category
    original_label: Optional[str] = "NORMAL"     # raw label string from dataset

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary representation."""
        return asdict(self)

    def get_available_features(self) -> Dict[str, float]:
        """
        Extract numerical telemetry feature dimensions that are not None.
        Returns a dictionary of feature_name -> float value.
        """
        features = {
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "network_utilization": self.network_utilization,
            "latency": self.latency,
            "packet_loss": self.packet_loss,
            "throughput": self.throughput,
            "workload": self.workload,
        }
        return {k: v for k, v in features.items() if v is not None}

    def to_feature_vector(self, feature_order: Optional[List[str]] = None) -> List[float]:
        """
        Convert available numerical features to a flat feature vector for ML models.
        """
        if feature_order is None:
            feature_order = [
                "cpu_utilization",
                "memory_utilization",
                "network_utilization",
                "latency",
                "packet_loss",
                "throughput",
                "workload",
            ]
        vec = []
        for feat in feature_order:
            val = getattr(self, feat, None)
            vec.append(float(val) if val is not None else 0.0)
        return vec

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get standard list of numerical telemetry feature names."""
        return [
            "cpu_utilization",
            "memory_utilization",
            "network_utilization",
            "latency",
            "packet_loss",
            "throughput",
            "workload",
        ]


class DataSource(ABC):
    """
    Abstract Interface for Telemetry Data Sources.
    """

    @abstractmethod
    def stream_telemetry(self) -> Generator[TelemetryRecord, None, None]:
        """Yield TelemetryRecord instances sequentially (streaming)."""
        pass

    @abstractmethod
    def get_dataset_name(self) -> str:
        """Return human-readable identifier of the data source."""
        pass
