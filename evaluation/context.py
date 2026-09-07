"""
Context and Data Objects for Benchmark Execution.

Enforces strict separation between:
1. OBSERVABLE INPUT (BenchmarkInput) - visible to algorithms.
2. GROUND TRUTH (EventGroundTruth) - visible ONLY to the evaluation layer.
3. MEASUREMENT STATE (BenchmarkState) - tracks metrics and instrumentation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np


@dataclass(frozen=True)
class BenchmarkInput:
    """
    Standardized, strictly observable input object passed to algorithms.
    Contains ONLY features and permitted telemetry metadata.
    Guaranteed to contain NO ground-truth labels, NO future records, and NO evaluation state.
    """
    stream_position: int
    timestamp: float
    timestamp_str: str
    device_id: str
    edge_node_id: str
    features: Dict[str, float]
    feature_vector: np.ndarray

    def get_feature(self, name: str, default: float = 0.0) -> float:
        return self.features.get(name, default)


@dataclass(frozen=True)
class EventGroundTruth:
    """
    Ground-truth event metadata visible ONLY to the evaluation harness.
    Never exposed to algorithm inputs.
    """
    stream_position: int
    timestamp: float
    device_id: str
    edge_node_id: str
    is_fault: bool
    fault_label: int                # 0: Normal, 1: Attack/Fault
    raw_attack_type: str            # Original Attack_type string
    canonical_fault_category: str   # Mapped benchmark category (e.g., "DOS", "INJECTION", "MALWARE")
    root_cause_metric: Optional[str] = None


@dataclass
class BenchmarkContext:
    """
    Coordinating context for a single observation step in the benchmark.
    Maintains clean boundaries between observable data, ground truth, and timing.
    """
    stream_position: int
    observable_input: BenchmarkInput
    ground_truth: EventGroundTruth
    is_warmup: bool = False
    timing_metadata: Dict[str, Any] = field(default_factory=dict)
