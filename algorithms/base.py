# Base interfaces and capability specifications for fault-tolerance, detection,
# and diagnostic algorithms.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Set


class AlgorithmCapability(Enum):
    # Specific functional capabilities supported by an algorithm.
    STREAMING_DETECTION = "streaming_detection"
    RESOURCE_PREDICTION = "resource_prediction"
    CAUSAL_ROOT_CAUSE_ANALYSIS = "causal_root_cause_analysis"
    PREEMPTIVE_MIGRATION = "preemptive_migration"
    CLOSED_LOOP_FAULT_TOLERANCE = "closed_loop_fault_tolerance"


@dataclass
class DetectionResult:
    # Output of streaming detection / prediction stage.
    is_anomaly: bool
    anomaly_score: float = 0.0
    predicted_class: Optional[str] = None
    confidence: float = 1.0
    raw_output: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RCADiagnosisResult:
    # Output of root-cause localization stage.
    ranked_root_causes: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    p_values: Dict[str, float] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MitigationResult:
    # Output of recovery / mitigation actuation stage.
    action_type: str = "NONE"  # "MIGRATION", "REPLICATION", "REBALANCE", "NONE"
    target_node: Optional[str] = None
    source_node: Optional[str] = None
    tasks_affected: int = 0
    state_bytes_transferred: int = 0
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseFaultToleranceAlgorithm(ABC):
    # Unified abstract interface for all five compared approaches.
    # Enforces capability-aware querying and prevents unsupported execution.

    @property
    @abstractmethod
    def name(self) -> str:
        # Display name of the algorithm.
        pass

    @property
    @abstractmethod
    def paper_id(self) -> str:
        # Canonical ID: 'proposed_causal_ft', 'paper1_ipft', 'paper2_bwoaif', 'paper3_rcd', 'paper4_pregan'.
        pass

    @property
    @abstractmethod
    def supported_capabilities(self) -> Set[AlgorithmCapability]:
        # Exact set of functional capabilities natively supported by the algorithm.
        pass

    def supports(self, capability: AlgorithmCapability) -> bool:
        # Query if capability is natively supported.
        return capability in self.supported_capabilities

    @abstractmethod
    def initialize(self, config: Any) -> None:
        # Initialize models, hyperparameters, and internal structures.
        pass

    def fit(self, training_data: Any = None) -> None:
        # Optional model training/fitting stage on processed training data.
        # Algorithms requiring offline fitting must implement this method.
        pass

    def start(self) -> None:
        # Called immediately before the evaluation stream starts.
        pass

    def stop(self) -> None:
        # Called immediately after the evaluation stream finishes.
        pass

    def process(self, record: Any) -> DetectionResult:
        # Process a single streaming observation BenchmarkInput.
        # Lifecycle method invoking detect() or process_observation().
        return self.detect(record)

    def process_observation(self, record: Any) -> DetectionResult:
        # Process a single streaming observation.
        return self.detect(record)

    def detect(self, record: Any) -> DetectionResult:
        # Execute streaming detection on a BenchmarkInput.
        # Requires STREAMING_DETECTION or RESOURCE_PREDICTION capability.
        if not (self.supports(AlgorithmCapability.STREAMING_DETECTION) or self.supports(AlgorithmCapability.RESOURCE_PREDICTION)):
            raise NotImplementedError(f"{self.name} does not support detection/prediction capability.")
        raise NotImplementedError(f"{self.name} has not implemented detect().")

    def diagnose(self, normal_window: Any, anomalous_window: Any) -> RCADiagnosisResult:
        # Perform root-cause localization given normal and anomalous observation windows.
        # Requires CAUSAL_ROOT_CAUSE_ANALYSIS capability.
        if not self.supports(AlgorithmCapability.CAUSAL_ROOT_CAUSE_ANALYSIS):
            raise NotImplementedError(f"{self.name} does not support causal root cause diagnosis.")
        return self.diagnose_root_cause(normal_window, anomalous_window)

    def diagnose_root_cause(self, normal_window: Any, anomalous_window: Any) -> RCADiagnosisResult:
        raise NotImplementedError(f"{self.name} has not implemented diagnose_root_cause().")

    def recover(self, context: Any) -> MitigationResult:
        # Execute or plan a proactive/reactive recovery decision.
        # Requires PREEMPTIVE_MIGRATION or CLOSED_LOOP_FAULT_TOLERANCE capability.
        if not (self.supports(AlgorithmCapability.PREEMPTIVE_MIGRATION) or self.supports(AlgorithmCapability.CLOSED_LOOP_FAULT_TOLERANCE)):
            raise NotImplementedError(f"{self.name} does not support recovery/mitigation capability.")
        return self.execute_mitigation(context)

    def execute_mitigation(self, context: Any) -> MitigationResult:
        raise NotImplementedError(f"{self.name} has not implemented execute_mitigation().")

    @abstractmethod
    def reset(self) -> None:
        # Reset internal states, counters, and streaming windows between evaluation runs.
        pass
