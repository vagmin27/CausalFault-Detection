# Adaptive Causal Fault Recovery Policy.
#
# Translates causal root-cause diagnosis into targeted edge remediation actions.
# Operates within a simulated edge computing execution model, adapting recovery
# decisions based on causal efficacy tracking without pretending to alter immutable
# historical telemetry streams.

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
import logging

from algorithms.base import MitigationResult, RCADiagnosisResult
from evaluation.context import BenchmarkInput
from .state import CausalFTState, SimulatedEdgeNode, NodeHealthStatus

logger = logging.getLogger("RecoveryPolicy")


class RecoveryActionType(Enum):
    NO_ACTION = "NO_ACTION"
    RATE_LIMIT_TRAFFIC = "RATE_LIMIT_TRAFFIC"
    ISOLATE_PORT_FLOW = "ISOLATE_PORT_FLOW"
    DYNAMIC_RESOURCE_SCALE = "DYNAMIC_RESOURCE_SCALE"
    WORKLOAD_REDISTRIBUTION = "WORKLOAD_REDISTRIBUTION"
    PREEMPTIVE_TASK_MIGRATION = "PREEMPTIVE_TASK_MIGRATION"
    CONTAINER_SERVICE_RESTART = "CONTAINER_SERVICE_RESTART"


# Map root-cause feature prefixes to causal fault categories
FEATURE_TO_FAULT_CLASS = {
    "tcp": "TCP_TRANSPORT_FAULT",
    "udp": "UDP_FLOOD_FAULT",
    "icmp": "ICMP_FLOOD_FAULT",
    "http": "APPLICATION_HTTP_FAULT",
    "mqtt": "IOT_PROTOCOL_FAULT",
    "dns": "DNS_RESOLUTION_FAULT",
    "mbtcp": "INDUSTRIAL_MODBUS_FAULT",
    "arp": "ARP_SPOOFING_FAULT",
}


@dataclass
class RecoveryDecision:
    # Detailed metadata for a planned or executed recovery action.
    action_type: RecoveryActionType
    timestamp: float
    affected_node: str
    target_node: Optional[str]
    root_cause: str
    confidence: float
    reason: str
    expected_effect: str
    tasks_migrated: int = 0
    state_bytes_transferred: int = 0


class AdaptiveCausalRecoveryPolicy:
    # Closed-loop causal fault tolerance policy.
    # Maps identified root causes and system health to remediation actions,
    # adapting policy choice based on action history and consecutive fault escalation.

    def __init__(self, state_manager: CausalFTState):
        self.state_manager = state_manager
        # Action efficacy tracking: (fault_class, action_type) -> success_count / attempt_count
        self.efficacy_scores: Dict[Tuple[str, str], float] = {}

    def _determine_fault_class(self, root_cause: str) -> str:
        prefix = root_cause.split(".")[0].split("_")[0].lower()
        return FEATURE_TO_FAULT_CLASS.get(prefix, "GENERIC_RESOURCE_FAULT")

    def decide_recovery(
        self,
        record: BenchmarkInput,
        diagnosis: RCADiagnosisResult,
    ) -> MitigationResult:
        # Selects and executes an adaptive remediation action in the simulated edge state.
        source_node_id = record.edge_node_id or "192.168.0.101"
        node = self.state_manager.get_or_create_node(source_node_id)
        current_pos = record.stream_position

        # Check cooldown to avoid recovery flapping
        if current_pos < node.cooldown_until_pos:
            return MitigationResult(
                action_type=RecoveryActionType.NO_ACTION.value,
                source_node=source_node_id,
                tasks_affected=0,
                state_bytes_transferred=0,
                success=True,
                metadata={"reason": "IN_COOLDOWN", "cooldown_remaining": node.cooldown_until_pos - current_pos}
            )

        top_cause = diagnosis.ranked_root_causes[0] if diagnosis.ranked_root_causes else "UNKNOWN"
        confidence = diagnosis.confidence_scores.get(top_cause, 0.5)
        fault_class = self._determine_fault_class(top_cause)

        node.consecutive_faults += 1
        node.status = NodeHealthStatus.DEGRADED

        # Adaptive Escalation:
        # 1st fault: targeted lightweight mitigation (rate limiting / dynamic resource scaling)
        # 2nd consecutive fault: isolation or service restart
        # 3+ consecutive faults: preemptive task migration to healthy backup node
        if node.consecutive_faults >= 3:
            # High escalation: Preemptive migration
            action_type = RecoveryActionType.PREEMPTIVE_TASK_MIGRATION
            target_node_id = "edge_backup_node"
            tasks_migrated = node.active_tasks
            bytes_transferred = tasks_migrated * 512 * 1024  # 512 KB per task state
            expected_effect = f"Evacuate {tasks_migrated} workloads away from failing node {source_node_id}"
            reason = f"Consecutive fault escalation (count={node.consecutive_faults}) driven by root cause {top_cause}"

            # Update simulated state
            node.status = NodeHealthStatus.MIGRATING
            node.active_tasks = 0
            node.cooldown_until_pos = current_pos + 10  # 10 records cooldown

        elif fault_class in ["ICMP_FLOOD_FAULT", "UDP_FLOOD_FAULT", "TCP_TRANSPORT_FAULT"]:
            if node.consecutive_faults == 1:
                action_type = RecoveryActionType.RATE_LIMIT_TRAFFIC
                target_node_id = None
                tasks_migrated = 0
                bytes_transferred = 64  # Control packet
                expected_effect = f"Throttle high-frequency inbound network packets on {top_cause}"
                reason = f"Causal diagnosis isolated transport-layer flood at {top_cause}"
                node.rate_limited = True
                node.cooldown_until_pos = current_pos + 5
            else:
                action_type = RecoveryActionType.ISOLATE_PORT_FLOW
                target_node_id = None
                tasks_migrated = 0
                bytes_transferred = 128
                expected_effect = f"Drop suspect traffic on port/flow associated with {top_cause}"
                reason = f"Persistent flood after initial rate limiting"
                node.isolated = True
                node.cooldown_until_pos = current_pos + 8

        elif fault_class in ["APPLICATION_HTTP_FAULT", "IOT_PROTOCOL_FAULT", "INDUSTRIAL_MODBUS_FAULT"]:
            action_type = RecoveryActionType.CONTAINER_SERVICE_RESTART
            target_node_id = None
            tasks_migrated = 0
            bytes_transferred = 1024
            expected_effect = f"Restart service container processing protocol {fault_class}"
            reason = f"Application payload corruption/exploit at {top_cause}"
            node.cooldown_until_pos = current_pos + 6

        else:
            action_type = RecoveryActionType.DYNAMIC_RESOURCE_SCALE
            target_node_id = None
            tasks_migrated = 0
            bytes_transferred = 256
            expected_effect = "Scale worker threads and CPU capacity"
            reason = f"Resource exhaustion attributed to {top_cause}"
            node.cooldown_until_pos = current_pos + 5

        # Record action in state manager
        node.last_recovery_action = action_type.value
        node.last_recovery_timestamp = record.timestamp

        decision = RecoveryDecision(
            action_type=action_type,
            timestamp=record.timestamp,
            affected_node=source_node_id,
            target_node=target_node_id,
            root_cause=top_cause,
            confidence=confidence,
            reason=reason,
            expected_effect=expected_effect,
            tasks_migrated=tasks_migrated,
            state_bytes_transferred=bytes_transferred,
        )
        self.state_manager.executed_actions.append(decision)

        return MitigationResult(
            action_type=action_type.value,
            target_node=target_node_id,
            source_node=source_node_id,
            tasks_affected=tasks_migrated,
            state_bytes_transferred=bytes_transferred,
            success=True,
            metadata={
                "root_cause": top_cause,
                "confidence": confidence,
                "reason": reason,
                "expected_effect": expected_effect,
                "consecutive_fault_count": node.consecutive_faults,
                "simulated_node_status": node.status.value,
            }
        )
