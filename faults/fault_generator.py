# Fault Injection Generator.
#
# Manages scheduled, random, and configured fault injection into the simulation
# with full random seed reproducibility.

import random
import logging
from typing import List, Dict, Optional
from .fault_types import FaultType, FaultRecord

logger = logging.getLogger(__name__)


class FaultGenerator:
    # Fault Injection Orchestrator.

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.scheduled_faults: List[FaultRecord] = []
        self.injected_history: List[FaultRecord] = []

    def set_seed(self, seed: int):
        self.seed = seed
        random.seed(seed)

    def schedule_fault(
        self,
        fault_type: FaultType,
        target_node_id: str,
        start_time: float,
        duration: float,
        severity: float = 1.0,
    ) -> FaultRecord:
        # Schedule a controlled fault at a precise timestamp.
        record = FaultRecord(
            fault_type=fault_type,
            target_node_id=target_node_id,
            start_time=start_time,
            duration=duration,
            severity=severity,
        )
        self.scheduled_faults.append(record)
        logger.info(
            f"Scheduled Fault: {fault_type.value} on {target_node_id} "
            f"at t={start_time}s for duration={duration}s (severity={severity})"
        )
        return record

    def generate_random_faults(
        self,
        target_nodes: List[str],
        sim_duration: float = 200.0,
        num_faults: int = 5,
        min_duration: float = 15.0,
        max_duration: float = 30.0,
    ) -> List[FaultRecord]:
        # Generate reproducible random fault injection schedule.
        random.seed(self.seed)
        fault_list = list(FaultType)
        generated = []

        # Avoid time t < 20 to allow warm-up normal baseline telemetry
        available_start_times = [
            25.0 + i * (sim_duration - 40.0) / max(1, num_faults)
            for i in range(num_faults)
        ]

        for i, start_t in enumerate(available_start_times):
            target = random.choice(target_nodes)
            f_type = random.choice(fault_list)
            dur = random.uniform(min_duration, max_duration)
            sev = random.uniform(0.7, 1.0)

            rec = self.schedule_fault(
                fault_type=f_type,
                target_node_id=target,
                start_time=round(start_t, 1),
                duration=round(dur, 1),
                severity=round(sev, 2),
            )
            generated.append(rec)

        return generated

    def update_simulation_faults(self, current_time: float, simulation) -> List[FaultRecord]:
        # Check scheduled faults against current simulation timestamp and inject/clear state.

        active_now = []
        for fault in self.scheduled_faults:
            if fault.is_active_at(current_time):
                if not fault.active:
                    fault.active = True
                    logger.warning(
                        f"[INJECT FAULT] Triggering {fault.fault_type.value} "
                        f"on node '{fault.target_node_id}' at t={current_time}s"
                    )
                    simulation.register_active_fault(
                        node_id=fault.target_node_id,
                        fault_type_str=fault.fault_type.value,
                        severity=fault.severity,
                    )
                    if fault not in self.injected_history:
                        self.injected_history.append(fault)
                active_now.append(fault)
            else:
                if fault.active:
                    fault.active = False
                    logger.info(
                        f"[CLEAR FAULT] Fault {fault.fault_type.value} expired "
                        f"on node '{fault.target_node_id}' at t={current_time}s"
                    )
                    simulation.clear_faults_for_node(fault.target_node_id)
        return active_now
