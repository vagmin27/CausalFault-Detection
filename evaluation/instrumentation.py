# System Instrumentation and Resource Profiling Utilities.
#
# Provides real-time measurement tools for:
# - perf_counter_ns execution timing
# - Process-level CPU utilization sampling
# - Peak RSS memory tracking
# - Processing throughput (records/sec)
# - Event timestamps
# - Bandwidth accounting (telemetry & migration bytes)
# - Estimated energy consumption modeling
# - Operational recovery action counters

import time
import os
import threading
import psutil
from typing import Optional, Dict, Any, List


class ExecutionTimer:
    # High-precision execution timer utilizing time.perf_counter_ns().

    def __init__(self):
        self.start_ns: int = 0
        self.end_ns: int = 0
        self.elapsed_ns: int = 0
        self.is_running: bool = False

    def start(self) -> None:
        self.start_ns = time.perf_counter_ns()
        self.is_running = True

    def stop(self) -> float:
        if self.is_running:
            self.end_ns = time.perf_counter_ns()
            self.elapsed_ns += (self.end_ns - self.start_ns)
            self.is_running = False
        return self.elapsed_ms()

    def reset(self) -> None:
        self.start_ns = 0
        self.end_ns = 0
        self.elapsed_ns = 0
        self.is_running = False

    def elapsed_ms(self) -> float:
        current_ns = self.elapsed_ns
        if self.is_running:
            current_ns += (time.perf_counter_ns() - self.start_ns)
        return current_ns / 1_000_000.0

    def elapsed_seconds(self) -> float:
        return self.elapsed_ms() / 1000.0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class ResourceMonitor:
    # Monitors process-isolated CPU utilization and resident set size (RSS) memory.

    def __init__(self, sample_interval_sec: float = 0.05):
        self.sample_interval = sample_interval_sec
        self.process = psutil.Process(os.getpid())
        self.cpu_samples: List[float] = []
        self.peak_rss_bytes: int = 0
        self.baseline_rss_bytes: int = 0
        self._monitoring: bool = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.cpu_samples = []
        self.baseline_rss_bytes = self.process.memory_info().rss
        self.peak_rss_bytes = self.baseline_rss_bytes
        self._monitoring = True
        self.process.cpu_percent(interval=None)  # prime cpu counter

        def _monitor_loop():
            while self._monitoring:
                try:
                    cpu = self.process.cpu_percent(interval=None)
                    rss = self.process.memory_info().rss
                    if cpu > 0.0:
                        self.cpu_samples.append(cpu)
                    if rss > self.peak_rss_bytes:
                        self.peak_rss_bytes = rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                time.sleep(self.sample_interval)

        self._thread = threading.Thread(target=_monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._monitoring = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        # Take final sample
        try:
            rss = self.process.memory_info().rss
            if rss > self.peak_rss_bytes:
                self.peak_rss_bytes = rss
        except Exception:
            pass

    def get_average_cpu_percent(self) -> float:
        if not self.cpu_samples:
            return 0.0
        return float(sum(self.cpu_samples) / len(self.cpu_samples))

    def get_peak_rss_mb(self) -> float:
        return float(self.peak_rss_bytes / (1024 * 1024))

    def get_memory_overhead_mb(self) -> float:
        return float((self.peak_rss_bytes - self.baseline_rss_bytes) / (1024 * 1024))


class ThroughputCounter:
    # Calculates sustained observation ingestion throughput (records/sec).

    def __init__(self):
        self.total_records: int = 0
        self.timer = ExecutionTimer()

    def start(self) -> None:
        self.total_records = 0
        self.timer.start()

    def record_observation(self, count: int = 1) -> None:
        self.total_records += count

    def stop(self) -> float:
        self.timer.stop()
        return self.get_records_per_sec()

    def get_records_per_sec(self) -> float:
        sec = self.timer.elapsed_seconds()
        if sec <= 0.0 or self.total_records == 0:
            return 0.0
        return float(self.total_records / sec)


class BandwidthAccountant:
    # Tracks telemetry streaming byte volumes and migration state transfer bandwidth.

    def __init__(self):
        self.telemetry_bytes: int = 0
        self.control_bytes: int = 0
        self.migration_state_bytes: int = 0
        self.timer = ExecutionTimer()

    def start(self) -> None:
        self.telemetry_bytes = 0
        self.control_bytes = 0
        self.migration_state_bytes = 0
        self.timer.start()

    def add_telemetry_bytes(self, num_bytes: int) -> None:
        self.telemetry_bytes += num_bytes

    def add_control_bytes(self, num_bytes: int) -> None:
        self.control_bytes += num_bytes

    def add_migration_state_bytes(self, num_bytes: int) -> None:
        self.migration_state_bytes += num_bytes

    def stop(self) -> None:
        self.timer.stop()

    def get_total_bytes(self) -> int:
        return self.telemetry_bytes + self.control_bytes + self.migration_state_bytes

    def get_bandwidth_kb_per_sec(self) -> float:
        sec = self.timer.elapsed_seconds()
        if sec <= 0.0:
            return 0.0
        return float((self.get_total_bytes() / 1024.0) / sec)


class EnergyEstimator:
    # Estimates electrical energy consumption using a calibrated linear-polynomial
    # power model based on CPU utilization and execution time.

    def __init__(self, idle_watts: float = 2.7, peak_watts: float = 6.4):
        self.idle_watts = idle_watts
        self.peak_watts = peak_watts

    def estimate_energy_joules(self, cpu_utilization_percent: float, elapsed_seconds: float) -> float:
        # Energy (Joules) = Power (Watts) * Time (Seconds)
        # Power = P_idle + (P_peak - P_idle) * (CPU_util / 100)
        cpu_fraction = max(0.0, min(100.0, cpu_utilization_percent)) / 100.0
        instantaneous_power_watts = self.idle_watts + (self.peak_watts - self.idle_watts) * cpu_fraction
        return float(instantaneous_power_watts * elapsed_seconds)

    def estimate_energy_watt_hours(self, cpu_utilization_percent: float, elapsed_seconds: float) -> float:
        return float(self.estimate_energy_joules(cpu_utilization_percent, elapsed_seconds) / 3600.0)


class ActionCounter:
    # Tracks operational mitigation actions and SLA violation episodes.

    def __init__(self, slo_latency_threshold_ms: float = 100.0):
        self.slo_threshold_ms = slo_latency_threshold_ms
        self.total_observations: int = 0
        self.migrations_count: int = 0
        self.replications_count: int = 0
        self.rebalances_count: int = 0
        self.slo_violations_count: int = 0
        self.mitigation_successes: int = 0
        self.mitigation_failures: int = 0

    def record_action(self, action_type: str, success: bool = True) -> None:
        act = action_type.upper()
        if act == "MIGRATION":
            self.migrations_count += 1
        elif act == "REPLICATION":
            self.replications_count += 1
        elif act in ["REBALANCE", "REDISTRIBUTE"]:
            self.rebalances_count += 1

        if act != "NONE":
            if success:
                self.mitigation_successes += 1
            else:
                self.mitigation_failures += 1

    def record_latency(self, latency_ms: float) -> None:
        self.total_observations += 1
        if latency_ms > self.slo_threshold_ms:
            self.slo_violations_count += 1

    def get_total_actions(self) -> int:
        return self.migrations_count + self.replications_count + self.rebalances_count

    def get_slo_violation_rate(self) -> float:
        if self.total_observations == 0:
            return 0.0
        return float(self.slo_violations_count / self.total_observations) * 100.0

    def get_mitigation_success_rate(self) -> float:
        total = self.mitigation_successes + self.mitigation_failures
        if total == 0:
            return 0.0
        return float(self.mitigation_successes / total) * 100.0

    def calculate_operational_recovery_cost(self, w_action: float = 0.4, w_slo: float = 0.4, w_fail: float = 0.2) -> float:
        action_ratio = float(self.get_total_actions() / max(1, self.total_observations))
        slo_ratio = float(self.slo_violations_count / max(1, self.total_observations))
        fail_ratio = float(self.mitigation_failures / max(1, (self.mitigation_successes + self.mitigation_failures)))
        return float(w_action * action_ratio + w_slo * slo_ratio + w_fail * fail_ratio)


class SystemInstrumentation:
    # Integrated facade coordinating all instrumentation domains for an experiment run.

    def __init__(self, idle_watts: float = 2.7, peak_watts: float = 6.4, slo_threshold_ms: float = 100.0):
        self.timer = ExecutionTimer()
        self.resource_monitor = ResourceMonitor()
        self.throughput_counter = ThroughputCounter()
        self.bandwidth_accountant = BandwidthAccountant()
        self.energy_estimator = EnergyEstimator(idle_watts=idle_watts, peak_watts=peak_watts)
        self.action_counter = ActionCounter(slo_latency_threshold_ms=slo_threshold_ms)

    def start_session(self) -> None:
        self.timer.start()
        self.resource_monitor.start()
        self.throughput_counter.start()
        self.bandwidth_accountant.start()

    def end_session(self) -> Dict[str, Any]:
        elapsed_sec = self.timer.stop() / 1000.0
        self.resource_monitor.stop()
        rec_per_sec = self.throughput_counter.stop()
        self.bandwidth_accountant.stop()

        avg_cpu = self.resource_monitor.get_average_cpu_percent()
        peak_rss = self.resource_monitor.get_peak_rss_mb()
        est_joules = self.energy_estimator.estimate_energy_joules(avg_cpu, elapsed_sec)
        est_wh = self.energy_estimator.estimate_energy_watt_hours(avg_cpu, elapsed_sec)
        bw_kb_sec = self.bandwidth_accountant.get_bandwidth_kb_per_sec()
        op_cost = self.action_counter.calculate_operational_recovery_cost()

        return {
            "elapsed_seconds": round(elapsed_sec, 4),
            "throughput_rec_sec": round(rec_per_sec, 2),
            "avg_cpu_percent": round(avg_cpu, 2),
            "peak_rss_mb": round(peak_rss, 2),
            "bandwidth_kb_sec": round(bw_kb_sec, 2),
            "estimated_energy_joules": round(est_joules, 4),
            "estimated_energy_wh": round(est_wh, 6),
            "total_actions": self.action_counter.get_total_actions(),
            "slo_violation_rate": round(self.action_counter.get_slo_violation_rate(), 2),
            "operational_recovery_cost": round(op_cost, 4),
        }
