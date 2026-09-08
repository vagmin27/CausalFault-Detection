# Centralized Metric Definitions, Statistical Formulations, and Calculation Functions.
#
# Implements all 10 Headline Parameters:
# 1. Latency / Delay
# 2. Execution / Response Time
# 3. Accuracy
# 4. Computational Capacity & Processing Throughput
# 5. Resource Utilization
# 6. Bandwidth
# 7. Energy Consumption / Efficiency
# 8. Operational Recovery Cost
# 9. Reliability & Availability
# 10. Scalability

from typing import List, Dict, Any, Optional, Tuple
import math
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from .results_schema import BenchmarkResultRecord, MeasurementType, Comparability


# -----------------------------------------------------------------------------
# Statistical Formulations
# -----------------------------------------------------------------------------

def calculate_mean_std(values: List[float]) -> Tuple[float, float]:
    # Calculate sample mean and standard deviation.
    clean = [v for v in values if v is not None and not math.isnan(v) and not math.isinf(v)]
    if not clean:
        return 0.0, 0.0
    if len(clean) == 1:
        return float(clean[0]), 0.0
    return float(np.mean(clean)), float(np.std(clean, ddof=1))


def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> List[float]:
    # Calculate 95% (or specified) two-sided Student's t / normal confidence interval.
    clean = [v for v in values if v is not None and not math.isnan(v) and not math.isinf(v)]
    n = len(clean)
    if n < 2:
        val = clean[0] if clean else 0.0
        return [float(val), float(val)]
    mean = np.mean(clean)
    std_err = np.std(clean, ddof=1) / math.sqrt(n)
    # Z-factor approximation for 95% CI is 1.96
    z = 1.96 if confidence == 0.95 else 2.576
    margin = z * std_err
    return [float(mean - margin), float(mean + margin)]


# -----------------------------------------------------------------------------
# Parameter 1: Latency / Delay Formulations
# -----------------------------------------------------------------------------

def calculate_detection_latency_mttd(onset_times: List[float], detect_times: List[float]) -> float:
    # MTTD = (1 / |F|) * sum(t_detect - t_onset)
    # Lower is better (ms).
    delays = [max(0.0, dt - ot) for ot, dt in zip(onset_times, detect_times) if dt >= ot]
    return float(np.mean(delays)) if delays else 0.0


def calculate_diagnosis_latency(detect_times: List[float], diagnosis_times: List[float]) -> float:
    # T_diag = (1 / |F|) * sum(t_rca_complete - t_detect)
    # Lower is better (ms).
    delays = [max(0.0, dgt - dt) for dt, dgt in zip(detect_times, diagnosis_times) if dgt >= dt]
    return float(np.mean(delays)) if delays else 0.0


def calculate_recovery_latency(diagnosis_times: List[float], recovery_times: List[float]) -> float:
    # T_act = (1 / |F|) * sum(t_recovery - t_rca_complete)
    # Lower is better (ms).
    delays = [max(0.0, rt - dgt) for dgt, rt in zip(diagnosis_times, recovery_times) if rt >= dgt]
    return float(np.mean(delays)) if delays else 0.0


def calculate_end_to_end_latency(onset_times: List[float], recovery_times: List[float]) -> float:
    # T_e2e = (1 / |F|) * sum(t_restored - t_onset)
    # Lower is better (ms).
    delays = [max(0.0, rt - ot) for ot, rt in zip(onset_times, recovery_times) if rt >= ot]
    return float(np.mean(delays)) if delays else 0.0


# -----------------------------------------------------------------------------
# Parameter 2: Execution / Response Time Formulations
# -----------------------------------------------------------------------------

def calculate_per_record_inference_latency(execution_times_ms: List[float]) -> float:
    # Average pure algorithmic CPU time per observation (ms/rec). Lower is better.
    return float(np.mean(execution_times_ms)) if execution_times_ms else 0.0


def calculate_service_response_time(response_times_ms: List[float]) -> float:
    # Average task service response time (ms). Lower is better.
    return float(np.mean(response_times_ms)) if response_times_ms else 0.0


def calculate_algorithmic_overhead_ratio(algorithm_cpu_sec: float, total_wall_clock_sec: float) -> float:
    # Dimensionless overhead fraction = Alg_CPU / Total_Runtime. Lower is better.
    if total_wall_clock_sec <= 0.0:
        return 0.0
    return float(min(1.0, algorithm_cpu_sec / total_wall_clock_sec))


# -----------------------------------------------------------------------------
# Parameter 3: Accuracy Formulations
# -----------------------------------------------------------------------------

def calculate_detection_accuracy(
    y_true: List[int],
    y_pred: List[int],
    y_scores: Optional[List[float]] = None
) -> Dict[str, float]:
    # Submetric 3.1: Detection Accuracy.
    # Returns: precision, recall, f1, accuracy, roc_auc (if scores provided).
    # Higher is better (0.0 - 100.0%).
    if not y_true or not y_pred:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "roc_auc": 0.0}

    prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
    rec = recall_score(y_true, y_pred, zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100.0
    acc = accuracy_score(y_true, y_pred) * 100.0

    auc = 0.0
    if y_scores is not None and len(set(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_scores)
        except Exception:
            auc = 0.0

    return {
        "precision": round(float(prec), 2),
        "recall": round(float(rec), 2),
        "f1": round(float(f1), 2),
        "accuracy": round(float(acc), 2),
        "roc_auc": round(float(auc), 4),
    }


def calculate_rca_localization_accuracy(
    ground_truth_causes: List[str],
    predicted_rankings: List[List[str]]
) -> Dict[str, float]:
    # Submetric 3.2: Root-Cause Localization Top-k Recall.
    # Higher is better (0.0 - 100.0%).
    if not ground_truth_causes or not predicted_rankings:
        return {"top_1_recall": 0.0, "top_3_recall": 0.0, "top_5_recall": 0.0}

    top_1_hits = 0
    top_3_hits = 0
    top_5_hits = 0
    total = len(ground_truth_causes)

    for true_cause, ranked_list in zip(ground_truth_causes, predicted_rankings):
        clean_true = str(true_cause).strip().lower()
        clean_ranked = [str(r).strip().lower() for r in ranked_list]

        if len(clean_ranked) >= 1 and clean_true == clean_ranked[0]:
            top_1_hits += 1
        if any(clean_true == r for r in clean_ranked[:3]):
            top_3_hits += 1
        if any(clean_true == r for r in clean_ranked[:5]):
            top_5_hits += 1

    return {
        "top_1_recall": round(float(top_1_hits / total) * 100.0, 2),
        "top_3_recall": round(float(top_3_hits / total) * 100.0, 2),
        "top_5_recall": round(float(top_5_hits / total) * 100.0, 2),
    }


def calculate_recovery_success_rate(total_actions: int, successful_actions: int) -> float:
    # Submetric 3.3: Recovery Mitigation Success Rate.
    # Higher is better (0.0 - 100.0%).
    if total_actions <= 0:
        return 0.0
    return round(float(successful_actions / total_actions) * 100.0, 2)


# -----------------------------------------------------------------------------
# Parameter 4: Computational Capacity & Processing Throughput
# -----------------------------------------------------------------------------

def get_hardware_capacity_profile(config: Any) -> Dict[str, Any]:
    # Submetric 4.1: Hardware Computational Capacity (Static System Profile).
    if hasattr(config, "hardware"):
        return config.hardware.to_dict()
    return {}


def calculate_processing_throughput(num_records: int, total_wall_time_sec: float) -> float:
    # Submetric 4.2: Processing Throughput (records/second). Higher is better.
    if total_wall_time_sec <= 0.0:
        return 0.0
    return round(float(num_records / total_wall_time_sec), 2)


# -----------------------------------------------------------------------------
# Parameter 5: Resource Utilization Formulations
# -----------------------------------------------------------------------------

def calculate_average_cpu_utilization(cpu_samples: List[float]) -> float:
    # Submetric 5.1: Average CPU Utilization (%). Lower is better.
    return round(float(np.mean(cpu_samples)), 2) if cpu_samples else 0.0


def calculate_peak_ram_footprint(peak_rss_bytes: int) -> float:
    # Submetric 5.2: Resident Memory Footprint (Peak RSS in MB). Lower is better.
    return round(float(peak_rss_bytes / (1024 * 1024)), 2)


# -----------------------------------------------------------------------------
# Parameter 6: Bandwidth Formulations
# -----------------------------------------------------------------------------

def calculate_telemetry_streaming_bandwidth(telemetry_bytes: int, elapsed_sec: float) -> float:
    # Submetric 6.1: Telemetry Streaming Rate (KB/s). Lower is better.
    if elapsed_sec <= 0.0:
        return 0.0
    return round(float((telemetry_bytes / 1024.0) / elapsed_sec), 2)


def calculate_migration_bandwidth_overhead(migration_bytes: int) -> float:
    # Submetric 6.2: State Migration Bandwidth Overhead (Total MB). Lower is better.
    return round(float(migration_bytes / (1024 * 1024)), 2)


# -----------------------------------------------------------------------------
# Parameter 7: Energy Consumption / Efficiency Formulations
# -----------------------------------------------------------------------------

def calculate_estimated_energy_joules(
    avg_cpu_percent: float,
    elapsed_sec: float,
    idle_watts: float = 2.7,
    peak_watts: float = 6.4
) -> float:
    # Submetric 7.2: Estimated Energy Consumption (Joules).
    # Polynomial model: P(t) = P_idle + (P_peak - P_idle) * (CPU% / 100)
    # Explicitly labeled as ESTIMATED. Lower is better.
    cpu_fraction = max(0.0, min(100.0, avg_cpu_percent)) / 100.0
    power_watts = idle_watts + (peak_watts - idle_watts) * cpu_fraction
    return round(float(power_watts * elapsed_sec), 4)


def calculate_estimated_energy_watt_hours(joules: float) -> float:
    # Submetric 7.2: Estimated Energy Consumption in Watt-hours (Wh). Lower is better.
    return round(float(joules / 3600.0), 6)


# -----------------------------------------------------------------------------
# Parameter 8: Operational Recovery Cost Formulations
# -----------------------------------------------------------------------------

def calculate_slo_violation_rate(total_tasks: int, violated_tasks: int) -> float:
    # Submetric 8.2: SLA / SLO Latency Violation Ratio (%). Lower is better.
    if total_tasks <= 0:
        return 0.0
    return round(float(violated_tasks / total_tasks) * 100.0, 2)


def calculate_operational_recovery_cost(
    total_actions: int,
    total_tasks: int,
    slo_violations: int,
    w_action: float = 0.4,
    w_slo: float = 0.4,
    w_overhead: float = 0.2,
    overhead_ratio: float = 0.0
) -> float:
    # Submetric 8.3: Normalized Operational Recovery Cost (Dimensionless Penalty Score).
    # Lower is better.
    act_ratio = float(total_actions / max(1, total_tasks))
    slo_ratio = float(slo_violations / max(1, total_tasks))
    score = w_action * act_ratio + w_slo * slo_ratio + w_overhead * overhead_ratio
    return round(float(score), 4)


# -----------------------------------------------------------------------------
# Parameter 9: Reliability & Availability Formulations
# -----------------------------------------------------------------------------

def calculate_service_availability(total_operational_sec: float, total_downtime_sec: float) -> float:
    # Submetric 9.1: Service Availability (%). Higher is better.
    if total_operational_sec <= 0.0:
        return 0.0
    uptime = max(0.0, total_operational_sec - total_downtime_sec)
    return round(float(uptime / total_operational_sec) * 100.0, 2)


def calculate_task_reliability(total_tasks: int, successful_tasks: int) -> float:
    # Submetric 9.2: Task Completion Reliability (%). Higher is better.
    if total_tasks <= 0:
        return 0.0
    return round(float(successful_tasks / total_tasks) * 100.0, 2)


def calculate_mttf_mttr(
    healthy_time_sec: float,
    repair_time_sec: float,
    fault_count: int
) -> Tuple[float, float]:
    # Submetric 9.3: Mean Time To Failure (MTTF, Higher is better)
    # and Mean Time To Repair (MTTR, Lower is better) in seconds.
    if fault_count <= 0:
        return round(float(healthy_time_sec), 2), 0.0
    mttf = round(float(healthy_time_sec / fault_count), 2)
    mttr = round(float(repair_time_sec / fault_count), 2)
    return mttf, mttr


# -----------------------------------------------------------------------------
# Parameter 10: Scalability Formulations
# -----------------------------------------------------------------------------

def calculate_complexity_scaling_exponent(scale_points: List[float], execution_times: List[float]) -> float:
    # Submetric 10.3: Complexity Scaling Exponent alpha.
    # Empirical log-log slope: log(T) = alpha * log(Scale) + c.
    # Lower is better (alpha <= 1.0 indicates sub-linear or linear scalability).
    clean_pairs = [
        (math.log(x), math.log(y))
        for x, y in zip(scale_points, execution_times)
        if x > 0 and y > 0 and not math.isnan(x) and not math.isnan(y)
    ]
    if len(clean_pairs) < 2:
        return 1.0

    log_x = [p[0] for p in clean_pairs]
    log_y = [p[1] for p in clean_pairs]

    slope, _ = np.polyfit(log_x, log_y, 1)
    return round(float(slope), 4)


# -----------------------------------------------------------------------------
# Standardized Result Record Builder
# -----------------------------------------------------------------------------

def build_result_record(
    algorithm: str,
    paper_id: str,
    metric: str,
    submetric: str,
    value: Optional[float],
    unit: str,
    measurement_type: MeasurementType = MeasurementType.MEASURED,
    comparability: Comparability = Comparability.DIRECT,
    run_values: Optional[List[float]] = None,
    dataset: str = "edge_iiotset",
    dataset_split: str = "test",
    run_id: str = "run_001",
    seed: Optional[int] = 42,
    configuration: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None,
) -> BenchmarkResultRecord:
    # Constructs a validated BenchmarkResultRecord with mean, std, and CI.
    mean_val = None
    std_val = None
    ci_val = None

    if measurement_type == MeasurementType.NR:
        val = None
    elif run_values and len(run_values) > 1:
        mean_val, std_val = calculate_mean_std(run_values)
        ci_val = calculate_confidence_interval(run_values)
        val = mean_val
    else:
        val = value
        if val is not None:
            mean_val = val
            std_val = 0.0
            ci_val = [val, val]

    return BenchmarkResultRecord(
        algorithm=algorithm,
        paper_id=paper_id,
        metric=metric,
        submetric=submetric,
        value=val,
        unit=unit,
        mean=mean_val,
        std=std_val,
        confidence_interval=ci_val,
        dataset=dataset,
        dataset_split=dataset_split,
        run_id=run_id,
        seed=seed,
        measurement_type=measurement_type,
        comparability=comparability,
        configuration=configuration or {},
        notes=notes,
    )
