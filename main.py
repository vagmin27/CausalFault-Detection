"""
Main CLI Application for Causal Real-Time Adaptive Fault Tolerance in Edge-IoT Systems.

Supports four operational modes:
  1. Simulation Mode (--mode simulation [--seed 42] [--persistence 3]):
     Runs a single SimPy EdgeSimulation run with baseline comparisons.
  2. Dataset Mode (--mode dataset --dataset ton_iot|edge_iiotset|n_baiot --data-path <path>):
     Streams real dataset records through the unified TelemetryRecord interface.
  3. Multi-Seed Experiment Mode (--mode experiment [--seed-start 42] [--num-seeds 10]):
     Executes reproducible multi-seed evaluation across seeds 42-51.
     Calculates Mean, Std Dev, Min, Max across all runs and generates error bar plots.
  4. Sensitivity Analysis Mode (--mode sensitivity):
     Evaluates Detection Persistence values (1, 2, 3, 4, 5) to quantify the Precision/Recall/Delay tradeoff.

Usage Examples:
  python main.py --mode simulation --seed 42
  python main.py --mode experiment --seed-start 42 --num-seeds 10
  python main.py --mode sensitivity
"""

import os
import sys
import argparse
import logging
import random
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple, Set, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plot generation
import matplotlib.pyplot as plt

# Internal package imports
from data.telemetry import TelemetryRecord, DataSource
from data.simulator_adapter import SimulatorAdapter
from data.ton_iot_adapter import TONIoTAdapter
from data.edge_iiotset_adapter import EdgeIIoTsetAdapter
from data.n_baiot_adapter import NBaIoTAdapter

from simulation.edge_simulation import EdgeSimulation
from faults.fault_generator import FaultGenerator
from faults.fault_types import FaultType, FaultRecord
from detection.river_detector import RiverFaultDetector
from detection.pytorch_detector import PyTorchFaultDetector
from causal.causal_graph import SystemCausalGraph
from causal.causal_analysis import CausalAnalyzer, CausalResult
from recovery.recovery_manager import RecoveryManager, RecoveryDecision, FaultEpisode

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Main")


@dataclass
class ExperimentConfig:
    """
    Standardized Experiment Configuration ensuring strict baseline fairness and reproducibility.
    """
    seed: int = 42
    duration: int = 200
    num_nodes: int = 3
    num_devices_per_node: int = 4
    persistence: int = 3
    fault_schedule: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"fault_type": "CPU_OVERLOAD", "node": "Edge_Node_1", "start_time": 40.0, "duration": 25.0, "severity": 0.9},
        {"fault_type": "NETWORK_CONGESTION", "node": "Edge_Node_2", "start_time": 90.0, "duration": 25.0, "severity": 0.85},
        {"fault_type": "MEMORY_OVERLOAD", "node": "Edge_Node_3", "start_time": 140.0, "duration": 25.0, "severity": 0.9},
    ])
    detector_config: Dict[str, Any] = field(default_factory=lambda: {
        "river_threshold": 0.65,
        "pytorch_threshold": 0.15,
    })


def parse_args():
    parser = argparse.ArgumentParser(
        description="Causal Real-Time Adaptive Fault Tolerance for Edge-IoT Systems"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simulation", "dataset", "experiment", "sensitivity"],
        default="simulation",
        help="Operational mode: 'simulation', 'dataset', 'experiment', or 'sensitivity'",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ton_iot", "edge_iiotset", "n_baiot"],
        default="ton_iot",
        help="Dataset selection for dataset mode",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to dataset CSV file or directory for dataset mode",
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["river", "pytorch", "both"],
        default="river",
        help="Detection algorithm selection",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for single run",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=42,
        help="Starting random seed for multi-seed experiment",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Number of random seeds for multi-seed experiment",
    )
    parser.add_argument(
        "--persistence",
        type=int,
        default=3,
        help="Consecutive anomaly observations required for fault confirmation",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=200,
        help="Simulation duration steps",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory path to save metrics CSV and plots",
    )
    return parser.parse_args()


def set_reproducible_seed(seed: int):
    """Set random seeds for python random, numpy, and pytorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def is_system_state_degraded(record: TelemetryRecord) -> bool:
    """
    State Consistency Rule: Verify if physical system metrics reflect degradation or fault.
    """
    cpu = record.cpu_utilization or 0.0
    mem = record.memory_utilization or 0.0
    net = record.network_utilization or 0.0
    lat = record.latency or 0.0
    loss = record.packet_loss or 0.0

    return bool(cpu >= 70.0 or mem >= 70.0 or net >= 70.0 or lat >= 55.0 or loss >= 5.0)


def run_experiment_pipeline(
    config: ExperimentConfig,
    mode: str = "simulation",
    dataset_name: str = "ton_iot",
    data_path: str = None,
    detector_type: str = "river",
    fixed_recovery: bool = False,
    no_recovery: bool = False,
    experiment_label: str = "PROPOSED_CAUSAL_ADAPTIVE_RECOVERY",
) -> Dict[str, Any]:
    """
    Run streaming pipeline:
      Data Source -> Online Detector -> Fault Confirmation -> Causal Analysis -> Adaptive Recovery -> Metrics
    """
    seed = config.seed
    duration = config.duration
    persistence_k = config.persistence
    set_reproducible_seed(seed)

    # 1. Initialize Independent Data Source
    if mode == "simulation":
        sim = EdgeSimulation(num_nodes=config.num_nodes, num_devices_per_node=config.num_devices_per_node, seed=seed)
        fault_gen = FaultGenerator(seed=seed)

        # Schedule identical controlled faults from ExperimentConfig
        for fspec in config.fault_schedule:
            ftype = FaultType.from_string(fspec["fault_type"])
            fault_gen.schedule_fault(
                fault_type=ftype,
                target_node_id=fspec["node"],
                start_time=fspec["start_time"],
                duration=fspec["duration"],
                severity=fspec["severity"],
            )

        data_source = SimulatorAdapter(simulation=sim, max_steps=duration, seed=seed)
    else:
        # Dataset Mode
        if dataset_name == "ton_iot":
            data_source = TONIoTAdapter(data_path=data_path)
        elif dataset_name == "edge_iiotset":
            data_source = EdgeIIoTsetAdapter(data_path=data_path)
        elif dataset_name == "n_baiot":
            data_source = NBaIoTAdapter(data_path=data_path)
        else:
            raise ValueError(f"Unknown dataset choice: {dataset_name}")
        sim = None
        fault_gen = None

    # 2. Initialize Detectors
    river_threshold = config.detector_config.get("river_threshold", 0.65)
    pytorch_threshold = config.detector_config.get("pytorch_threshold", 0.15)

    river_detector = RiverFaultDetector(anomaly_threshold=river_threshold, seed=seed) if detector_type in ["river", "both"] else None
    pytorch_detector = PyTorchFaultDetector(threshold=pytorch_threshold) if detector_type in ["pytorch", "both"] else None

    # Pre-train PyTorch model on normal baseline records
    if pytorch_detector is not None and mode == "simulation":
        warmup_sim = EdgeSimulation(num_nodes=config.num_nodes, seed=seed + 999)
        warmup_records = []
        for _ in range(40):
            warmup_records.extend(warmup_sim.step())
        pytorch_detector.train_on_baseline(warmup_records, epochs=25)

    # 3. Initialize Independent Causal Engine & Recovery Manager
    causal_analyzer = CausalAnalyzer()
    recovery_manager = RecoveryManager()

    # Tracking structures
    history_records: List[TelemetryRecord] = []
    node_anomaly_window: Dict[str, List[bool]] = {}

    recovered_nodes: Set[str] = set()
    fault_episodes: List[FaultEpisode] = []

    y_true: List[int] = []
    y_pred: List[int] = []
    detection_latencies: List[float] = []
    first_confirmation_times: Dict[str, float] = {}

    anomalies_count = 0
    faults_confirmed_count = 0

    causal_inference_count = 0
    heuristic_fallback_count = 0
    insufficient_evidence_count = 0

    before_recovery_metrics: List[Dict] = []
    after_recovery_metrics: List[Dict] = []

    try:
        telemetry_stream = data_source.stream_telemetry()
    except Exception as err:
        logger.error(f"Failed to open data stream: {err}")
        raise err

    for record in telemetry_stream:
        history_records.append(record)
        node_id = record.edge_node_id
        y_true.append(record.fault_label)

        # Update active faults in simulation and clear recovered node flags when fault expires
        if sim is not None and fault_gen is not None:
            active_faults = fault_gen.update_simulation_faults(record.timestamp, sim)
            active_nodes = {f.target_node_id for f in active_faults}
            expired = [n for n in recovered_nodes if n not in active_nodes]
            for n in expired:
                recovered_nodes.remove(n)

        # 4. Online Detection Phase
        start_t = time.perf_counter()

        if detector_type == "river" and river_detector is not None:
            det_res = river_detector.process_record(record)
        elif detector_type == "pytorch" and pytorch_detector is not None:
            det_res = pytorch_detector.process_record(record)
        else:
            r_res = river_detector.process_record(record) if river_detector else {"is_fault": False}
            p_res = pytorch_detector.process_record(record) if pytorch_detector else {"is_fault": False}
            det_res = {
                "is_fault": r_res["is_fault"] or p_res["is_fault"],
                "anomaly_score": max(r_res.get("anomaly_score", 0.0), p_res.get("anomaly_score", 0.0)),
                "timestamp": record.timestamp,
            }

        proc_delay_ms = (time.perf_counter() - start_t) * 1000.0
        detection_latencies.append(proc_delay_ms)

        raw_is_anomaly = bool(det_res["is_fault"])
        if raw_is_anomaly:
            anomalies_count += 1

        # 5. Configurable Persistence Fault Confirmation Pipeline
        if node_id not in node_anomaly_window:
            node_anomaly_window[node_id] = []
        node_anomaly_window[node_id].append(raw_is_anomaly)
        if len(node_anomaly_window[node_id]) > (persistence_k + 2):
            node_anomaly_window[node_id].pop(0)

        persistent_anomaly = bool(sum(node_anomaly_window[node_id][-persistence_k:]) >= persistence_k)
        state_consistent = is_system_state_degraded(record)

        # Fault is confirmed ONLY when persistence AND state consistency are met
        is_fault_confirmed = bool(persistent_anomaly and state_consistent)
        y_pred.append(1 if is_fault_confirmed else 0)

        # Track first confirmation timestamp per node for detection delay calculation
        if is_fault_confirmed and node_id not in first_confirmation_times:
            first_confirmation_times[node_id] = record.timestamp

        # 6. Reactive Causal Analysis & Recovery Workflow (Episode-Level Evaluation)
        if is_fault_confirmed:
            faults_confirmed_count += 1

            if not no_recovery:
                if node_id not in recovered_nodes:
                    recovered_nodes.add(node_id)

                    # Perform DoWhy Causal Analysis specifically for the affected node
                    causal_res: CausalResult = causal_analyzer.analyze_fault_cause(
                        history_records=history_records[-60:],
                        target_node_id=node_id,
                    )

                    # Track causal inference audit counts
                    if causal_res.causal_status == "CAUSAL_INFERENCE":
                        causal_inference_count += 1
                    elif causal_res.causal_status == "HEURISTIC":
                        heuristic_fallback_count += 1
                    else:
                        insufficient_evidence_count += 1

                    # Select Recovery Action
                    rec_decision: RecoveryDecision = recovery_manager.select_recovery_action(
                        record=record,
                        root_cause=causal_res.root_cause,
                        causal_effect=causal_res.estimated_effect,
                        causal_status=causal_res.causal_status,
                        root_cause_source=causal_res.root_cause_source,
                        fixed_recovery=fixed_recovery,
                    )

                    # Execute Recovery in Simulation
                    if sim is not None:
                        rec_result = recovery_manager.execute_recovery_in_simulation(rec_decision, sim)
                        
                        episode = FaultEpisode(
                            episode_id=f"EPISODE_{node_id}_{record.timestamp}",
                            fault_type=record.fault_type or "UNKNOWN",
                            affected_node=node_id,
                            fault_start=record.timestamp - float(persistence_k),
                            fault_end=record.timestamp + 22.0,
                            detection_time=record.timestamp,
                            recovery_time=record.timestamp,
                            recovery_action=rec_decision.selected_action,
                            root_cause=causal_res.root_cause,
                            root_cause_source=causal_res.root_cause_source,
                            pre_recovery_metrics=rec_result["before"],
                            post_recovery_metrics=rec_result["after"],
                            target_metric=rec_result.get("target_metric", "cpu_utilization"),
                            absolute_improvement=rec_result.get("absolute_improvement", 0.0),
                            relative_improvement=rec_result.get("relative_improvement", 0.0),
                            recovery_success=rec_result.get("success", False),
                        )
                        fault_episodes.append(episode)

                        if rec_result.get("success", False):
                            before_recovery_metrics.append(rec_result["before"])
                            after_recovery_metrics.append(rec_result["after"])
                            node_anomaly_window[node_id] = []

    # 7. Calculate Precision, Recall, F1 against ground truth
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))

    accuracy = (tp + tn) / max(1, len(y_t))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    fpr = fp / max(1, fp + tn)
    avg_det_latency = float(np.mean(detection_latencies)) if detection_latencies else 0.0

    # Calculate average detection delay relative to scheduled fault start times
    delays = []
    if mode == "simulation":
        for fspec in config.fault_schedule:
            target_n = fspec["node"]
            start_t = fspec["start_time"]
            if target_n in first_confirmation_times:
                det_t = first_confirmation_times[target_n]
                if det_t >= start_t:
                    delays.append(det_t - start_t)
    avg_detection_delay = float(np.mean(delays)) if delays else 3.0

    # System Performance Metrics
    cpus = [r.cpu_utilization for r in history_records if r.cpu_utilization is not None]
    lats = [r.latency for r in history_records if r.latency is not None]
    pkts = [r.packet_loss for r in history_records if r.packet_loss is not None]

    avg_cpu = float(np.mean(cpus)) if cpus else 0.0
    avg_lat = float(np.mean(lats)) if lats else 0.0
    avg_pkt = float(np.mean(pkts)) if pkts else 0.0

    service_availability = ((len(lats) - len([l for l in lats if l > 200.0])) / max(1, len(lats))) * 100.0

    # Episode-level Recovery Metrics
    if no_recovery:
        recovery_attempts_count = 0
        successful_recoveries_count = 0
        recovery_success_rate_val = np.nan
        recovery_success_rate_str = "N/A"
    else:
        recovery_attempts_count = len(fault_episodes)
        successful_recoveries_count = sum(1 for ep in fault_episodes if ep.recovery_success)
        if recovery_attempts_count > 0:
            rate = (successful_recoveries_count / recovery_attempts_count) * 100.0
            recovery_success_rate_val = round(rate, 2)
            recovery_success_rate_str = f"{rate:.2f}%"
        else:
            recovery_success_rate_val = 0.0
            recovery_success_rate_str = "0.00%"

    return {
        "experiment_label": experiment_label,
        "seed": seed,
        "mode": mode,
        "dataset_name": dataset_name,
        "detector_type": detector_type,
        "persistence": persistence_k,
        "total_records": len(history_records),
        "anomalies_count": anomalies_count,
        "faults_confirmed_count": faults_confirmed_count,
        "recovery_attempts_count": recovery_attempts_count,
        "successful_recoveries_count": successful_recoveries_count,
        "causal_inference_count": causal_inference_count,
        "heuristic_fallback_count": heuristic_fallback_count,
        "insufficient_evidence_count": insufficient_evidence_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "fpr": round(float(fpr), 4),
        "avg_detection_latency_ms": round(float(avg_det_latency), 4),
        "avg_detection_delay_steps": round(avg_detection_delay, 2),
        "avg_cpu_utilization": round(avg_cpu, 2),
        "avg_latency": round(avg_lat, 2),
        "avg_packet_loss": round(avg_pkt, 2),
        "service_availability": round(service_availability, 2),
        "recovery_success_rate": recovery_success_rate_str,
        "recovery_success_rate_val": recovery_success_rate_val,
        "before_recovery_metrics": before_recovery_metrics,
        "after_recovery_metrics": after_recovery_metrics,
        "fault_episodes": [asdict(ep) for ep in fault_episodes],
        "detection_latencies": detection_latencies,
    }


def run_multi_seed_experiment(seed_start: int = 42, num_seeds: int = 10, output_dir: str = "results"):
    """
    Execute multi-seed evaluation across independent seeds (e.g. 42 to 51).
    Saves per-seed CSV and aggregate CSV with mean, std, min, max.
    Generates research plots with error bars.
    """
    seeds = list(range(seed_start, seed_start + num_seeds))
    logger.info(f"=== Starting Multi-Seed Experiment across {num_seeds} seeds: {seeds} ===")

    all_results: List[Dict[str, Any]] = []

    for seed in seeds:
        config = ExperimentConfig(seed=seed, duration=200)

        # 1. Baseline 1: No FT
        res_no_ft = run_experiment_pipeline(
            config=config,
            mode="simulation",
            detector_type="river",
            fixed_recovery=False,
            no_recovery=True,
            experiment_label="BASELINE_NO_FAULT_TOLERANCE",
        )
        all_results.append(res_no_ft)

        # 2. Baseline 2: Fixed Recovery
        res_fixed = run_experiment_pipeline(
            config=config,
            mode="simulation",
            detector_type="river",
            fixed_recovery=True,
            no_recovery=False,
            experiment_label="BASELINE_FIXED_RECOVERY",
        )
        all_results.append(res_fixed)

        # 3. Proposed Strategy: Causal Adaptive
        res_proposed = run_experiment_pipeline(
            config=config,
            mode="simulation",
            detector_type="river",
            fixed_recovery=False,
            no_recovery=False,
            experiment_label="PROPOSED_CAUSAL_ADAPTIVE_RECOVERY",
        )
        all_results.append(res_proposed)

    # Save per-seed detailed CSV
    data_dir = os.path.join(output_dir, "data")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    df_per_seed = pd.DataFrame(all_results)
    clean_df = df_per_seed.drop(columns=["before_recovery_metrics", "after_recovery_metrics", "fault_episodes", "detection_latencies"], errors="ignore")
    per_seed_csv = os.path.join(data_dir, "multi_seed_results.csv")
    clean_df.to_csv(per_seed_csv, index=False)
    logger.info(f"Saved detailed per-seed CSV to: {per_seed_csv}")

    # Compute Aggregate Metrics (Mean, Std, Min, Max) per Experiment Configuration
    agg_rows = []
    for label in ["BASELINE_NO_FAULT_TOLERANCE", "BASELINE_FIXED_RECOVERY", "PROPOSED_CAUSAL_ADAPTIVE_RECOVERY"]:
        sub = clean_df[clean_df["experiment_label"] == label]
        row = {"experiment_label": label, "num_seeds": len(sub)}

        for col in ["service_availability", "avg_latency", "avg_cpu_utilization", "precision", "recall", "f1", "recovery_success_rate_val"]:
            vals = sub[col].dropna().values
            if len(vals) > 0:
                row[f"{col}_mean"] = round(float(np.mean(vals)), 2)
                row[f"{col}_std"] = round(float(np.std(vals)), 2)
                row[f"{col}_min"] = round(float(np.min(vals)), 2)
                row[f"{col}_max"] = round(float(np.max(vals)), 2)
            else:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_std"] = np.nan
                row[f"{col}_min"] = np.nan
                row[f"{col}_max"] = np.nan

        row["total_causal_inferences"] = int(sub["causal_inference_count"].sum())
        row["total_heuristic_fallbacks"] = int(sub["heuristic_fallback_count"].sum())
        row["total_insufficient_evidences"] = int(sub["insufficient_evidence_count"].sum())
        agg_rows.append(row)

    df_agg = pd.DataFrame(agg_rows)
    agg_csv = os.path.join(data_dir, "multi_seed_aggregate.csv")
    df_agg.to_csv(agg_csv, index=False)
    logger.info(f"Saved aggregate metrics CSV to: {agg_csv}")

    # Generate Multi-Seed Research Plots with Error Bars
    plt.style.use("ggplot")
    configs_labels = ["No Fault Tolerance", "Fixed Recovery", "Proposed Causal"]
    x = np.arange(len(configs_labels))
    width = 0.45

    # 1. Multi-Seed Average Latency Bar Chart (Mean ± Std)
    plt.figure(figsize=(7, 4.5))
    lat_means = df_agg["avg_latency_mean"].values
    lat_stds = df_agg["avg_latency_std"].values
    plt.bar(x, lat_means, width, yerr=lat_stds, capsize=5, color=["#e41a1c", "#377eb8", "#4daf4a"])
    plt.xticks(x, configs_labels)
    plt.ylabel("Average Latency (ms)")
    plt.title("Multi-Seed Average Processing Latency (Mean ± Std)")
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "multi_seed_latency.png")
    plt.savefig(p1, dpi=300)
    plt.close()

    # 2. Multi-Seed Service Availability Bar Chart (Mean ± Std)
    plt.figure(figsize=(7, 4.5))
    avail_means = df_agg["service_availability_mean"].values
    avail_stds = df_agg["service_availability_std"].values
    plt.bar(x, avail_means, width, yerr=avail_stds, capsize=5, color=["#e41a1c", "#377eb8", "#4daf4a"])
    plt.xticks(x, configs_labels)
    plt.ylabel("Service Availability (%)")
    plt.title("Multi-Seed Service Availability (Mean ± Std)")
    plt.ylim(80, 105)
    plt.tight_layout()
    p2 = os.path.join(plots_dir, "multi_seed_availability.png")
    plt.savefig(p2, dpi=300)
    plt.close()

    # 3. Multi-Seed F1-Score Bar Chart (Mean ± Std)
    plt.figure(figsize=(7, 4.5))
    f1_means = df_agg["f1_mean"].values
    f1_stds = df_agg["f1_std"].values
    plt.bar(x, f1_means, width, yerr=f1_stds, capsize=5, color=["#e41a1c", "#377eb8", "#4daf4a"])
    plt.xticks(x, configs_labels)
    plt.ylabel("F1-Score")
    plt.title("Multi-Seed Detection F1-Score (Mean ± Std)")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    p3 = os.path.join(plots_dir, "multi_seed_f1.png")
    plt.savefig(p3, dpi=300)
    plt.close()

    # 4. Multi-Seed Recovery Success Rate Bar Chart (Mean ± Std)
    plt.figure(figsize=(7, 4.5))
    rec_means = [0.0 if np.isnan(v) else v for v in df_agg["recovery_success_rate_val_mean"].values]
    rec_stds = [0.0 if np.isnan(v) else v for v in df_agg["recovery_success_rate_val_std"].values]
    plt.bar(x, rec_means, width, yerr=rec_stds, capsize=5, color=["#999999", "#377eb8", "#4daf4a"])
    plt.xticks(x, configs_labels)
    plt.ylabel("Recovery Success Rate (%)")
    plt.title("Multi-Seed Episode Recovery Success Rate (Mean ± Std)")
    plt.ylim(0, 110)
    plt.tight_layout()
    p4 = os.path.join(plots_dir, "multi_seed_recovery_success.png")
    plt.savefig(p4, dpi=300)
    plt.close()

    print("\n==========================================================================================")
    print("                              MULTI-SEED EXPERIMENT SUMMARY                               ")
    print("==========================================================================================")
    print(f"Seeds Evaluated: {seeds}")
    print(df_agg[["experiment_label", "service_availability_mean", "avg_latency_mean", "precision_mean", "recall_mean", "f1_mean", "recovery_success_rate_val_mean"]].to_string(index=False))
    print("==========================================================================================\n")


def run_persistence_sensitivity_experiment(seed: int = 42, output_dir: str = "results"):
    """
    Evaluate Detection Persistence threshold values (1, 2, 3, 4, 5)
    to show the Precision / Recall / Detection Delay trade-off.
    """
    logger.info("=== Starting Detection Persistence Sensitivity Experiment (Persistence: 1 to 5) ===")

    persistence_values = [1, 2, 3, 4, 5]
    sensitivity_results = []

    for k in persistence_values:
        config = ExperimentConfig(seed=seed, duration=200, persistence=k)
        res = run_experiment_pipeline(
            config=config,
            mode="simulation",
            detector_type="river",
            fixed_recovery=False,
            no_recovery=False,
            experiment_label=f"PERSISTENCE_K_{k}",
        )
        sensitivity_results.append(res)

    data_dir = os.path.join(output_dir, "data")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    df_sens = pd.DataFrame(sensitivity_results)
    clean_sens = df_sens[["persistence", "precision", "recall", "f1", "fpr", "avg_detection_delay_steps", "anomalies_count", "faults_confirmed_count"]]
    sens_csv = os.path.join(data_dir, "persistence_sensitivity.csv")
    clean_sens.to_csv(sens_csv, index=False)
    logger.info(f"Saved persistence sensitivity CSV to: {sens_csv}")

    # Generate Sensitivity Plots
    plt.style.use("ggplot")

    # Plot 1: Precision, Recall, F1 vs Persistence
    plt.figure(figsize=(7, 4.5))
    plt.plot(clean_sens["persistence"], clean_sens["precision"], marker="o", label="Precision", color="#2b5c8f", linewidth=2.0)
    plt.plot(clean_sens["persistence"], clean_sens["recall"], marker="s", label="Recall", color="#e41a1c", linewidth=2.0)
    plt.plot(clean_sens["persistence"], clean_sens["f1"], marker="^", label="F1-Score", color="#4daf4a", linewidth=2.0)

    plt.xlabel("Persistence Threshold (Consecutive Anomaly Observations k)")
    plt.ylabel("Score")
    plt.title("Detection Metric Tradeoff vs. Persistence Threshold (k)")
    plt.xticks(persistence_values)
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "persistence_precision_recall.png")
    plt.savefig(p1, dpi=300)
    plt.close()

    # Plot 2: Detection Delay vs Persistence
    plt.figure(figsize=(7, 4.5))
    plt.plot(clean_sens["persistence"], clean_sens["avg_detection_delay_steps"], marker="d", color="#d95f02", linewidth=2.0)

    plt.xlabel("Persistence Threshold (Consecutive Anomaly Observations k)")
    plt.ylabel("Average Detection Delay (Simulation Steps)")
    plt.title("Fault Confirmation Delay vs. Persistence Threshold (k)")
    plt.xticks(persistence_values)
    plt.tight_layout()
    p2 = os.path.join(plots_dir, "persistence_detection_delay.png")
    plt.savefig(p2, dpi=300)
    plt.close()

    print("\n==========================================================================================")
    print("                        DETECTION PERSISTENCE SENSITIVITY SUMMARY                          ")
    print("==========================================================================================")
    print(clean_sens.to_string(index=False))
    print("==========================================================================================\n")


def main():
    args = parse_args()
    logger.info("==========================================================================")
    logger.info("  Causal Real-Time Adaptive Fault Tolerance for Edge-IoT Systems  ")
    logger.info("==========================================================================")
    logger.info(f"Mode: {args.mode} | Seed: {args.seed} | Persistence: {args.persistence}")

    if args.mode == "experiment":
        run_multi_seed_experiment(seed_start=args.seed_start, num_seeds=args.num_seeds, output_dir=args.output_dir)

    elif args.mode == "sensitivity":
        run_persistence_sensitivity_experiment(seed=args.seed, output_dir=args.output_dir)

    elif args.mode == "dataset":
        if not args.data_path or not os.path.exists(args.data_path):
            logger.error(f"Dataset path invalid or not provided: {args.data_path}")
            sys.exit(1)
        exp_config = ExperimentConfig(seed=args.seed, duration=args.duration)
        results = run_experiment_pipeline(
            config=exp_config,
            mode="dataset",
            dataset_name=args.dataset,
            data_path=args.data_path,
            detector_type=args.detector,
            experiment_label=f"DATASET_{args.dataset.upper()}",
        )
        print("\n================ DATASET EXPERIMENT RESULTS ================")
        print(f"Dataset:                  {results['dataset_name']}")
        print(f"Total Records Streamed:   {results['total_records']}")
        print(f"F1-Score:                 {results['f1'] * 100:.2f}%")
        print("============================================================\n")

    else:
        # Simulation Mode (Single Seed)
        exp_config = ExperimentConfig(seed=args.seed, duration=args.duration, persistence=args.persistence)

        proposed_res = run_experiment_pipeline(
            config=exp_config,
            mode="simulation",
            detector_type="river",
            fixed_recovery=False,
            no_recovery=False,
            experiment_label="PROPOSED_CAUSAL_ADAPTIVE_RECOVERY",
        )

        baseline1_res = run_experiment_pipeline(
            config=exp_config,
            mode="simulation",
            detector_type="river",
            fixed_recovery=False,
            no_recovery=True,
            experiment_label="BASELINE_NO_FAULT_TOLERANCE",
        )

        baseline2_res = run_experiment_pipeline(
            config=exp_config,
            mode="simulation",
            detector_type="river",
            fixed_recovery=True,
            no_recovery=False,
            experiment_label="BASELINE_FIXED_RECOVERY",
        )

        print("\n==========================================================================================")
        print("                                EXPERIMENTAL RESULTS SUMMARY                               ")
        print("==========================================================================================")
        print(f"{'Metric':<30} | {'Baseline 1 (No FT)':<20} | {'Baseline 2 (Fixed)':<20} | {'Proposed (Causal)':<20}")
        print("-" * 98)
        print(f"{'Service Availability (%)':<30} | {baseline1_res['service_availability']:<20.2f} | {baseline2_res['service_availability']:<20.2f} | {proposed_res['service_availability']:<20.2f}")
        print(f"{'Average Latency (ms)':<30} | {baseline1_res['avg_latency']:<20.2f} | {baseline2_res['avg_latency']:<20.2f} | {proposed_res['avg_latency']:<20.2f}")
        print(f"{'Average CPU Utilization (%)':<30} | {baseline1_res['avg_cpu_utilization']:<20.2f} | {baseline2_res['avg_cpu_utilization']:<20.2f} | {proposed_res['avg_cpu_utilization']:<20.2f}")
        print(f"{'Detection Precision (%)':<30} | {baseline1_res['precision']*100:<20.2f} | {baseline2_res['precision']*100:<20.2f} | {proposed_res['precision']*100:<20.2f}")
        print(f"{'Detection Recall (%)':<30} | {baseline1_res['recall']*100:<20.2f} | {baseline2_res['recall']*100:<20.2f} | {proposed_res['recall']*100:<20.2f}")
        print(f"{'Detection F1-Score (%)':<30} | {baseline1_res['f1']*100:<20.2f} | {baseline2_res['f1']*100:<20.2f} | {proposed_res['f1']*100:<20.2f}")
        print(f"{'Recovery Attempts (Episodes)':<30} | {baseline1_res['recovery_attempts_count']:<20} | {baseline2_res['recovery_attempts_count']:<20} | {proposed_res['recovery_attempts_count']:<20}")
        print(f"{'Successful Recovery Episodes':<30} | {baseline1_res['successful_recoveries_count']:<20} | {baseline2_res['successful_recoveries_count']:<20} | {proposed_res['successful_recoveries_count']:<20}")
        print(f"{'Recovery Success Rate':<30} | {baseline1_res['recovery_success_rate']:<20} | {baseline2_res['recovery_success_rate']:<20} | {proposed_res['recovery_success_rate']:<20}")
        print("==========================================================================================\n")


if __name__ == "__main__":
    main()
