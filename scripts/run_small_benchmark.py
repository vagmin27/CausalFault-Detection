# Small Validation Benchmark Runner (Phase 6B).
# Executes a 2,000-record streaming evaluation on Edge-IIoTset:
# - 1,000 warm-up records (excluded from metrics)
# - 1,000 evaluation records
# - Standardized SystemInstrumentation
# - Complete algorithm isolation per run

import os
import sys
import json
import time

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from typing import Dict, Any

from algorithms import (
    CausalFaultTolerancePipeline,
    IPFTAlgorithm,
    BWOAIAlgorithm,
    RCDAlgorithm,
    PreGANAlgorithm,
)
from evaluation.data_harness import CommonDataHarness, StreamConfig
from evaluation.instrumentation import SystemInstrumentation


def load_training_sample(n_rows: int = 500) -> np.ndarray:
    train_path = os.path.join("data", "processed", "train", "train_01_processed.csv")
    with open(os.path.join("data", "processed", "artifacts", "feature_names.json")) as f:
        feature_names = json.load(f)
    df = pd.read_csv(train_path, nrows=n_rows)
    return df[feature_names].values


def run_benchmark_for_algorithm(algo_cls, name: str, train_data: np.ndarray) -> Dict[str, Any]:
    print(f"\n--- Running Small Benchmark for: {name} ---")
    config = StreamConfig(
        dataset_name="edge_iiotset",
        split="test",
        data_dir="data/processed/test",
        warmup_count=1000,
        max_records=2000,
        batch_size=1000,
        seed=42,
    )
    harness = CommonDataHarness(config)

    algo = algo_cls()
    algo.initialize()
    print(f"Fitting {name} on {len(train_data)} normal training samples...")
    t_fit_start = time.perf_counter()
    algo.fit(train_data)
    fit_duration_sec = time.perf_counter() - t_fit_start
    print(f"Fit completed in {fit_duration_sec:.2f}s.")

    instrumentation = SystemInstrumentation()
    algo.start()

    warmup_processed = 0
    eval_processed = 0
    detected_count = 0
    action_count = 0
    rca_count = 0

    instrumentation.start_session()

    for ctx in harness.stream_contexts():
        obs = ctx.observable_input

        if ctx.is_warmup:
            # Process warm-up without recording metrics
            algo.process(obs)
            warmup_processed += 1
            continue

        # Evaluated record
        t0 = time.perf_counter_ns()
        res = algo.process(obs)
        elapsed_ns = time.perf_counter_ns() - t0

        eval_processed += 1
        instrumentation.throughput_counter.record_observation(1)
        instrumentation.bandwidth_accountant.add_telemetry_bytes(len(obs.feature_vector) * 8)

        if res.is_anomaly:
            detected_count += 1

        mit = res.raw_output.get("mitigation")
        if mit and mit.action_type not in ["NONE", "NO_ACTION"]:
            action_count += 1
            if mit.state_bytes_transferred > 0:
                instrumentation.bandwidth_accountant.add_migration_state_bytes(mit.state_bytes_transferred)
            instrumentation.action_counter.record_action(
                "MIGRATION" if "MIGRATION" in mit.action_type else "REBALANCE"
            )

        diag = res.raw_output.get("diagnosis")
        if diag:
            rca_count += 1

    summary = instrumentation.end_session()
    algo.stop()
    algo.reset()

    results = {
        "algorithm": name,
        "paper_id": algo.paper_id,
        "warmup_records": warmup_processed,
        "eval_records": eval_processed,
        "fit_time_sec": round(fit_duration_sec, 3),
        "elapsed_seconds": summary["elapsed_seconds"],
        "throughput_rec_sec": summary["throughput_rec_sec"],
        "avg_cpu_percent": summary["avg_cpu_percent"],
        "peak_rss_mb": summary["peak_rss_mb"],
        "bandwidth_kb_sec": summary["bandwidth_kb_sec"],
        "estimated_energy_joules": summary["estimated_energy_joules"],
        "detections": detected_count,
        "mitigations": action_count,
        "rca_diagnoses": rca_count,
        "total_actions": summary["total_actions"],
    }
    print(f"Results for {name}:")
    print(json.dumps(results, indent=2))
    return results


def main():
    train_data = load_training_sample(500)

    algorithms = [
        (CausalFaultTolerancePipeline, "Proposed Causal FT"),
        (IPFTAlgorithm, "IPFT (Theodoropoulos et al., 2022)"),
        (BWOAIAlgorithm, "BWOAIF (Hannák et al., 2023)"),
        (RCDAlgorithm, "RCD (Ikram et al., 2022)"),
        (PreGANAlgorithm, "PreGAN (Tuli et al., 2022)"),
    ]

    all_results = {}
    for algo_cls, name in algorithms:
        res = run_benchmark_for_algorithm(algo_cls, name, train_data)
        all_results[algo_cls().paper_id] = res

    out_path = os.path.join("results", "raw", "small_benchmark_summary.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll small benchmarks completed successfully. Saved to {out_path}.")


if __name__ == "__main__":
    main()
