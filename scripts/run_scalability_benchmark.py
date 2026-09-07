"""
Scalability Benchmark Runner (Phase 7 Part P).
Evaluates:
1. Feature Dimensional Scalability: D in {10, 25, 50, 62}
2. System / Workload Scalability: M in {5, 10, 20, 50} emulated edge nodes/workloads
Fits empirical scaling exponent alpha: T ~ c * D^alpha, T ~ c * M^alpha.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
torch.set_num_threads(1)

from algorithms import (
    CausalFaultTolerancePipeline,
    IPFTAlgorithm,
    BWOAIAlgorithm,
    RCDAlgorithm,
    PreGANAlgorithm,
)
from evaluation.context import BenchmarkInput
from evaluation.metrics import calculate_complexity_scaling_exponent


def generate_mock_stream(n_records: int, dim: int) -> List[BenchmarkInput]:
    stream = []
    for i in range(n_records):
        vec = np.random.normal(0, 1, size=dim)
        features = {f"feat_{j}": float(vec[j]) for j in range(dim)}
        inp = BenchmarkInput(
            stream_position=i,
            timestamp=float(i),
            timestamp_str="",
            device_id=f"device_{i % 5}",
            edge_node_id=f"node_{i % 5}",
            features=features,
            feature_vector=vec,
        )
        stream.append(inp)
    return stream


def benchmark_feature_scaling():
    print("--- Running Feature Dimensional Scalability (D = 10, 25, 50, 62) ---")
    dims = [10, 25, 50, 62]
    n_recs = 1000

    algorithms = [
        ("proposed_causal_ft", "Proposed Causal FT", CausalFaultTolerancePipeline),
        ("paper1_ipft", "IPFT", IPFTAlgorithm),
        ("paper2_bwoaif", "BWOAIF", BWOAIAlgorithm),
        ("paper3_rcd", "RCD", RCDAlgorithm),
        ("paper4_pregan", "PreGAN", PreGANAlgorithm),
    ]

    results = {}

    for algo_id, name, algo_cls in algorithms:
        latencies_by_dim = []
        throughputs_by_dim = []

        for d in dims:
            stream = generate_mock_stream(n_recs, d)
            cfg = {"feature_dim": d} if algo_id in ("paper1_ipft", "paper4_pregan") else None
            algo = algo_cls(config=cfg) if cfg else algo_cls()
            algo.initialize()
            X_train = np.random.normal(0, 1, size=(200, d))
            algo.fit(X_train)
            algo.start()

            t0 = time.perf_counter()
            for inp in stream:
                algo.process(inp)
            elapsed = time.perf_counter() - t0
            algo.stop()
            algo.reset()

            rec_per_sec = n_recs / max(1e-4, elapsed)
            lat_ms = (elapsed / n_recs) * 1000.0
            throughputs_by_dim.append(round(rec_per_sec, 2))
            latencies_by_dim.append(round(lat_ms, 4))

        # Fit scaling exponent alpha
        alpha_val = calculate_complexity_scaling_exponent(dims, latencies_by_dim)

        results[algo_id] = {
            "name": name,
            "dimension_levels": dims,
            "latencies_ms": latencies_by_dim,
            "throughputs_rec_sec": throughputs_by_dim,
            "empirical_scaling_exponent_alpha": alpha_val,
        }
        print(f"[{name}] Feature scaling exponent alpha: {alpha_val}")

    return results


def benchmark_system_scaling():
    print("\n--- Running System Workload Scalability (M = 5, 10, 20, 50 nodes) ---")
    node_levels = [5, 10, 20, 50]
    recs_per_node = 200

    algorithms = [
        ("proposed_causal_ft", "Proposed Causal FT", CausalFaultTolerancePipeline),
        ("paper1_ipft", "IPFT", IPFTAlgorithm),
        ("paper2_bwoaif", "BWOAIF", BWOAIAlgorithm),
        ("paper3_rcd", "RCD", RCDAlgorithm),
        ("paper4_pregan", "PreGAN", PreGANAlgorithm),
    ]

    results = {}

    for algo_id, name, algo_cls in algorithms:
        latencies_by_m = []
        throughputs_by_m = []

        for m in node_levels:
            total_recs = m * recs_per_node
            stream = generate_mock_stream(total_recs, 62)
            algo = algo_cls()
            algo.initialize()
            algo.fit(np.random.normal(0, 1, size=(200, 62)))
            algo.start()

            t0 = time.perf_counter()
            for inp in stream:
                algo.process(inp)
            elapsed = time.perf_counter() - t0
            algo.stop()
            algo.reset()

            rec_per_sec = total_recs / max(1e-4, elapsed)
            lat_ms = (elapsed / total_recs) * 1000.0
            throughputs_by_m.append(round(rec_per_sec, 2))
            latencies_by_m.append(round(lat_ms, 4))

        alpha_val = calculate_complexity_scaling_exponent(node_levels, latencies_by_m)

        results[algo_id] = {
            "name": name,
            "node_levels": node_levels,
            "latencies_ms": latencies_by_m,
            "throughputs_rec_sec": throughputs_by_m,
            "empirical_scaling_exponent_alpha": alpha_val,
        }
        print(f"[{name}] System scaling exponent alpha: {alpha_val}")

    return results


def main():
    feat_res = benchmark_feature_scaling()
    sys_res = benchmark_system_scaling()

    combined = {
        "feature_scaling_D": feat_res,
        "system_scaling_M": sys_res,
    }

    out_path = os.path.join("results", "raw", "scalability_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nScalability Benchmark successfully saved to {out_path}.")


if __name__ == "__main__":
    main()
