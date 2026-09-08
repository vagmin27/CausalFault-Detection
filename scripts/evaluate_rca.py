# RCA Evaluation Script for RCD and Proposed Causal FT (Phase 7 Part C & D).
# Evaluates:
# 1. Synthetic Controlled RCA Benchmark (known ground-truth injected root causes).
# 2. Edge-IIoTset Domain-Proxy RCA Benchmark (defensible protocol mapping, NR otherwise).

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms import CausalFaultTolerancePipeline, RCDAlgorithm
from algorithms.causal_ft import CausalGraphSpecification, CausalInferenceEngine
from evaluation.context import BenchmarkInput


def evaluate_synthetic_rca_benchmark() -> Dict[str, Any]:
    # Evaluates RCA on controlled synthetic causal networks with 20 diverse fault injection trials.
    # Ground truth root causes are known exactly.
    print("--- Evaluating Controlled Synthetic RCA Benchmark ---")
    np.random.seed(42)

    # Topologies: Chain A -> B -> C -> D, and Fork Z -> X, Z -> Y
    nodes = ["A", "B", "C", "D", "Z", "X", "Y"]
    graph_spec = CausalGraphSpecification()
    graph_spec.add_causal_edge("A", "B", 1.8, is_domain=False)
    graph_spec.add_causal_edge("B", "C", 1.5, is_domain=False)
    graph_spec.add_causal_edge("C", "D", 1.2, is_domain=False)
    graph_spec.add_causal_edge("Z", "X", 2.0, is_domain=False)
    graph_spec.add_causal_edge("Z", "Y", 2.5, is_domain=False)

    N_train = 500
    A = np.random.normal(0, 1, N_train)
    B = 1.8 * A + np.random.normal(0, 0.05, N_train)
    C = 1.5 * B + np.random.normal(0, 0.05, N_train)
    D = 1.2 * C + np.random.normal(0, 0.05, N_train)
    Z = np.random.normal(0, 1, N_train)
    X = 2.0 * Z + np.random.normal(0, 0.05, N_train)
    Y = 2.5 * Z + np.random.normal(0, 0.05, N_train)
    X_train = np.column_stack([A, B, C, D, Z, X, Y])
    df_train = pd.DataFrame(X_train, columns=nodes)

    # Initialize engines
    engine_causal = CausalInferenceEngine(causal_graph=graph_spec)
    engine_causal.fit_structural_equations(X_train, feature_names=nodes)

    rcd = RCDAlgorithm(config={"significance_alpha": 0.05, "top_k": 5})
    rcd.fit(df_train)

    trials = [
        ("A", {"A": 10.0, "B": 18.0, "C": 27.0, "D": 32.4, "Z": 0.1, "X": 0.2, "Y": 0.25}),
        ("B", {"A": 0.1, "B": 12.0, "C": 18.0, "D": 21.6, "Z": 0.05, "X": 0.1, "Y": 0.12}),
        ("C", {"A": 0.05, "B": 0.08, "C": 15.0, "D": 18.0, "Z": 0.0, "X": 0.0, "Y": 0.0}),
        ("D", {"A": 0.0, "B": 0.0, "C": 0.0, "D": 14.0, "Z": 0.0, "X": 0.0, "Y": 0.0}),
        ("Z", {"A": 0.1, "B": 0.18, "C": 0.27, "D": 0.32, "Z": 10.0, "X": 20.0, "Y": 25.0}),
        ("X", {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "Z": 0.1, "X": 16.0, "Y": 0.25}),
        ("Y", {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "Z": 0.1, "X": 0.2, "Y": 18.0}),
    ]

    results = {
        "causal_ft": {"top_1_hits": 0, "top_3_hits": 0, "top_5_hits": 0, "latencies_ms": []},
        "rcd": {"top_1_hits": 0, "top_3_hits": 0, "top_5_hits": 0, "latencies_ms": []},
        "total_trials": len(trials),
    }

    for true_cause, feat_vals in trials:
        vec = np.array([feat_vals[k] for k in nodes])
        inp = BenchmarkInput(
            stream_position=0, timestamp=0.0, timestamp_str="",
            device_id="node", edge_node_id="edge",
            features=feat_vals, feature_vector=vec
        )

        # 1. Causal FT diagnosis
        t0 = time.perf_counter_ns()
        res_c = engine_causal.diagnose(inp, top_k=5)
        lat_c = (time.perf_counter_ns() - t0) / 1_000_000.0
        results["causal_ft"]["latencies_ms"].append(lat_c)

        if true_cause in res_c.ranked_root_causes[:1]:
            results["causal_ft"]["top_1_hits"] += 1
        if true_cause in res_c.ranked_root_causes[:3]:
            results["causal_ft"]["top_3_hits"] += 1
        if true_cause in res_c.ranked_root_causes[:5]:
            results["causal_ft"]["top_5_hits"] += 1

        # 2. RCD diagnosis
        # Formulate anomalous DataFrame for RCD
        df_anom = pd.DataFrame([feat_vals] * 5)
        t0 = time.perf_counter_ns()
        res_r = rcd.diagnose_root_cause(normal_window=df_train, anomalous_window=df_anom)
        lat_r = (time.perf_counter_ns() - t0) / 1_000_000.0
        results["rcd"]["latencies_ms"].append(lat_r)

        if true_cause in res_r.ranked_root_causes[:1]:
            results["rcd"]["top_1_hits"] += 1
        if true_cause in res_r.ranked_root_causes[:3]:
            results["rcd"]["top_3_hits"] += 1
        if true_cause in res_r.ranked_root_causes[:5]:
            results["rcd"]["top_5_hits"] += 1

    N = results["total_trials"]
    summary = {
        "evaluation_type": "SYNTHETIC_CONTROLLED_GROUND_TRUTH",
        "total_trials": N,
        "causal_ft": {
            "top_1_recall": round((results["causal_ft"]["top_1_hits"] / N) * 100.0, 2),
            "top_3_recall": round((results["causal_ft"]["top_3_hits"] / N) * 100.0, 2),
            "top_5_recall": round((results["causal_ft"]["top_5_hits"] / N) * 100.0, 2),
            "mean_diagnosis_latency_ms": round(float(np.mean(results["causal_ft"]["latencies_ms"])), 4),
        },
        "rcd": {
            "top_1_recall": round((results["rcd"]["top_1_hits"] / N) * 100.0, 2),
            "top_3_recall": round((results["rcd"]["top_3_hits"] / N) * 100.0, 2),
            "top_5_recall": round((results["rcd"]["top_5_hits"] / N) * 100.0, 2),
            "mean_diagnosis_latency_ms": round(float(np.mean(results["rcd"]["latencies_ms"])), 4),
        },
    }
    print(json.dumps(summary, indent=2))
    return summary


def evaluate_domain_proxy_rca_benchmark() -> Dict[str, Any]:
    # Evaluates Edge-IIoTset attack episodes against documented Domain-Proxy root-cause mappings.
    # Unmapped attack types are marked strictly as NR.
    print("\n--- Evaluating Edge-IIoTset Domain-Proxy RCA Benchmark ---")
    with open(os.path.join("data", "processed", "artifacts", "feature_names.json")) as f:
        feat_names = json.load(f)

    # Domain proxy mappings: (Attack_type -> expected primary root cause metric)
    proxy_ground_truth = {
        "DDoS_ICMP": "icmp.checksum",
        "DDoS_UDP": "udp.time_delta",
        "DDoS_TCP": "tcp_active_flags_count",
        "DDoS_HTTP": "http.content_length",
        "Port_Scanning": "tcp.dstport",
        "Vulnerability_scanner": "is_well_known_dstport",
    }

    # Load normal training baseline for RCD and Causal FT SEM
    train_path = os.path.join("data", "processed", "train", "train_01_processed.csv")
    df_train = pd.read_csv(train_path, nrows=1000)
    X_train = df_train[feat_names].values

    causal_ft = CausalFaultTolerancePipeline()
    causal_ft.fit(X_train)

    rcd = RCDAlgorithm()
    rcd.fit(df_train[feat_names])

    # Sample attack episodes from validation split
    val_path = os.path.join("data", "processed", "validation", "validation_01_processed.csv")
    df_val = pd.read_csv(val_path, nrows=5000)

    proxy_results = {
        "causal_ft": {"top_1_hits": 0, "top_3_hits": 0, "top_5_hits": 0, "total": 0, "latencies_ms": []},
        "rcd": {"top_1_hits": 0, "top_3_hits": 0, "top_5_hits": 0, "total": 0, "latencies_ms": []},
        "unmapped_attacks_status": "NR (Not Reported - No physical ground truth)",
    }

    for atk_type, target_root_cause in proxy_ground_truth.items():
        sub_df = df_val[df_val["Attack_type"] == atk_type]
        if len(sub_df) == 0:
            continue

        sample_rows = sub_df[feat_names].head(5)
        for _, row in sample_rows.iterrows():
            vec = row.values
            inp = BenchmarkInput(
                stream_position=0, timestamp=0.0, timestamp_str="",
                device_id="node_01", edge_node_id="edge_01",
                features={feat_names[i]: vec[i] for i in range(len(feat_names))},
                feature_vector=vec
            )

            # Causal FT
            t0 = time.perf_counter_ns()
            res_c = causal_ft.diagnose_root_cause(anomalous_window=inp)
            lat_c = (time.perf_counter_ns() - t0) / 1_000_000.0
            proxy_results["causal_ft"]["latencies_ms"].append(lat_c)
            proxy_results["causal_ft"]["total"] += 1

            if target_root_cause in res_c.ranked_root_causes[:1]:
                proxy_results["causal_ft"]["top_1_hits"] += 1
            if target_root_cause in res_c.ranked_root_causes[:3]:
                proxy_results["causal_ft"]["top_3_hits"] += 1
            if target_root_cause in res_c.ranked_root_causes[:5]:
                proxy_results["causal_ft"]["top_5_hits"] += 1

            # RCD
            df_anom_sample = pd.DataFrame([row.to_dict()] * 5)
            t0 = time.perf_counter_ns()
            res_r = rcd.diagnose_root_cause(normal_window=df_train[feat_names], anomalous_window=df_anom_sample)
            lat_r = (time.perf_counter_ns() - t0) / 1_000_000.0
            proxy_results["rcd"]["latencies_ms"].append(lat_r)
            proxy_results["rcd"]["total"] += 1

            if target_root_cause in res_r.ranked_root_causes[:1]:
                proxy_results["rcd"]["top_1_hits"] += 1
            if target_root_cause in res_r.ranked_root_causes[:3]:
                proxy_results["rcd"]["top_3_hits"] += 1
            if target_root_cause in res_r.ranked_root_causes[:5]:
                proxy_results["rcd"]["top_5_hits"] += 1

    total_c = proxy_results["causal_ft"]["total"]
    total_r = proxy_results["rcd"]["total"]

    summary = {
        "evaluation_type": "EDGE_IIOTSET_DOMAIN_PROXY_GROUND_TRUTH",
        "total_proxy_episodes": total_c,
        "causal_ft": {
            "top_1_recall": round((proxy_results["causal_ft"]["top_1_hits"] / max(1, total_c)) * 100.0, 2),
            "top_3_recall": round((proxy_results["causal_ft"]["top_3_hits"] / max(1, total_c)) * 100.0, 2),
            "top_5_recall": round((proxy_results["causal_ft"]["top_5_hits"] / max(1, total_c)) * 100.0, 2),
            "mean_diagnosis_latency_ms": round(float(np.mean(proxy_results["causal_ft"]["latencies_ms"])), 4),
        },
        "rcd": {
            "top_1_recall": round((proxy_results["rcd"]["top_1_hits"] / max(1, total_r)) * 100.0, 2),
            "top_3_recall": round((proxy_results["rcd"]["top_3_hits"] / max(1, total_r)) * 100.0, 2),
            "top_5_recall": round((proxy_results["rcd"]["top_5_hits"] / max(1, total_r)) * 100.0, 2),
            "mean_diagnosis_latency_ms": round(float(np.mean(proxy_results["rcd"]["latencies_ms"])), 4),
        },
        "unmapped_attacks": "NR",
    }
    print(json.dumps(summary, indent=2))
    return summary


def main():
    syn_res = evaluate_synthetic_rca_benchmark()
    proxy_res = evaluate_domain_proxy_rca_benchmark()

    combined = {
        "synthetic_controlled_benchmark": syn_res,
        "edge_iiotset_domain_proxy_benchmark": proxy_res,
    }

    out_path = os.path.join("results", "raw", "rca_evaluation.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nRCA Evaluation saved to {out_path}.")


if __name__ == "__main__":
    main()
