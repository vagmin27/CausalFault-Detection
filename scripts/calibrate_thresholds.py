# Validation-Only Threshold Calibration Script (Phase 7 Part B).
# Evaluates candidate thresholds on data/processed/validation/ ONLY.
# Selects operating thresholds maximizing validation F1.

import os
import sys
import json
import time
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Part A: Fair runtime configuration
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

from algorithms import (
    CausalFaultTolerancePipeline,
    IPFTAlgorithm,
    BWOAIAlgorithm,
    PreGANAlgorithm,
)
from evaluation.data_harness import CommonDataHarness, StreamConfig


def load_validation_data(n_records: int = 25000):
    # Loads validation telemetry records and true labels.
    print(f"Loading {n_records} records from data/processed/validation/...")
    with open(os.path.join("data", "processed", "artifacts", "feature_names.json")) as f:
        feature_names = json.load(f)

    val_path = os.path.join("data", "processed", "validation", "validation_01_processed.csv")
    df = pd.read_csv(val_path, nrows=n_records)
    X = df[feature_names].values
    y_true = df["Attack_label"].values
    return X, y_true, feature_names


def sweep_thresholds(y_true: np.ndarray, scores: np.ndarray, algo_name: str):
    # Sweeps candidate thresholds to select threshold maximizing validation F1.
    candidates = np.linspace(0.01, 0.99, 99)
    best_tau = 0.50
    best_f1 = -1.0
    best_p = 0.0
    best_r = 0.0

    # Calculate ROC-AUC and PR-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, scores))
    except Exception:
        pr_auc = 0.0

    for tau in candidates:
        preds = (scores >= tau).astype(int)
        # Calculate F1
        p = float(precision_score(y_true, preds, zero_division=0))
        r = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))

        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
            best_p = p
            best_r = r

    return {
        "algorithm": algo_name,
        "selected_threshold": round(best_tau, 4),
        "selection_rule": "maximize_validation_F1",
        "validation_precision": round(best_p, 4),
        "validation_recall": round(best_r, 4),
        "validation_f1": round(best_f1, 4),
        "validation_roc_auc": round(roc_auc, 4),
        "validation_pr_auc": round(pr_auc, 4),
        "score_min": round(float(np.min(scores)), 4),
        "score_mean": round(float(np.mean(scores)), 4),
        "score_max": round(float(np.max(scores)), 4),
    }


def main():
    X_val, y_true, feat_names = load_validation_data(25000)

    # Train baselines on training sample
    train_path = os.path.join("data", "processed", "train", "train_01_processed.csv")
    df_train = pd.read_csv(train_path, nrows=1000)
    X_train = df_train[feat_names].values

    algorithms = [
        ("Proposed Causal FT", CausalFaultTolerancePipeline()),
        ("IPFT (Theodoropoulos et al., 2022)", IPFTAlgorithm()),
        ("BWOAIF (Hannák et al., 2023)", BWOAIAlgorithm()),
        ("PreGAN (Tuli et al., 2022)", PreGANAlgorithm()),
    ]

    calibration_results = {}
    rows = []

    for name, algo in algorithms:
        print(f"\n--- Calibrating {name} ---")
        algo.initialize()
        algo.fit(X_train)
        algo.start()

        scores = []
        t0 = time.perf_counter()
        for i in range(len(X_val)):
            vec = X_val[i]
            # Construct BenchmarkInput for fair interface execution
            from evaluation.context import BenchmarkInput
            inp = BenchmarkInput(
                stream_position=i,
                timestamp=float(i),
                timestamp_str="",
                device_id="node_01",
                edge_node_id="edge_01",
                features={feat_names[j]: vec[j] for j in range(len(feat_names))},
                feature_vector=vec,
            )
            res = algo.detect(inp)
            scores.append(res.anomaly_score)

        elapsed = time.perf_counter() - t0
        print(f"Scored {len(scores)} records in {elapsed:.2f}s.")
        scores_arr = np.array(scores)

        res_dict = sweep_thresholds(y_true, scores_arr, name)
        res_dict["scoring_time_sec"] = round(elapsed, 2)
        print(f"Optimal Threshold for {name}: {res_dict['selected_threshold']} (F1: {res_dict['validation_f1']})")

        calibration_results[algo.paper_id] = res_dict
        rows.append(res_dict)
        algo.stop()
        algo.reset()

    # Save results
    os.makedirs(os.path.join("results", "tables"), exist_ok=True)
    json_path = os.path.join("results", "tables", "threshold_calibration.json")
    with open(json_path, "w") as f:
        json.dump(calibration_results, f, indent=2)

    csv_path = os.path.join("results", "tables", "threshold_calibration.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nThreshold calibration successfully completed. Saved to {json_path} and {csv_path}.")


if __name__ == "__main__":
    main()
