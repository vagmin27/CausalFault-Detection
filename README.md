"""
# Causal Real-Time Adaptive Fault Tolerance for Edge-IoT Systems

This research project implements a **Causal Real-Time Adaptive Fault Tolerance** framework for Edge-IoT systems. The architecture combines real-time streaming anomaly detection (River & PyTorch Autoencoders), NetworkX system dependency DAGs, DoWhy causal effect estimation, and adaptive recovery mechanisms to autonomously detect system faults, identify their root causes, and execute optimal recovery operations.

---

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────┐
│                       DATA SOURCES                      │
│                                                         │
│   SimPy Edge Simulation │ TON_IoT │ Edge-IIoTset │ N-BaIoT│
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
                  Data Source Adapters
                             │
                             ▼
                Unified TelemetryRecord Format
                             │
                             ▼
                    Streaming Pipeline
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
         River Detector            PyTorch Detector
      (Half-Space Trees)             (Autoencoder)
               │                           │
               └─────────────┬─────────────┘
                             ▼
                      Fault Detected
                             │
                             ▼
                    Causal Analysis Engine
                    (NetworkX + DoWhy)
                             │
                             ▼
                     Root Cause Identified
                             │
                             ▼
                     Adaptive Recovery
             (Workload Redistribution, Task Migration,
              Traffic Rerouting, Failover, Rebalancing)
                             │
                             ▼
                    System State Update
                             │
                             ▼
                  Evaluation & Plots Output
```

---

## Causal Assumptions & Confounder Rationale

The domain causal relationships are modeled in `SystemCausalGraph` (`causal/causal_graph.py`):

```text
  Workload ──► CPU Utilization ──► Processing Latency
  Workload ──► Memory Utilization ──► Processing Latency
  Network Utilization ──► Packet Loss ──► Processing Latency
```

### Confounder / Backdoor Adjustment Rationale:
- When estimating the Average Causal Effect (ACE) of treatment **CPU Utilization** on **Latency**, `workload` acts as a common cause (confounder) because `workload` drives both CPU utilization and overall processing latency.
- DoWhy's backdoor criterion identifies `workload` as the adjustment set ($Z = \{\text{workload}\}$). Conditioning on `workload` blocks the backdoor path $\text{CPU} \leftarrow \text{Workload} \rightarrow \text{Latency}$, yielding an unconfounded causal estimate.
- Similarly, for treatment **Network Utilization** on **Packet Loss**, `workload` is included in the backdoor adjustment set.

---

## Operational Commands

### 1. Single Simulation Mode
```bash
python main.py --mode simulation --seed 42 --persistence 3
```

### 2. Multi-Seed Robustness Experiment (10 Seeds: 42 to 51)
Runs `BASELINE_NO_FAULT_TOLERANCE`, `BASELINE_FIXED_RECOVERY`, and `PROPOSED_CAUSAL_ADAPTIVE_RECOVERY` independently across 10 random seeds with error bar reporting:
```bash
python main.py --mode experiment --seed-start 42 --num-seeds 10
```

### 3. Detection Persistence Sensitivity Analysis
Evaluates consecutive anomaly persistence values ($k = 1, 2, 3, 4, 5$) to quantify the Precision/Recall/Delay tradeoff:
```bash
python main.py --mode sensitivity
```

### 4. Dataset Mode (Pluggable Datasets)
```bash
python main.py --mode dataset --dataset ton_iot --data-path "path/to/TON_IoT.csv"
python main.py --mode dataset --dataset edge_iiotset --data-path "path/to/Edge-IIoTset.csv"
python main.py --mode dataset --dataset n_baiot --data-path "path/to/N-BaIoT.csv"
```

---

## Generated Results & Visualizations

Results are saved under `results/`:
- `results/data/metrics_summary.csv`: Single run evaluation metrics summary.
- `results/data/multi_seed_results.csv`: Detailed per-seed metrics across seeds 42-51.
- `results/data/multi_seed_aggregate.csv`: Aggregate statistics (Mean, Std Dev, Min, Max).
- `results/data/persistence_sensitivity.csv`: Precision, Recall, F1, FPR, and Delay vs Persistence $k$.
- `results/plots/multi_seed_latency.png`: Multi-seed average latency comparison (Mean ± Std).
- `results/plots/multi_seed_availability.png`: Multi-seed service availability comparison (Mean ± Std).
- `results/plots/multi_seed_f1.png`: Multi-seed F1-score comparison (Mean ± Std).
- `results/plots/multi_seed_recovery_success.png`: Multi-seed recovery success rate comparison (Mean ± Std).
- `results/plots/persistence_precision_recall.png`: Precision, Recall, F1 vs Persistence $k$.
- `results/plots/persistence_detection_delay.png`: Detection Delay vs Persistence $k$.
"""
