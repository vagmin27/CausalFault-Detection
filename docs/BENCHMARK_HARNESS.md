# Common Benchmark Data Harness Specification

## Technical Design & Execution Contract for Edge-IIoTset Comparative Evaluation

---

## 1. Overview & Research Objective

The Common Benchmark Data Harness establishes a completely standardized, mathematically rigorous experimental stream for comparing:
1. **Proposed Causal Fault-Tolerance (Causal FT)**
2. **Paper 1: IPFT** (*Theodoropoulos et al., 2022*)
3. **Paper 2: BWOAIF** (*Hannák et al., 2023*)
4. **Paper 3: RCD** (*Ikram et al., 2022*)
5. **Paper 4: PreGAN** (*Tuli et al., 2022*)

**Guiding Axiom**: All five approaches must receive the **SAME** evaluation stream, **SAME** test records, **SAME** observation sequence, **SAME** preprocessing, **SAME** fault/event labels, and **SAME** runtime conditions wherever their specific task is applicable.

---

## 2. Common Data Source & Partitioning Protocol

The harness consumes the leakage-free, zero-imputation preprocessed **Edge-IIoTset** splits generated under `data/processed/`:

```text
data/processed/
├── train/          16 CSV files, 1,467,486 rows (Fitted StandardScaler, training vocabularies)
├── validation/      4 CSV files,   314,456 rows (Validation tuning / threshold selection)
├── test/            4 CSV files,   314,473 rows (Canonical common evaluation stream)
└── artifacts/
    ├── scaler.joblib              Fitted StandardScaler across 62 input features
    ├── feature_names.json         Canonical ordered list of 62 input features
    └── preprocessing_report.json  Partition statistics and dropped corrupted row counts
```

### Partitioning Rules:
- **Training Set (`train/`)**: Supervised models (e.g. PreGAN prototype encoder, IPFT multi-channel regression) may access training features and training labels solely during the offline training stage. Unsupervised models (BWOAIF) and localized causal discovery models (RCD) do not access labels during fitting.
- **Validation Set (`validation/`)**: Used exclusively for hyperparameter tuning (e.g. IPFT replication threshold grid search, BWOAIF $\sigma_v$ tuning). Validation data must never contaminate test evaluations.
- **Test Set (`test/`)**: Strictly unseen. No fitting, updating of scalers, or parameter tuning is permitted on test data.

---

## 3. Standard Test Stream Configuration

The canonical evaluation stream is configured via `StreamConfig` in [`evaluation/data_harness.py`](file:///c:/Users/Dell/Desktop/CausalFault-Detection/evaluation/data_harness.py):

- **Dataset**: `Edge-IIoTset`
- **Split**: `test` (4 CSV files in deterministic alphabetical order: `test_01_processed.csv` to `test_04_processed.csv`)
- **Features**: Exactly 62 standardized numerical features (verified by SHA-256 hash in manifest)
- **Warm-Up**: Exactly **1,000 records** (flagged with `is_warmup=True`)
- **Evaluation Records**: Remaining records following the warm-up window
- **Batch Size**: 5,000 records per streaming chunk (bounded memory usage $< 200\text{ MB}$)
- **Random Seed**: Fixed baseline seed `42` (with repeated runs across seeds `42, 43, 44, 45, 46`)

---

## 4. Ground-Truth Isolation & Observable Input Contract

To ensure 100% academic integrity and eliminate target leakage, the data harness completely decouples observable inputs from ground-truth verification:

```text
Stream Step (Row i)
     │
     ├─► BenchmarkInput (EXPOSED TO ALGORITHM)
     │     ├── stream_position: int
     │     ├── timestamp: float (epoch seconds)
     │     ├── timestamp_str: str (ISO format)
     │     ├── device_id: str (ip.src_host)
     │     ├── edge_node_id: str (ip.dst_host)
     │     ├── features: Dict[str, float] (62 features)
     │     └── feature_vector: np.ndarray (shape (62,), float64)
     │     [NO Attack_label, NO Attack_type, NO future records]
     │
     └─► EventGroundTruth (EXPOSED ONLY TO EVALUATION HARNESS)
           ├── stream_position: int
           ├── timestamp: float
           ├── is_fault: bool (Attack_label == 1)
           ├── fault_label: int (0 or 1)
           ├── raw_attack_type: str (e.g. "DDoS_ICMP", "Normal")
           ├── canonical_fault_category: str (e.g. "NETWORK_FLOOD_FAULT")
           └── root_cause_metric: Optional[str] (e.g. "icmp.checksum")
```

### Attack-to-Fault Mapping:
In this benchmark, cyberattack and anomalous traffic episodes are mapped to edge fault categories:
- `DDoS_ICMP`, `DDoS_UDP`, `DDoS_TCP`, `DDoS_HTTP` $\to$ `NETWORK_FLOOD_FAULT`
- `SQL_injection`, `XSS`, `Uploading` $\to$ `APPLICATION_EXPLOIT_FAULT`
- `Password`, `Vulnerability_scanner`, `Port_Scanning`, `Fingerprinting` $\to$ `RECONNAISSANCE_BURST_FAULT`
- `Ransomware` $\to$ `RESOURCE_EXHAUSTION_FAULT`
- `MITM` $\to$ `TRAFFIC_HIJACK_FAULT`
- `Normal` $\to$ `NORMAL_OPERATION`

---

## 5. Timing Boundaries & Overhead Exclusions

To eliminate bias from disk I/O, file reading, and garbage collection, the execution timer strictly isolates algorithm CPU execution:

```text
┌────────────────────────────────────────────────────────┐
│ Streaming Harness Loop                                 │
│                                                        │
│  Chunk Read & Deserialization (EXCLUDED FROM T_ALGO)   │
│  Context Creation (EXCLUDED FROM T_ALGO)               │
│                                                        │
│    T_start = time.perf_counter_ns()                    │
│    ┌──────────────────────────────────────────────┐    │
│    │ algorithm.process(observable_input)          │    │
│    └──────────────────────────────────────────────┘    │
│    T_end = time.perf_counter_ns()                      │
│                                                        │
│    T_algo = (T_end - T_start) / 1,000,000.0  [ms]      │
│                                                        │
│  Metrics Accumulation (EXCLUDED FROM T_ALGO)           │
└────────────────────────────────────────────────────────┘
```

- **Per-Record Latency**: Measures strictly $T_{\text{algo}}$ for single-observation processing.
- **Throughput**: Calculated as $\frac{N_{\text{evaluated}}}{T_{\text{total\_wall\_clock}}}$.
- **Warm-Up Exclusion**: The first 1,000 records allow internal streaming trees, sliding window buffers, and neural graph caches to stabilize. All metrics computed during warm-up are strictly discarded from final tables.

---

## 6. Resource Measurement & Bandwidth Accounting

- **CPU & Memory**: Sampled via isolated background OS threads (`psutil.Process(os.getpid())`) recording average CPU utilization percentage and peak Resident Set Size (RSS in MB).
- **Energy Estimation**: Calculated via a calibrated linear-polynomial power model ($P_{\text{idle}} + (P_{\text{peak}} - P_{\text{idle}}) \times \frac{\text{CPU}\%}{100}$) and labeled strictly as **`ESTIMATED`**.
- **Bandwidth Accounting**:
  - *Input Telemetry Bandwidth*: Serialized payload bytes of incoming records (common to all algorithms).
  - *Mitigation Bandwidth*: Measured strictly for algorithms executing task/container migration or node replication (IPFT, PreGAN, Causal FT).
  - *Non-Migrating Algorithms*: BWOAIF and RCD are marked as **`NOT_APPLICABLE`** for migration bandwidth (never as $0\text{ MB}$ to avoid deceptive superiority claims).

---

## 7. Algorithm Process Isolation Protocol

To guarantee that CPU and memory overheads are 100% attributable to the algorithm being tested:
1. **Single-Algorithm Execution**: Algorithms are evaluated sequentially in separate process invocations.
2. **Process Termination & Reset**: Between algorithm runs, OS process memory and heap allocations are cleanly destroyed.
3. **Deterministic Stream Manifest**: The harness writes a cryptographic manifest to `results/raw/benchmark_manifest.json` containing the dataset name, file list, feature hash, warm-up count, and configuration signature to ensure every algorithm consumes the exact same sequence.

---

## 8. Capability-Aware Evaluation Contract

The execution contract enforces that algorithms are evaluated only on capabilities they natively support:

| Algorithm | Primary Supported Capabilities | Evaluated Metric Groups | Excluded / Incomparable Metrics |
| :--- | :--- | :--- | :--- |
| **Proposed Causal FT** | Detection, RCA, Recovery | All 10 parameters (Detection, RCA, E2E Latency, Cost, Reliability) | None (Full integrated loop) |
| **Paper 1 (IPFT)** | Prediction, Migration | Forecasting, Detection (thresholded), Migration count, Cost, Reliability | Root-Cause Localization |
| **Paper 2 (BWOAIF)** | Streaming Detection | Anomaly F1, Per-record latency, Throughput, CPU, RAM, Scalability | RCA, Migration, Operational Cost |
| **Paper 3 (RCD)** | Causal Discovery (RCA) | Top-$k$ RCA Recall, Diagnostic Latency, Execution Time, Scalability | Streaming Detection, Migration, Cost |
| **Paper 4 (PreGAN)** | Prediction, Preemptive Migration | Detection F1, Prototype Diagnosis, Migration count, Cost, Reliability | Causal Root-Cause Localization |

---

## 9. Methodological Limitations

1. **Passive Telemetry Benchmark**: Edge-IIoTset is an offline recorded network telemetry dataset. Simulated migrations do not physically alter historical packet streams; mitigation impact is measured within a coupled edge simulation harness.
2. **Physical Power Measurement**: The host operating system does not expose hardware power meters or unvirtualized RAPL registers. Energy metrics are strictly model-based estimates.
3. **Asymmetric Algorithmic Paradigms**: Standalone detectors and diagnostic tools are not penalized with artificial zero-recovery scores; their evaluation is strictly capability-aware.
