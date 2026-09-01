# Dataset Placement & Integration Guide

This directory (`data/datasets/`) is reserved exclusively for user-provided real-world IoT telemetry datasets.

---

## 1. Where to Place Your Dataset

Place your raw dataset files under `data/datasets/<dataset_name>/`. 

Recommended layout:

```
data/
└── datasets/
    └── <dataset_name>/
        ├── raw/            # Place raw data files here (.csv, .parquet, .xlsx, .json)
        ├── documentation/  # Optional dataset documentation or papers
        └── metadata/       # Optional dataset metadata, dictionary, or schema specs
```

---

## 2. Supported File Formats

The dataset adapter will support standard telemetry formats:
* **CSV Files** (`.csv`)
* **Excel Worksheets** (`.xlsx`, `.xls`)
* **Apache Parquet** (`.parquet`)
* **JSON / JSON Lines** (`.json`, `.jsonl`)

---

## 3. Schema Detection Workflow

When a new dataset is provided, the ingestion tool performs automated schema inspection before processing:
* File inspection & format validation
* Header and column name extraction
* Row count, column count, and data type analysis
* Missing value ratios per column
* Identification of timestamp fields and device/edge/node identifiers
* Identification of telemetry metrics (CPU, memory, network, throughput, latency, packet loss, workload)
* Identification of ground-truth attack/anomaly/fault labels

---

## 4. Column Mapping to `TelemetryRecord`

All real dataset records are converted into the internal `TelemetryRecord` standard interface.
* Pipeline components (detection, causal analysis, fault recovery) rely exclusively on `TelemetryRecord`.
* The adapter handles mapping raw column names to `TelemetryRecord` attributes without mutating downstream algorithms.

---

## 5. Handling Missing Telemetry Fields

**No missing telemetry fields will be fabricated or randomly generated.**
For each field in `TelemetryRecord`, the status will be classified as:
* **`AVAILABLE`**: Direct mapping from a raw dataset column.
* **`DERIVED`**: Computed/derived deterministically from existing raw columns (documented explicitly).
* **`MISSING`**: Field is absent from the dataset and reported clearly without synthetic noise injection.

---

## 6. Label Isolation (Ground Truth)

* Ground-truth attack/fault/anomaly labels will **NEVER** be used as inputs to the detector or causal analysis engine.
* Labels are preserved inside `TelemetryRecord` as `fault_label` / `original_label` strictly for **offline evaluation** (e.g. calculating precision, recall, F1-score).

---

## 7. Execution Modes

* **Simulation Mode** (Always available):
  ```bash
  python main.py --mode simulation --seed 42
  ```
* **Dataset Mode** (To be executed once dataset files and mappings are provided):
  ```bash
  python main.py --mode dataset --dataset-path data/datasets/<dataset_name>
  ```
