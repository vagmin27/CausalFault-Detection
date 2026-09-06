# Comprehensive Dataset Inspection Report

**Dataset Identifier**: TON_IoT Dataset (IoT Device Telemetry Subsets)  
**Provider / Origin**: UNSW Canberra Cyber IoT Lab  
**Inspection Date**: September 1, 2026  
**Inspection Mode**: Read-Only / Non-Destructive Inspection  

---

## 1. Dataset Inventory

The dataset located in `data/datasets/` consists of three main subdirectories:

### A. Raw Telemetry Data Files (`data/datasets/Processed_IoT_dataset/`)

| Filename | File Type | File Size | Record Count | Column Count | Primary Classification / Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `IoT_Fridge.csv` | CSV | 23.81 MB | 587,076 | 6 | Raw Telemetry (Smart Refrigerator Sensor) |
| `IoT_GPS_Tracker.csv` | CSV | 29.97 MB | 595,686 | 6 | Raw Telemetry (GPS Tracker Coordinates) |
| `IoT_Garage_Door.csv` | CSV | 24.57 MB | 591,446 | 6 | Raw Telemetry (Garage Door & Smartphone Signal) |
| `IoT_Modbus.csv` | CSV | 14.64 MB | 287,194 | 8 | Raw Telemetry (Industrial Modbus Protocol Registers) |
| `IoT_Motion_Light.csv` | CSV | 15.79 MB | 452,262 | 6 | Raw Telemetry (Motion & Light Actuator Status) |
| `IoT_Weather.csv` | CSV | 40.06 MB | 650,242 | 7 | Raw Telemetry (Weather Station: Temp, Pressure, Humidity) |

**Total Active Dataset Records**: **3,164,006 rows** (~148.84 MB total data).

### B. Documentation & Metadata (`data/datasets/Description_stats_IoT_dataset/`)

| Filename | File Type | File Size | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `IoT Features- Description.xlsx` | XLSX | 19.15 KB | Feature dictionary defining column names, data types, and device descriptions. |
| `IoT Features-Description.docx` | DOCX | 24.72 KB | MS Word version of the IoT feature specifications. |
| `IoT Features-Description.pdf` | PDF | 129.15 KB | PDF version of the IoT feature specifications. |
| `Statistics of IoT Records.xlsx` | XLSX | 12.31 KB | Summary statistical table of normal vs. attack record breakdowns. |
| `Statistics of IoT Records.docx` | DOCX | 26.50 KB | MS Word version of record statistics and train/test splits. |
| `Statistics of IoT Records.pdf` | PDF | 145.51 KB | PDF version of record statistics and train/test splits. |

### C. Error & Log Files (`data/datasets/__Processed_IoT_dataset/` & root)

| Filename | File Type | File Size | Category | Description |
| :--- | :--- | :--- | :--- | :--- |
| `IoT_Thermostat.csv_Error.txt` | TXT | 175 Bytes | Error Log | Download error log for missing `IoT_Thermostat.csv`. |
| `___All_Errors.txt` | TXT | 184 Bytes | Error Log | Aggregated download error manifest. |

---

## 2. CSV Dataset Schema & Statistical Inspection

### 1. `IoT_Fridge.csv`
* **Shape**: 587,076 rows × 6 columns
* **Columns & Types**:
  * `date` (string): Log date (e.g. `'31-Mar-19'`)
  * `time` (string): Log time with whitespace (e.g. `'   12:36:52   '`)
  * `fridge_temperature` (float64): Temperature reading (Range: `1.0` to `14.0`, Median: `6.7`)
  * `temp_condition` (string): Categorical status (`'high '`, `'low '`)
  * `label` (int64): Binary target (`0` = Normal: 500,827 | `1` = Attack: 86,249)
  * `type` (string): Multi-class attack category (`normal`: 500,827, `backdoor`: 35,568, `password`: 28,425, `ddos`: 10,233, `injection`: 7,079, `ransomware`: 2,902, `xss`: 2,042)
* **Missing Values**: 0 (0.0%)

### 2. `IoT_GPS_Tracker.csv`
* **Shape**: 595,686 rows × 6 columns
* **Columns & Types**:
  * `date` (string): Log date
  * `time` (string): Log time
  * `latitude` (float64): GPS Latitude (Range: `0.0` to `1498.53`, Median: `77.68`)
  * `longitude` (float64): GPS Longitude (Range: `10.0` to `1508.08`, Median: `87.97`)
  * `label` (int64): Binary target (`0` = Normal: 513,849 | `1` = Attack: 81,837)
  * `type` (string): Attack category (`normal`: 513,849, `backdoor`: 35,571, `password`: 25,176, `ddos`: 10,226, `injection`: 6,904, `ransomware`: 2,833, `xss`: 577, `scanning`: 550)
* **Missing Values**: 0 (0.0%)

### 3. `IoT_Garage_Door.csv`
* **Shape**: 591,446 rows × 6 columns
* **Columns & Types**:
  * `date` (string): Log date (21,335 missing values / 3.6%)
  * `time` (string): Log time (21,335 missing values / 3.6%)
  * `door_state` (string): Door state (`'closed'`, `'open'`) (50,269 missing / 8.5%)
  * `sphone_signal` (string/object): Smartphone signal (`'true  '`, `'false  '`, mixed types) (50,269 missing / 8.5%)
  * `label` (int64): Binary target (`0` = Normal: 515,443 | `1` = Attack: 76,003)
  * `type` (string): Attack category (`normal`: 515,443, `backdoor`: 35,568, `password`: 19,287, `ddos`: 10,230, `injection`: 6,331, `ransomware`: 2,902, `xss`: 1,156, `scanning`: 529)
* **Missing Values**: Present in `date`, `time`, `door_state`, `sphone_signal`.

### 4. `IoT_Modbus.csv`
* **Shape**: 287,194 rows × 8 columns
* **Columns & Types**:
  * `date` (string): Log date
  * `time` (string): Log time
  * `FC1_Read_Input_Register` (int64): Modbus Function Code 1 reading (Range: `0` to `65,533`)
  * `FC2_Read_Discrete_Value` (int64): Modbus Function Code 2 reading (Range: `0` to `65,535`)
  * `FC3_Read_Holding_Register` (int64): Modbus Function Code 3 reading (Range: `0` to `65,535`)
  * `FC4_Read_Coil` (int64): Modbus Function Code 4 reading (Range: `0` to `65,535`)
  * `label` (int64): Binary target (`0` = Normal: 222,855 | `1` = Attack: 64,339)
  * `type` (string): Attack category (`normal`: 222,855, `backdoor`: 40,011, `password`: 18,115, `injection`: 5,186, `scanning`: 529, `xss`: 498)
* **Missing Values**: 0 (0.0%)

### 5. `IoT_Motion_Light.csv`
* **Shape**: 452,262 rows × 6 columns
* **Columns & Types**:
  * `date` (string): Log date
  * `time` (string): Log time
  * `motion_status` (int64): Motion sensor state (`0` = off, `1` = on)
  * `light_status` (string): Light actuator state (`' off'`, `' on'`)
  * `label` (int64): Binary target (`0` = Normal: 388,328 | `1` = Attack: 63,934)
  * `type` (string): Attack category (`normal`: 388,328, `backdoor`: 28,209, `password`: 17,521, `ddos`: 8,121, `injection`: 5,595, `ransomware`: 2,264, `scanning`: 1,775, `xss`: 449)
* **Missing Values**: 0 (0.0%)

### 6. `IoT_Weather.csv`
* **Shape**: 650,242 rows × 7 columns
* **Columns & Types**:
  * `date` (string): Log date
  * `time` (string): Log time
  * `temperature` (float64): Ambient temperature (Range: `20.51` to `50.0`)
  * `pressure` (float64): Atmospheric pressure (Range: `-37.63` to `26.69`)
  * `humidity` (float64): Relative humidity (Range: `0.00016` to `99.89`)
  * `label` (int64): Binary target (`0` = Normal: 559,718 | `1` = Attack: 90,524)
  * `type` (string): Attack category (`normal`: 559,718, `backdoor`: 35,641, `password`: 25,715, `ddos`: 15,182, `injection`: 9,726, `ransomware`: 2,865, `xss`: 866, `scanning`: 529)
* **Missing Values**: 0 (0.0%)

---

## 3. Documentation Inspection Findings

From `IoT Features- Description.xlsx` and `Statistics of IoT Records.xlsx`:
1. **Dataset Origin**: TON_IoT dataset generated by UNSW Canberra Cyber IoT Lab.
2. **Environment**: Testbed combining IoT sensors (Raspberry Pi/Arduino devices) connected via Node-RED, Modbus PLCs, and cloud infrastructure.
3. **Attack Categories**:
   * **Normal**: Standard background operating conditions.
   * **Backdoor**: Remote unauthorized access channel creation.
   * **DDoS**: Distributed Denial of Service flooding.
   * **Injection**: Command/SQL injection attack payloads.
   * **Password**: Password cracking / brute force.
   * **Ransomware**: File encryption / ransom payload activity.
   * **Scanning**: Network port scanning and vulnerability probing.
   * **XSS**: Cross-site scripting payload injection.
4. **Data Characteristics**: The telemetry records capture **IoT Application/Sensor Layer data** (temperatures, motion states, Modbus registers, GPS coordinates) rather than Linux Kernel system metrics (CPU %, RAM MB, OS Kernel packet loss).

---

## 4. Telemetry Model Mapping (`TelemetryRecord`)

| Field | Status | Mapping / Derivation / Explanation |
| :--- | :--- | :--- |
| `timestamp` | **DERIVABLE** | Derived by concatenating `date` + `time` columns and parsing into POSIX epoch / ISO string. |
| `device_id` | **DERIVABLE** | Derived explicitly from file name / device service profile (e.g. `'IoT_Fridge'`, `'IoT_Weather'`). |
| `edge_node_id` | **DERIVABLE** | Mapped from `device_id` to edge node topology identifiers (e.g. `'Edge_Node_1'`). |
| `cpu_utilization` | **MISSING** | **Field does not exist in dataset.** (Telemetry contains sensor values, not host CPU metrics). |
| `memory_utilization` | **MISSING** | **Field does not exist in dataset.** (Host memory usage is absent). |
| `network_utilization` | **DERIVABLE** | Can be approximated via message/record throughput over time windows, but direct host NIC % is **MISSING**. |
| `latency` | **DERIVABLE** | Can be calculated as time-delta between consecutive sensor events ($\Delta t$), but network RTT is **MISSING**. |
| `packet_loss` | **MISSING** | **Field does not exist in dataset.** |
| `throughput` | **DERIVABLE** | Calculated as number of sensor records received per second over sliding time windows. |
| `workload` | **DERIVABLE** | Mapped from active sensor state changes or register transaction volume per second. |
| `fault_label` | **AVAILABLE** | Mapped directly from dataset column `label` (`0` = Normal, `1` = Anomaly/Fault). |
| `fault_type` | **AVAILABLE** | Mapped directly from dataset column `type` (`'normal'`, `'ddos'`, `'backdoor'`, etc.). |
| `original_label` | **AVAILABLE** | Retained directly from column `type` / `label` for exact offline validation. |

> [!IMPORTANT]
> **NO TELEMETRY FABRICATION**: Missing fields (`cpu_utilization`, `memory_utilization`, `packet_loss`) will be reported as `MISSING` (or `None`/`NaN`) rather than injected with synthetic random noise.

---

## 5. Research Suitability Evaluation

Evaluation against paper objectives: *"Causal Real-Time Adaptive Fault Tolerance for Edge-IoT Systems"*

* **A. Real-Time Fault/Anomaly Detection**: **SUITABLE**  
  Rich 3.16M event log with ground-truth anomaly/attack labels allows thorough evaluation of anomaly detection algorithms.
* **B. Device-Level Telemetry**: **SUITABLE FOR APPLICATION/SENSOR LAYER, UNSUITABLE FOR SYSTEM/HOST LAYER**  
  Contains real IoT device measurements (temperatures, actuator signals, Modbus registers), but lacks Linux OS kernel metrics.
* **C. Time-Series Analysis**: **PARTIALLY SUITABLE**  
  Contains time logs, though timestamps have 1-second resolution with high burst concurrency (multiple logs per second).
* **D. Fault/Attack Labels for Offline Evaluation**: **SUITABLE**  
  Includes 8 granular attack classes for computing Precision, Recall, F1-Score, and ROC-AUC.
* **E. Causal / Root-Cause Analysis**: **PARTIALLY SUITABLE (REQUIRES SENSOR-LAYER CAUSAL GRAPH)**  
  The default causal DAG (`cpu -> latency`, `network -> packet_loss`) cannot be directly evaluated because CPU/Memory/Packet Loss are missing. To evaluate causal graph discovery/inference, the graph must map available domain attributes (e.g. `workload -> throughput -> sensor_anomaly_state`).
* **F. Recovery Evaluation**: **UNSUITABLE / SIMULATED RECOVERY ONLY**  
  The dataset consists of static offline CSV logs. Closed-loop recovery actions (e.g., traffic rerouting, workload redistribution) cannot execute live physical control feedback on static historical logs; recovery must be evaluated via counterfactual simulation or offline policy replay.

---

## 6. Ground-Truth Label Isolation

* The `label` and `type` columns represent ground-truth attack labels.
* **Isolation Rule**: During streaming ingestion, `label` and `type` will be stored in `TelemetryRecord.fault_label`, `TelemetryRecord.fault_type`, and `TelemetryRecord.original_label`.
* **Control Flow Scoping**: These label fields are strictly isolated from feature vectors passed to the `RealTimeDetector` and `CausalEngine`. They are used exclusively by the `EvaluationEngine` after predictions are made.

---

## 7. Data Quality Issues

1. **Missing File (`IoT_Thermostat.csv`)**:
   * File `IoT_Thermostat.csv` is missing from `Processed_IoT_dataset/`.
   * Cause verified in `IoT_Thermostat.csv_Error.txt`: OneDrive download rate limit error (`TooManyRequestsMeTAException`).
2. **Missing Values**:
   * `IoT_Garage_Door.csv` contains 21,335 missing dates/times (3.6%) and 50,269 missing `door_state`/`sphone_signal` entries (8.5%).
   * All other 5 CSV dataset files have 0 missing values.
3. **Timestamp Formatting & Padding**:
   * `time` strings contain irregular leading/trailing spaces (e.g. `'   12:36:52   '`).
   * High timestamp duplication: Up to 500+ records share the exact same 1-second timestamp string.
4. **Class Imbalance**:
   * ~85-88% Normal records vs. ~12-15% Anomaly/Attack records across all files.

---

## 8. Dataset Mode Readiness

### Readiness Verdict: **YES WITH DERIVATIONS**

### Required Adapter Implementation Plan (When Authorized):
1. **`TONIoTDatasetAdapter` Creation**:
   * Streaming reader supporting multi-file CSV ingestion from `data/datasets/Processed_IoT_dataset/`.
   * String sanitization for timestamps (stripping whitespace, merging `date` + `time`).
2. **Feature Mapping & Surrogate Derivation**:
   * Map numeric sensor fields (`temperature`, `pressure`, `humidity`, Modbus registers, motion status) into structured `TelemetryRecord` payload fields.
   * Derive `throughput` and `workload` metrics from rolling window record frequencies.
   * Mark `cpu_utilization`, `memory_utilization`, and `packet_loss` explicitly as `MISSING`.
3. **Domain-Specific Causal Graph Definition**:
   * Define a sensor-layer causal graph mapping available variables (`throughput`, `workload`, `sensor_deviation`) to support causal inference on TON_IoT telemetry.

---

## 9. Confirmation of Code Integrity

* **Source Code Changes**: **0 Python files modified during inspection phase.**
* All core system modules (`data/telemetry.py`, `data/simulator_adapter.py`, `detection/`, `causal/`, `recovery/`, `simulation/`, `faults/`, `main.py`) remain completely untouched in simulation mode.
* Simulation mode (`python main.py --mode simulation --seed 42`) remains 100% operational.

---

## 10. Full Dataset Ingestion & Evaluation Final Results

- **Records Processed**: **3,163,906** across all 6 CSV files.
- **Streaming Throughput**: **11,624.36 rec/sec** (Total latency: **0.0452 ms/rec**).
- **Detection Precision**: **8.13%** overall (**65.71%** on `IoT_Fridge.csv`).
- **Detection False Positive Rate (FPR)**: **0.01%**.
- **Physical Recovery Status**: `DISABLED_STATIC_LOGS`.
- **Output Artifacts**:
  - `results/data/dataset_detection_metrics.csv`
  - `results/data/dataset_device_metrics.csv`
  - `results/data/causal_results.csv`
  - `results/plots/dataset_precision_recall_f1.png`
  - `results/plots/dataset_detection_latency.png`
  - `results/plots/device_comparison.png`
  - `results/plots/attack_type_performance.png`

