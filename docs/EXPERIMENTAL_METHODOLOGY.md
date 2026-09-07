# Experimental Methodology & Benchmark Specification (Revised)

## Comprehensive Evaluation Framework for Causal Real-Time Adaptive Fault-Tolerance and Baseline Approaches

---

## 1. Executive Summary & Core Principles

This document establishes the formal, academically rigorous evaluation framework to compare our proposed **Causal Fault-Tolerance (Causal FT)** architecture against four reference papers:

1. **Proposed Approach**: *Causal Real-Time Adaptive Fault-Tolerance* (Modular Closed-Loop: Streaming Anomaly Detection $\to$ Causal Root-Cause Discovery $\to$ Adaptive State-Aware Recovery).
2. **Paper 1 (IPFT)**: *Intelligent Proactive Fault Tolerance at the Edge through Resource Usage Prediction* (Theodoropoulos et al., ITU JFET 2022).
3. **Paper 2 (BWOAIF)**: *Bilateral-Weighted Online Adaptive Isolation Forest for anomaly detection in streaming data* (Hannák et al., Wiley SAM 2023).
4. **Paper 3 (RCD)**: *Root Cause Analysis of Failures in Microservices through Causal Discovery* (Ikram et al., NeurIPS 2022).
5. **Paper 4 (PreGAN)**: *PreGAN: Preemptive Migration Prediction Network for Proactive Fault-Tolerant Edge Computing* (Tuli et al., IEEE INFOCOM 2022).

### Academic Non-Negotiables:
1. **Strict Dual-Track Results**:
   - **Track A: Paper-Reported Results**: Numbers directly extracted from the published manuscripts. Any metric not explicitly published is marked **`NR` (Not Reported)**. Paper numbers are never modified, guessed, or overwritten.
   - **Track B: Common-Benchmark Results**: Numbers empirically measured under our standardized runtime environment using the Edge-IIoTset benchmark.
   - Tracks A and B are **never** plotted on the same unsegmented axis.
2. **Functional Boundary Preservation**:
   - We do **not** force fundamentally different tasks into a single aggregated score.
   - Standalone anomaly detectors (BWOAIF) are evaluated on detection. Diagnostic causal discovery algorithms (RCD) are evaluated on root-cause localization. Proactive systems (IPFT, PreGAN, Causal FT) are evaluated on closed-loop prediction and mitigation.
3. **No Fabricated Data or Misleading Proxies**:
   - Software-estimated energy is labeled as **"Estimated Energy Consumption"**, never as measured physical power.
   - Non-monetary recovery penalties are labeled as **"Operational Recovery Cost"**, never as monetary currency.
   - Processing throughput (records/sec) is strictly separated from hardware **Computational Capacity**.
   - Distinct latencies (detection, diagnosis, recovery, end-to-end) are reported separately and never conflated.

---

## 2. Functional Scope & Algorithmic Capabilities

```text
Pipeline Stage        Causal FT       IPFT (P1)     BWOAIF (P2)    RCD (P3)      PreGAN (P4)
──────────────────────────────────────────────────────────────────────────────────────────
Primary Objective     Closed-loop FT  Resource Pred  Online Stream  Causal RCA   Preemptive Mig
                                      & Threshold    Outlier Det    via F-NODE   via GAN+FewShot
Data Ingestion        Streaming       Streaming      Batch Stream   Normal/Fault Streaming
                                                                    Dataset Pair
Anomaly Detection     Native (HST)    Thresholded    Native (IF)    External Req GAT + GRU
Root-Cause Discovery  DoWhy Causal    None           None           Localized    Prototypical
                      Inference                                     Psi-PC       Classifier
Mitigation / Recovery Adaptive Task   Proactive      None           None         Preemptive
                      Rebalancing     Replication/                               Migration (GAN)
                                      Migration                                  
Execution Loop        Full Closed     Prediction +   Detection      Diagnosis    Prediction +
                      Loop            Migration      Only           Only         Migration
```

---

## 3. The 10 Headline Evaluation Parameters & Submetric Hierarchy

The evaluation framework defines exactly **10 Headline Parameters**, organized under four foundational pillars: **Performance**, **Resources**, **Efficiency**, and **System Quality**. Each headline parameter encapsulates dedicated submetrics to prevent conflation of distinct operations.

```text
10 HEADLINE EVALUATION PARAMETERS
├── PILLAR I: PERFORMANCE
│   ├── 1. Latency / Delay
│   │   ├── 1.1 Detection Latency (MTTD)
│   │   ├── 1.2 Root-Cause Diagnosis Latency
│   │   ├── 1.3 Recovery Action Latency
│   │   └── 1.4 End-to-End Fault-to-Recovery Latency
│   ├── 2. Execution / Response Time
│   │   ├── 2.1 Per-Record Ingestion / Inference Latency
│   │   ├── 2.2 Task Service Response Time
│   │   └── 2.3 Algorithmic Execution Overhead Ratio
│   └── 3. Accuracy
│       ├── 3.1 Fault Detection Accuracy (Precision, Recall, F1, ROC-AUC)
│       ├── 3.2 Root-Cause Localization Accuracy (Top-1, Top-3, Top-5 Recall)
│       └── 3.3 Recovery Mitigation Success Rate
├── PILLAR II: RESOURCES
│   ├── 4. Computational Capacity & Throughput
│   │   ├── 4.1 Hardware Computational Capacity (Baseline Core/Thread/Frequency Profile)
│   │   └── 4.2 Processing Throughput (Records/sec)
│   ├── 5. Resource Utilization
│   │   ├── 5.1 Average CPU Utilization (%)
│   │   └── 5.2 Resident Memory Footprint (Peak RSS in MB)
│   └── 6. Bandwidth
│       ├── 6.1 Telemetry Streaming Bandwidth (KB/s)
│       └── 6.2 Control & State Migration Bandwidth Overhead (KB or MB)
├── PILLAR III: EFFICIENCY
│   ├── 7. Energy Consumption & Efficiency
│   │   ├── 7.1 Measured Energy (Physical Power Meter, where hardware supported)
│   │   └── 7.2 Estimated Energy Consumption (Joules / Watt-hours via Documented CPU Model)
│   └── 8. Operational Recovery Cost
│       ├── 8.1 Total Mitigation / Migration Action Count
│       ├── 8.2 SLA / SLO Latency Violation Ratio
│       └── 8.3 Normalized Operational Penalty Score
└── PILLAR IV: SYSTEM QUALITY
    ├── 9. Reliability & Availability
    │   ├── 9.1 Service Availability (%)
    │   └── 9.2 Mean Time To Failure (MTTF) & Mean Time To Repair (MTTR)
    └── 10. Scalability
        ├── 10.1 Feature-Dimensional Scalability (D in {10, 25, 50, 62})
        └── 10.2 Edge-System / Workload Scalability (M in {5, 10, 20, 50} Nodes)
```

---

## 4. Rigorous Specifications for Every Parameter

---

### PARAMETER 1: Latency / Delay
- **Core Concept**: Time intervals elapsed between distinct operational milestones during an anomalous episode.
- **Submetrics & Formulations**:
  1. **Detection Latency ($\text{MTTD}$)**:
     $$\text{MTTD} = \frac{1}{|F|} \sum_{i=1}^{|F|} \left(t_{\text{detect}}^{(i)} - t_{\text{onset}}^{(i)}\right) \quad [\text{ms}]$$
  2. **Root-Cause Diagnosis Latency ($T_{\text{diag}}$)**:
     $$T_{\text{diag}} = t_{\text{rca\_complete}} - t_{\text{detect}} \quad [\text{ms or s}]$$
  3. **Recovery Action Latency ($T_{\text{act}}$)**:
     $$T_{\text{act}} = t_{\text{recovery\_executed}} - t_{\text{rca\_complete}} \quad [\text{ms}]$$
  4. **End-to-End Fault-to-Recovery Latency ($T_{\text{e2e}}$)**:
     $$T_{\text{e2e}} = \text{MTTD} + T_{\text{diag}} + T_{\text{act}} = t_{\text{restored}} - t_{\text{onset}} \quad [\text{ms}]$$
- **Unit**: Milliseconds (ms) or seconds (s).
- **Optimization Direction**: Lower is Better ($\downarrow$).
- **Instrumentation**: High-resolution monotonic hardware clock (`time.perf_counter_ns()`).
- **Fairness & Comparability**:
  - Detection Latency is directly comparable across Causal FT, BWOAIF, PreGAN, and IPFT.
  - Diagnosis Latency is directly comparable between Causal FT and RCD.
  - Recovery & End-to-End Latency are only comparable across closed-loop mitigation approaches (Causal FT, IPFT, PreGAN). BWOAIF and RCD are **NOT COMPARABLE** for recovery latency.

---

### PARAMETER 2: Execution / Response Time
- **Core Concept**: Computational execution efficiency of the algorithm itself versus the task execution time experienced by the end-user application.
- **Submetrics & Formulations**:
  1. **Per-Record Inference Latency ($T_{\text{infer}}$)**:
     $$T_{\text{infer}} = \frac{1}{N} \sum_{i=1}^{N} \left(t_{\text{step\_end}}^{(i)} - t_{\text{step\_start}}^{(i)}\right) \quad [\mu\text{s or ms/rec}]$$
  2. **Task Response Time ($T_{\text{resp}}$)**:
     $$T_{\text{resp}} = t_{\text{task\_received}} - t_{\text{task\_submitted}} \quad [\text{ms}]$$
  3. **Algorithmic Overhead Ratio ($\theta_{\text{over}}$)**:
     $$\theta_{\text{over}} = \frac{T_{\text{algorithm\_cpu\_time}}}{T_{\text{total\_system\_runtime}}}$$
- **Unit**: $\mu$s/rec, ms/rec, or dimensionless ratio.
- **Optimization Direction**: Lower is Better ($\downarrow$).
- **Instrumentation**: Process CPU user/system clock counters (`psutil.Process().cpu_times()`, `time.perf_counter()`).
- **Fairness & Comparability**: **DIRECT** for Per-Record Inference Latency across all five approaches on the exact same CPU core.

---

### PARAMETER 3: Accuracy
- **Core Concept**: The statistical fidelity of decisions made by the algorithm, reported strictly by sub-task category.
- **Submetrics & Formulations**:
  1. **Fault Detection Accuracy (Binary Detection)**:
     $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{F1} = \frac{2 \cdot P \cdot R}{P + R}, \quad \text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}) \, d(\text{FPR})$$
  2. **Root-Cause Localization Accuracy (Diagnostic Ranking)**:
     $$\text{Top-}k \text{ Recall} = \frac{1}{|E_{\text{fault}}|} \sum_{e \in E_{\text{fault}}} \mathbb{I}\left(\text{TrueCause}(e) \in \text{RankedCauses}_{1..k}(e)\right) \quad \text{for } k \in \{1, 3, 5\}$$
  3. **Recovery Mitigation Success Rate**:
     $$\text{Success Rate} = \frac{N_{\text{successful\_mitigations}}}{N_{\text{triggered\_actions}}} \times 100\%$$
- **Unit**: Percentage (%) or decimal ($0.0 - 1.0$).
- **Optimization Direction**: Higher is Better ($\uparrow$).
- **Instrumentation**: Zero-leakage ground-truth accumulator recording confusion matrix and diagnostic rank sets.
- **Fairness & Comparability**:
  - Detection F1: Causal FT, BWOAIF, PreGAN (DIRECT); IPFT (CONDITIONAL via thresholded prediction). RCD is **NOT COMPARABLE** (requires prior fault detection).
  - Localization Top-$k$ Recall: Causal FT vs. RCD (DIRECT). BWOAIF, IPFT, PreGAN are **NOT COMPARABLE** (they output anomaly scores or generic prototype classes, not causal graph parentage).
  - Recovery Success Rate: Causal FT, IPFT, PreGAN (DIRECT). BWOAIF and RCD are **NOT COMPARABLE**.

---

### PARAMETER 4: Computational Capacity & Throughput
- **Core Concept**: Separation between physical hardware capacity of the edge node and the sustained algorithmic processing rate.
- **Submetrics & Formulations**:
  1. **Hardware Computational Capacity (Static System Specification)**:
     - Processor microarchitecture, physical core count, logical thread count, base and boost clock frequency (GHz), available RAM capacity (GB).
     - *Reporting Standard*: Descriptive baseline specification characterizing the testbed (e.g., AMD/Intel x86-64 or ARM Cortex-A72/A53).
  2. **Processing Throughput ($X_{\text{stream}}$)**:
     $$X_{\text{stream}} = \frac{N_{\text{records\_evaluated}}}{T_{\text{wall\_clock\_time}}} \quad [\text{records/second}]$$
- **Unit**: Static hardware specs (Cores, GHz, GB) for Capacity; records/sec for Throughput.
- **Optimization Direction**: Higher Throughput is Better ($\uparrow$).
- **Instrumentation**: Wall-clock ingestion pipeline timer and hardware telemetry inspection (`platform.processor()`, `psutil.cpu_freq()`, `os.cpu_count()`).
- **Fairness & Comparability**: **DIRECT** across all five implementations executed on the same physical host.

---

### PARAMETER 5: Resource Utilization
- **Core Concept**: The physical computational overhead consumed by the algorithm during continuous streaming inference.
- **Submetrics & Formulations**:
  1. **Average CPU Utilization ($\overline{\text{CPU}}$)**:
     $$\overline{\text{CPU}} = \frac{1}{K} \sum_{k=1}^{K} \text{CPU}_k\% \quad [\%]$$
  2. **Resident Memory Footprint ($\text{RSS}_{\text{peak}}$)**:
     $$\text{RSS}_{\text{peak}} = \max_{t} \left(\text{Memory}_{\text{RSS}}(t)\right) \quad [\text{MB}]$$
- **Unit**: Percentage (%) for CPU; Megabytes (MB) for RAM.
- **Optimization Direction**: Lower is Better ($\downarrow$).
- **Instrumentation**: Dedicated background thread sampling `psutil.Process(os.getpid()).cpu_percent()` and `.memory_info().rss` every 100 ms.
- **Fairness & Comparability**: **DIRECT**. All 5 approaches are profiled in isolated single-process runs.

---

### PARAMETER 6: Bandwidth
- **Core Concept**: Network traffic volume required to operate the fault-tolerance mechanism.
- **Submetrics & Formulations**:
  1. **Telemetry Streaming Bandwidth ($B_{\text{telemetry}}$)**:
     $$B_{\text{telemetry}} = \frac{\sum_{i=1}^{N} \text{ByteSize}(\mathbf{x}_i)}{\Delta t} \quad [\text{KB/s}]$$
  2. **Mitigation Control & State Migration Bandwidth ($B_{\text{control}}$)**:
     $$B_{\text{control}} = \frac{\text{Bytes}_{\text{state\_transfers}} + \text{Bytes}_{\text{RPC\_control}}}{\Delta t} \quad [\text{KB/s or Total MB}]$$
- **Unit**: Kilobytes per second (KB/s) or Total Megabytes (MB).
- **Optimization Direction**: Lower is Better ($\downarrow$).
- **Instrumentation**: Network I/O byte accounting per transaction (`psutil.net_io_counters()` and serialization profiler).
- **Fairness & Comparability**:
  - Telemetry bandwidth is identical across all algorithms.
  - Migration/control bandwidth is **NOT COMPARABLE** for BWOAIF and RCD (they have no migration/actuation mechanism). Assigning them "0 KB" as a score of superiority would be an invalid claim.

---

### PARAMETER 7: Energy Consumption / Efficiency
- **Core Concept**: Total electrical energy consumed by the edge device during the execution of the fault-tolerance pipeline.
- **Submetrics & Protocol**:
  1. **Measured Energy**: Directly captured via physical hardware power meters (e.g., USB power meters, WattsUp Pro, or hardware RAPL counters on Linux). If physical hardware power meters are unavailable on the evaluation host, this submetric is explicitly marked **`UNAVAILABLE_IN_HOST_ENV`**.
  2. **Estimated Energy Consumption ($E_{\text{est}}$)**:
     $$E_{\text{est}} = \sum_{k=1}^{K} P_{\text{est}}(t_k) \cdot \Delta t_k \quad [\text{Joules or Watt-hours}]$$
     Using an established linear-polynomial power model derived from edge CPU load:
     $$P_{\text{est}}(\text{CPU}_k) = P_{\text{idle}} + (P_{\text{peak}} - P_{\text{idle}}) \cdot \left(\frac{\text{CPU}_k}{100}\right) + c_{\text{arch}} \cdot \left(\frac{\text{CPU}_k}{100}\right)^2$$
     - Baseline coefficients documented: Raspberry Pi 4B ($P_{\text{idle}} = 2.7\text{W}$, $P_{\text{peak}} = 6.4\text{W}$) or Intel Xeon E3 baseline ($P_{\text{idle}} = 35\text{W}$, $P_{\text{peak}} = 95\text{W}$).
- **Unit**: Joules (J) or Watt-hours (Wh).
- **Optimization Direction**: Lower is Better ($\downarrow$).
- **Reporting Rule**: **Must strictly be labeled as "Estimated Energy Consumption"**. It must never be presented as physical meter-measured power.
- **Fairness & Comparability**: **CONDITIONAL** (model-based estimate applied to measured CPU utilization profiles across all 5 algorithms).

---

### PARAMETER 8: Operational Recovery Cost
- **Core Concept**: Non-monetary operational disruption and SLA degradation incurred by system failures and recovery operations.
- **Submetrics & Formulations**:
  1. **Total Remediation Action Count ($N_{\text{actions}}$)**:
     $$N_{\text{actions}} = N_{\text{migrations}} + N_{\text{replications}} + N_{\text{rebalances}}$$
  2. **SLA / SLO Latency Violation Ratio ($\phi_{\text{SLO}}$)**:
     $$\phi_{\text{SLO}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left(T_{\text{latency}}^{(i)} > T_{\text{SLO\_threshold}}\right) \times 100\% \quad [\%]$$
  3. **Normalized Operational Recovery Penalty ($C_{\text{op}}$)**:
     $$C_{\text{op}} = w_1 \cdot \left(\frac{N_{\text{actions}}}{N_{\text{tasks}}}\right) + w_2 \cdot \phi_{\text{SLO}} + w_3 \cdot \theta_{\text{over}}$$
     (with weights $w_1 = 0.4, w_2 = 0.4, w_3 = 0.2$ explicitly documented).
- **Unit**: Percentage (%) for $\phi_{\text{SLO}}$; dimensionless penalty score for $C_{\text{op}}$.
- **Optimization Direction**: Lower is Better ($\downarrow$).
- **Reporting Rule**: This metric represents **operational/system cost**, NOT financial or monetary currency (unless a specific monetary pricing model is formally applied).
- **Fairness & Comparability**:
  - Closed-loop approaches (Causal FT, IPFT, PreGAN): **DIRECT**.
  - Standalone detectors/diagnosers (BWOAIF, RCD): **NOT COMPARABLE** (they do not take mitigation actions).

---

### PARAMETER 9: Reliability & Availability
- **Core Concept**: System capacity to sustain service delivery without failure during volatile streaming conditions.
- **Submetrics & Formulations**:
  1. **Service Availability ($A_{\text{service}}$)**:
     $$A_{\text{service}} = \frac{T_{\text{total\_uptime}}}{T_{\text{total\_runtime}}} \times 100\% \quad [\%]$$
  2. **Task Completion Reliability ($R_{\text{task}}$)**:
     $$R_{\text{task}} = \frac{N_{\text{tasks\_completed\_on\_time}}}{N_{\text{tasks\_injected}}} \times 100\% \quad [\%]$$
  3. **Mean Time To Failure ($\text{MTTF}$)** and **Mean Time To Repair ($\text{MTTR}$)**:
     $$\text{MTTF} = \frac{T_{\text{healthy\_total}}}{N_{\text{failures}}}, \quad \text{MTTR} = \frac{T_{\text{repair\_total}}}{N_{\text{repairs}}} \quad [\text{s}]$$
- **Unit**: Percentage (%) for Availability/Reliability; Seconds (s) for MTTF/MTTR.
- **Optimization Direction**: Higher is Better ($\uparrow$) for Availability/Reliability/MTTF; Lower is Better ($\downarrow$) for MTTR.
- **Instrumentation**: Continuous system health and episode monitor logging downtime intervals.
- **Fairness & Comparability**:
  - Closed-loop approaches (Causal FT, IPFT, PreGAN): **DIRECT**.
  - Standalone approaches (BWOAIF, RCD): **CONDITIONAL** (only comparable if paired with an identical standardized baseline recovery actuator; otherwise NOT COMPARABLE).

---

### PARAMETER 10: Scalability
- **Core Concept**: The algorithmic sensitivity of latency, throughput, and resource overhead as problem scale increases along two distinct axes.
- **Two Distinct Experimental Scalability Axes**:
  1. **Axis A: Feature / Telemetry Dimensional Scalability**:
     - Evaluate algorithms across input feature dimensions:
       $$D \in \{10, 25, 50, 62\}$$
     - *Metrics Measured*: Inference Latency vs. $D$, Memory vs. $D$, Detection F1 vs. $D$.
  2. **Axis B: Edge-System / Workload Scalability**:
     - Evaluate algorithms across simulated edge node/workload cluster sizes:
       $$M \in \{5, 10, 20, 50\} \text{ nodes}$$
     - *Metrics Measured*: Cluster Ingestion Throughput vs. $M$, Diagnostic Runtime vs. $M$, Cluster CPU % vs. $M$.
  3. **Complexity Scaling Exponent ($\alpha$)**:
     $$\alpha = \frac{\log(T_{\text{exec}}(M_2)) - \log(T_{\text{exec}}(M_1))}{\log(M_2) - \log(M_1)} \quad \left(\text{Empirical } O(M^\alpha)\right)$$
- **Unit**: Exponent $\alpha$ (dimensionless), or throughput scaling slope.
- **Optimization Direction**: Lower exponent $\alpha \le 1.0$ is Better; Higher maximum supported scale is Better ($\uparrow$).
- **Instrumentation**: Automated parameter sweep orchestrator running identical workloads across grid configurations.
- **Fairness & Comparability**: **DIRECT**. Reveals structural architectural bottlenecks (e.g., standard PC causal discovery scales exponentially with $D$ and $M$, whereas RCD scales polynomially via localized subsets; tree-based models scale linearly).

---

## 5. Formal Comparability Matrix

| # | Headline Parameter | Specific Submetric | Causal FT | IPFT (P1) | BWOAIF (P2) | RCD (P3) | PreGAN (P4) | Comparability Status |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Latency / Delay** | Detection Latency (MTTD) | Native | Conditional | Native | Not Comp. | Native | **DIRECT (Detectors)** |
| | | Diagnosis Latency ($T_{\text{diag}}$) | Native | Not Comp. | Not Comp. | Native | Conditional | **DIRECT (RCA)** |
| | | Recovery Latency ($T_{\text{act}}$) | Native | Native | Not Comp. | Not Comp. | Native | **DIRECT (FT Systems)** |
| | | End-to-End Latency ($T_{\text{e2e}}$) | Native | Native | Not Comp. | Not Comp. | Native | **DIRECT (FT Systems)** |
| **2** | **Execution Time** | Per-Record Inference Latency | Native | Native | Native | Not Comp. | Native | **DIRECT (Online)** |
| | | Batch / Cycle Execution Time | Native | Native | Native | Native | Native | **DIRECT (Overall)** |
| **3** | **Accuracy** | Detection F1-Score | Native | Conditional | Native | Not Comp. | Native | **DIRECT (Detectors)** |
| | | Causal RCA Top-$k$ Recall | Native | Not Comp. | Not Comp. | Native | Not Comp. | **DIRECT (RCA)** |
| | | Recovery Success Rate (%) | Native | Native | Not Comp. | Not Comp. | Native | **DIRECT (FT Systems)** |
| **4** | **Computational Capacity** | Hardware System Specs | Benchmark Hardware Profile (Common Host System Profile) | **COMMON BASELINE** |
| | | Processing Throughput (rec/s) | Native | Native | Native | Native | Native | **DIRECT** |
| **5** | **Resource Utilization** | Average CPU Utilization (%) | Native | Native | Native | Native | Native | **DIRECT** |
| | | Peak Memory Footprint (MB) | Native | Native | Native | Native | Native | **DIRECT** |
| **6** | **Bandwidth** | Telemetry Stream Bandwidth | Common Input Stream Rate (KB/s) across all 5 | **IDENTICAL INPUT** |
| | | Migration / Control Overhead | Native | Native | Not Comp. | Not Comp. | Native | **DIRECT (FT Systems)** |
| **7** | **Energy Consumption** | Measured Physical Energy | Hardware Meter Dependent (`UNAVAILABLE` on host) | **COMMON CONSTRAINT** |
| | | Estimated Energy (Model Wh) | Native | Native | Native | Native | Native | **DIRECT (Model-Based)** |
| **8** | **Operational Cost** | Task Migrations / Actions | Native | Native | Not Comp. | Not Comp. | Native | **DIRECT (FT Systems)** |
| | | SLA / SLO Violation Rate (%) | Native | Native | Not Comp. | Not Comp. | Native | **DIRECT (FT Systems)** |
| **9** | **Reliability** | Service Availability (%) | Native | Native | Conditional | Conditional | Native | **CONDITIONAL** |
| | | Mean Time to Failure (MTTF) | Native | Native | Conditional | Conditional | Native | **CONDITIONAL** |
| **10**| **Scalability** | Feature Scalability ($D$) | Native | Native | Native | Native | Native | **DIRECT** |
| | | System Scalability ($M$) | Native | Native | Native | Native | Native | **DIRECT** |

### Explanatory Notes for Non-Comparable Classifications:
1. **RCD on Detection & Recovery**: RCD is strictly a post-failure causal root-cause localization algorithm. It expects normal and anomalous dataset windows as input. It cannot be scored on MTTD or task migrations because it does not perform streaming detection or scheduling actuation.
2. **BWOAIF on Causal Diagnosis & Recovery**: BWOAIF is strictly an unsupervised streaming outlier detector. It outputs scalar anomaly scores $[0, 1]$. It does not identify root-cause variables, build causal graphs, or trigger container migrations.
3. **Bandwidth & Operational Cost on Detection-Only Methods**: Assigning $0\text{ KB}$ migration bandwidth and $0$ migrations to BWOAIF and claiming it has "lower cost than Causal FT" is a scientifically invalid comparison. These metrics apply exclusively to closed-loop fault tolerance systems.

---

## 6. Statistical Reporting Protocol

To eliminate transient noise from background OS processes, garbage collection, and scheduling jitter:
1. **Number of Independent Runs**: Minimum of **5 repeated runs** for all non-deterministic algorithms (e.g., deep learning models, random-sampling tree ensembles, Bayesian optimizers) across seeds:
   $$\text{Seeds} = [42, 43, 44, 45, 46]$$
2. **Deterministic Components**: For strictly deterministic algorithms (e.g., deterministic constraint-based independence tests in RCD given identical discrete sample matrices), single runs are permitted, provided variance over multiple dataset windows is reported.
3. **Reported Statistics**: All numerical benchmark outputs must report:
   $$\text{Result} = \mu \pm \sigma \quad [\text{95\% Confidence Interval}: \mu \pm 1.96 \cdot \frac{\sigma}{\sqrt{N}}]$$
4. **Warm-Up Protocol**: Every benchmark run must discard the first **1,000 streaming records** to ensure caches, sliding windows, and JIT optimizations reach steady state.

---

## 7. Baseline Reproduction & Adaptation Classifications

| Baseline Approach | Publication Citation | Implementation Classification | Concrete Adaptation & Boundaries |
| :--- | :--- | :--- | :--- |
| **Paper 1: IPFT** | Theodoropoulos et al., ITU JFET 2022 | **Adapted Reproduction** | - **Architecture**: Multi-channel PyTorch model combining a local sequential GRU/LSTM channel with a global cluster feedforward channel.<br>- **Adaptation**: Input adapted from CloudSim/Prometheus monitoring to the 62 standardized Edge-IIoTset telemetry features.<br>- **Actuation**: Dual-threshold trigger ($th_{\text{high}}$ for proactive migration/replication, $th_{\text{low}}$ for deactivation). |
| **Paper 2: BWOAIF** | Hannák et al., Wiley SAM 2023 | **Exact Reproduction** | - **Architecture**: Bilateral-Weighted Online Adaptive Isolation Forest.<br>- **Implementation**: Exact implementation of Equations (1–3) in paper: sliding window batch updates ($B=64, T=1024, E=64$), geometric history weighting ($\gamma=0.5$), and bilateral anomaly score weighting ($\sigma_v = 0.1, \sigma_a = 5.0$). |
| **Paper 3: RCD** | Ikram et al., NeurIPS 2022 | **Exact Reproduction** | - **Architecture**: Hierarchical localized root-cause discovery.<br>- **Implementation**: Exact implementation of Algorithm 1: auxiliary intervention node ($F$-NODE), subset size $\gamma=5$, localized conditional independence tests (Chi-squared / Fisher-$z$) strictly in the neighborhood of $F$-NODE to output Top-$k$ ranked causes. |
| **Paper 4: PreGAN** | Tuli et al., IEEE INFOCOM 2022 | **Adapted Reproduction** | - **Architecture**: Graph Attention Network (GAT) + GRU encoder with prototypical triplet loss for few-shot fault classification.<br>- **Adaptation**: Input features mapped from DeFog container traces to Edge-IIoTset telemetry. GAN generator outputs delta migration recommendations evaluated against baseline scheduler under simulated edge nodes. |

---

## 8. Remaining Methodological Limitations

1. **Hardware Power Measurement**: In virtualized, containerized, or headless cloud environments where physical hardware power meters or running average power limit (RAPL) interfaces are unexposed by the hypervisor, physical energy measurements cannot be obtained. The benchmark relies on calibrated edge CPU power models and must explicitly disclose this as an estimation.
2. **Network Protocol Emulation**: Because Edge-IIoTset is a real-world recorded network telemetry dataset, closed-loop task migrations cannot alter the historical packets inside the dataset itself. Mitigation impact must be evaluated in a coupled edge execution simulator fed by the real dataset stream.
3. **Cross-Domain Evaluation**: While detection and root-cause localization are measured directly from the dataset's ground truth, recovery comparisons necessarily evaluate the algorithmic decision quality within a standardized edge cluster simulator.

---

*This revised specification is finalized and replaces all prior methodological drafts. Algorithms and benchmark harnesses will adhere strictly to these definitions.*
