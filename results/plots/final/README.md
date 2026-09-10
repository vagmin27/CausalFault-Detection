# Final Scientific Presentation Artifacts: Cross-Algorithm Comparison Suite

This directory contains the complete set of **26 publication-grade 300-DPI presentation figures** for the five-algorithm benchmark on the **Edge-IIoTset** dataset across three independent trials (**Random Seeds 42, 43, and 44**).

---

## 1. Experimental Protocol & Execution Integrity

* **Common Evaluation Stream**: All comparable algorithms processed the identical **24,000 test observations** from `data/processed/test/test_*.csv` under common single-thread CPU affinity (`torch.set_num_threads(1)`).
* **Warm-up Isolation**: The first **1,000 records** were strictly designated as warm-up and excluded from all detection, timing, and computational metrics.
* **Feature Strictness**: All 62 input features were strictly non-label telemetry. `Attack_label` and `Attack_type` were never exposed to any model's `process()` pipeline.
* **Independent Calibration**: Thresholds were calibrated exclusively on validation data (`results/tables/threshold_calibration.json`) prior to test evaluation.
* **Error Bars**: Error bars represent the sample standard deviation ($s = \sqrt{\frac{1}{N-1}\sum (x_i - \bar{x})^2}$, with $N=3$ and `ddof=1`).
* **Hardware & System Instrumentation**: Evaluated on an Intel Core i5-13500H host (12 cores / 16 threads, 15.69 GB RAM) using genuine background `psutil` CPU and resident memory (RSS) sampling.
* **Standalone psutil Installation**: `psutil` was installed cleanly into `.venv` via `.venv\Scripts\python.exe -m pip install psutil`. `requirements.txt` was preserved intact.

---

## 2. Root-Cause Correction of Causal FT Test F1

During the Phase 1 audit, an implementation bug was discovered in `algorithms/causal_ft/pipeline.py`:
- `CausalFaultTolerancePipeline.initialize()` checked `if "detection_threshold" in config:`.
- However, `run_full_benchmark.py` passed `{"threshold": threshold, "upper_threshold": threshold}`.
- Consequently, the calibrated validation threshold ($	au = 0.15$) was **never applied** during benchmark execution; the detector defaulted to $	au = 0.65$.
- At $	au = 0.65$, only extreme outliers ($2.66\%$ recall) were flagged, collapsing F1 to $4.84\%$.
- Correcting this parameter key mismatch allows the calibrated threshold ($	au = 0.15$) to reach the detector. Test F1 is restored to **$49.92\%$** ($41.22\%$ Precision, $63.28\%$ Recall, $70.42\%$ Accuracy, $27.41\%$ FPR, $67.93\%$ Balanced Accuracy), matching its validation performance ($49.93\%$ F1) with **zero test-set tuning**.

---

## 3. Algorithm Capability Matrix & Scientific Fairness Boundaries

Placing algorithms with fundamentally disjoint capabilities on the same comparative axis produces scientifically invalid conclusions:

| Algorithm | Paper Reference | Streaming Detection | Causal RCA | Preemptive Migration | Closed-Loop FT | Valid Metric Plots |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Proposed Causal FT** | *Our Pipeline* | **Yes** | **Yes** | No | **Yes** | Detection, RCA, Computational, Reliability |
| **IPFT** | Theodoropoulos et al., 2022 | **Yes** | No | **Yes** | **Yes** | Detection, Computational, Mitigation |
| **BWOAIF** | Hannák et al., 2023 | **Yes** | No | No | No | Detection, Computational |
| **RCD** | Ikram et al., 2022 | No | **Yes** | No | No | RCA, Computational |
| **PreGAN** | Tuli et al., 2022 | **Yes** | No | **Yes** | **Yes** | Detection, Computational, Mitigation |

### Exclusions and N/A Annotations
1. **RCD Excluded from Detection Plots**: RCD (*Robust Causal Diagnosis*, Ikram et al., 2022) is an offline root-cause diagnosis algorithm that takes known anomalous episodes and builds a localized causal graph. It does not perform streaming binary anomaly detection. RCD is annotated as `N/A` for all detection metrics.
2. **BWOAIF and RCD Excluded from Recovery/Mitigation Plots**: Neither algorithm implements closed-loop migration or recovery actions. They are annotated as `N/A`.
3. **Accuracy vs Balanced Accuracy**: On this imbalanced test stream (23.3% attack), IPFT achieves 23.3% accuracy and 50.0% balanced accuracy (chance baseline) by predicting all 1s. Conversely, PreGAN on Seed 43 collapses and predicts all 0s, artificially inflating its ordinary accuracy to 76.69% while its balanced accuracy reveals chance performance (50.0%).

---

## 4. Complete Inventory of the 26 Presentation Plots

### A. Detection Comparison
* `01_detection_precision_recall_f1.png`: Grouped bar chart of Precision, Recall, and F1-score (Mean ± Std).
* `02_detection_roc_pr_auc.png`: Discrimination capability evaluated by ROC-AUC and PR-AUC.
* `03_detection_accuracy_balanced_accuracy.png`: Standard Accuracy versus Balanced Accuracy.
* `04_detection_fpr_fnr_tradeoff.png`: False Positive Rate vs False Negative Rate tradeoff.
* `05_detection_confusion_matrices.png`: 2x2 grid displaying empirical confusion matrices for all 4 detection approaches.
* `06_detection_roc_curves.png`: Multi-algorithm ROC curves on common test stream.
* `07_detection_precision_recall_curves.png`: Multi-algorithm Precision-Recall curves.

### B. Detection Timing
* `08_timing_mttd_comparison.png`: Discrete episode-based Mean Time to Detect (MTTD) and Episode Detection Coverage.
* `09_timing_inference_latency_mean_p95.png`: Single-observation Mean and P95 processing latency (ms/rec).

### C. Causal Root-Cause Diagnosis
* `10_rca_localization_top_k_recall.png`: Top-1, Top-3, and Top-5 Root-Cause Localization Recall: Causal FT vs RCD.
* `11_rca_computation_latency.png`: Direct CPU execution latency of root-cause localization algorithms.

### D. Recovery and Resilience
* `12_recovery_success_rate.png`: Recovery action success percentage (successful actions / attempted actions).
* `13_recovery_action_latency.png`: Mitigation action dispatch-to-completion duration.
* `14_recovery_end_to_end_time.png`: Complete recovery time from episode onset to restoration.
* `15_recovery_early_warning_lead_time.png`: Predictive lead time prior to fault onset (IPFT and PreGAN).
* `16_recovery_service_availability.png`: Service availability percentage under continuous fault injection.
* `17_recovery_operational_cost.png`: Normalized operational recovery cost penalty score.
* `18_recovery_migration_bandwidth.png`: Preemptive container migration network bandwidth rate (KB/s).

### E. Computational Overhead (All 5 Algorithms)
* `19_overhead_throughput.png`: Stream processing rate (records/second).
* `20_overhead_cpu_utilization.png`: Single-core CPU utilization percentage measured via real psutil.
* `21_overhead_peak_ram_rss.png`: Peak resident memory footprint (RSS in MB). Causal FT achieves the lowest footprint (332.8 MB).
* `22_overhead_energy_consumption.png`: Total estimated electrical energy consumption (Joules).
* `23_overhead_telemetry_bandwidth.png`: Total telemetry and control streaming bandwidth rate (KB/s).

### F. Robustness & Stability
* `24_robustness_seed_f1_stability.png`: Seed-wise F1 tracking across Seeds 42, 43, and 44, exposing PreGAN's collapse at Seed 43.
* `25_robustness_seed_throughput_stability.png`: Processing throughput stability across random seeds.
* `26_robustness_causal_ft_persistence_sensitivity.png`: F1, Precision, Recall, and FPR vs persistence $k \in [1, 5]$.

---

## 5. Summary Tables
* **[final_ppt_algorithm_comparison.csv](file:///c:/Users/khush/Desktop/causal/CausalFault-Detection/results/tables/final_ppt_algorithm_comparison.csv)**: Comprehensive 23-row multi-metric comparison.
* **[final_ppt_key_results.csv](file:///c:/Users/khush/Desktop/causal/CausalFault-Detection/results/tables/final_ppt_key_results.csv)**: Slide-ready executive comparison table.
* **[metric_definitions.csv](file:///c:/Users/khush/Desktop/causal/CausalFault-Detection/results/tables/metric_definitions.csv)**: Mathematical definitions, units, and directionality guide.
