# Reference Algorithm 2: BWOAIF Methodology Specification

## 1. Paper Reference and Classification
- **Title**: Bilateral-Weighted Online Adaptive Isolation Forest for anomaly detection in streaming data
- **Authors & Year**: Hannák et al. (2023), Neurocomputing / Pattern Recognition.
- **Classification**: **Adapted Reproduction** (adapted to streaming Edge-IIoTset schema from original financial/sensor benchmark streams).

---

## 2. Algorithm Objective
BWOAIF is a streaming, unsupervised anomaly-detection algorithm designed to handle non-stationary data streams and concept drift without requiring complete periodic retraining. It maintains an ensemble of age-grouped isolation trees and weights their isolation scores bilaterally (via exponential time decay and score responsiveness).

---

## 3. Original Architecture vs. Implemented Architecture

### Original Architecture:
- Streaming isolation forest consisting of $M$ randomized binary isolation trees.
- Trees organized into age tiers.
- Streaming sliding batches: as each batch arrives, the oldest or lowest-performing tree is replaced by a newly trained tree.
- Bilateral weighting:
  - *Time decay weight*: $w_{\text{time}}(i) = e^{-\beta (t_{\text{current}} - t_{\text{tree}_i})}$
  - *Anomaly sensitivity weight*: $w_{\text{score}}(i)$ prioritizing trees responsive to novel variance.
  - Overall anomaly score: $s(x) = 2^{-\bar{h}(x) / c(\psi)}$.

### Implemented Architecture in this Framework:
- Implemented as [`BWOAIAlgorithm`](file:///c:/Users/Dell/Desktop/CausalFault-Detection/algorithms/bwoaif/algorithm.py#L38-L187) conforming strictly to [`BaseFaultToleranceAlgorithm`](file:///c:/Users/Dell/Desktop/CausalFault-Detection/algorithms/base.py#L61-L149).
- Single-record streaming input via `BenchmarkInput`.
- Internal batch buffer of size $B = 50$: upon buffer saturation, replaces the oldest isolation tree in the ensemble ($M = 25$ trees) with a freshly trained tree.
- Bilateral weighting updates on every evaluation step.

---

## 4. Capability Boundaries and Unsupported Metrics
- **Supported Capabilities**:
  - `STREAMING_DETECTION`
- **Explicitly Unsupported Capabilities**:
  - `RESOURCE_PREDICTION`: **False**
  - `CAUSAL_ROOT_CAUSE_ANALYSIS`: **False**
  - `PREEMPTIVE_MIGRATION`: **False**
  - `CLOSED_LOOP_FAULT_TOLERANCE`: **False**
- **Evaluation Rule**: BWOAIF is an anomaly-detection baseline only. Root-cause localization and recovery actuation metrics (MTTR, diagnosis latency, migration bandwidth) are marked strictly as **`NOT_APPLICABLE` / `NR`**. Calls to `diagnose()` or `recover()` raise `NotImplementedError`.

---

## 5. Input Features and Training Procedure
- **Input Features**: 62-dimensional normalized telemetry feature vector.
- **Training Procedure**: Initial ensemble initialized on normal training data via `fit()`. Continuous online adaptation occurs incrementally as batches are consumed during streaming without label leakage.
