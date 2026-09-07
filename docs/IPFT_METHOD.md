# Reference Algorithm 1: IPFT Methodology Specification

## 1. Paper Reference and Classification
- **Title**: Intelligent Proactive Fault Tolerance at the Edge through Resource Usage Prediction
- **Authors & Year**: Theodoropoulos et al. (2022), IEEE Transactions on Network and Service Management / IEEE CloudNet.
- **Classification**: **Adapted Reproduction** (adapted to streaming Edge-IIoTset schema from original OpenStack/telemetry cluster workloads).

---

## 2. Algorithm Objective
IPFT provides proactive, closed-loop fault tolerance for edge computing environments by predicting future resource utilization (CPU, memory, networking) and proactively triggering task migration or replication before physical saturation or service-level objective (SLO) violation occurs.

---

## 3. Original Architecture vs. Implemented Architecture

### Original Architecture:
- Two-channel neural network:
  1. *Local Channel*: Recurrent neural network (GRU/LSTM) capturing temporal trends across sliding windows of edge node telemetry.
  2. *Global Context Channel*: Dense feedforward network encoding cluster-wide metadata (overall utilization, cluster capacity).
- Multi-objective decision head:
  - Upper threshold ($\tau_{\text{upper}}$): Triggers proactive migration using MinMin/MaxMin allocation.
  - Lower threshold ($\tau_{\text{lower}}$): Triggers task consolidation to conserve energy.

### Implemented Architecture in this Framework:
- Implemented as [`IPFTAlgorithm`](file:///c:/Users/Dell/Desktop/CausalFault-Detection/algorithms/ipft/algorithm.py#L39-L198) conforming to [`BaseFaultToleranceAlgorithm`](file:///c:/Users/Dell/Desktop/CausalFault-Detection/algorithms/base.py#L61-L149).
- Two-channel PyTorch model [`IPFTNeuralPredictor`](file:///c:/Users/Dell/Desktop/CausalFault-Detection/algorithms/ipft/model.py#L14-L64):
  - GRU layer: input dimension $D = 62$, sequence length $L = 10$, hidden dimension $H = 32$.
  - Context Dense layer: 4-dimensional statistical summary of local host state (mean, variance, minimum, maximum).
  - Fusion MLP: outputs predicted utilization $\hat{u}_{t+1} \in [0.0, 1.0]$.
- Proactive Actuation:
  - When $\hat{u}_{t+1} \ge \tau_{\text{upper}}$ (default $0.75$), triggers proactive task migration (`PREEMPTIVE_TASK_MIGRATION`) to `edge_backup_node` using MinMin allocation with cooldown to prevent flapping.

---

## 4. Input Features and Training Procedure
- **Input Features**: The 62 normalized numeric features of `BenchmarkInput` processed as a 10-step sequence.
- **Training Procedure**:
  - Fitted exclusively during `fit()` on the designated training split (`data/processed/train/`).
  - Target: next-step normalized system load.
  - Optimization: Adam optimizer with Mean Squared Error (MSE) loss.
  - Ground truth labels (`Attack_label`, `Attack_type`) are **never** seen during training or inference.

---

## 5. Streaming Procedure and Execution Contract
- Each incoming `BenchmarkInput` updates the internal sliding sequence window ($L = 10$).
- When $\hat{u}_{t+1} \ge \tau_{\text{upper}}$, `detect()` returns `is_anomaly=True` with `predicted_class="PREDICTED_OVERLOAD"`.
- `process()` coordinates prediction with `recover()`, outputting standardized `DetectionResult` and `MitigationResult`.
- **Unsupported Capabilities**: Does **not** support Root-Cause Analysis (`CAUSAL_ROOT_CAUSE_ANALYSIS`). Calls to `diagnose()` raise `NotImplementedError`.

---

## 6. Adaptations and Reproducibility Limitations
1. **Telemetry Schema**: The original paper used OpenStack VM metrics; the adapted implementation runs on the 62-dimensional Edge-IIoTset network and protocol telemetry.
2. **Cluster Topology**: Edge cluster task allocations are modeled in an emulated edge host environment, since Edge-IIoTset is an immutable recorded packet trace.
