# Reference Algorithm 4: PreGAN Methodology Specification

## 1. Paper Reference and Classification
- **Title**: PreGAN: Preemptive Migration Prediction Network for Proactive Fault-Tolerant Edge Computing
- **Authors & Year**: Tuli et al. (2022), IEEE Transactions on Network and Service Management / IEEE INFOCOM.
- **Classification**: **Adapted Reproduction** (adapted to 62-dimensional streaming Edge-IIoTset telemetry from original Cloud-Edge iFogSim testbed).

---

## 2. Algorithm Objective
PreGAN implements proactive fault-tolerant edge computing by predicting upcoming host/service failure episodes and generating preemptive task container migration plans to optimal surrogate edge nodes before quality of service (QoS) degradation occurs.

---

## 3. Architecture Details

### Original Architecture:
- Multi-component deep network:
  1. *Graph Attention Network (GAT)*: Models topological relationships and inter-metric correlations across edge nodes and telemetry dimensions.
  2. *Recurrent Neural Network (GRU)*: Encodes temporal sequence dynamics.
  3. *Prototypical Embedding*: Projects edge states into a metric space with learned prototypes for nominal and failure patterns.
  4. *Generator*: Predicts failure probability and outputs target host allocation probabilities.
  5. *Discriminator*: Adversarial critic scoring allocation realism against real scheduling distributions.

### Implemented Architecture in this Framework:
- Implemented as [`PreGANAlgorithm`](file:///c:/Users/Dell/Desktop/CausalFault-Detection/algorithms/pregan/algorithm.py#L39-L215) inheriting from [`BaseFaultToleranceAlgorithm`](file:///c:/Users/Dell/Desktop/CausalFault-Detection/algorithms/base.py#L61-L149).
- Feature self-attention layer [`FeatureAttentionLayer`](file:///c:/Users/Dell/Desktop/CausalFault-Detection/algorithms/pregan/model.py#L11-L24) computing GAT-style spatial correlations across the 62 input features.
- GRU layer processing temporal sequences ($L = 10$).
- Prototypical projection producing dense embeddings ($d_{\text{proto}} = 16$).
- Generator and Discriminator trained adversarially on normal training sequences.
- Proactive Actuation: When failure probability $p_{\text{fault}} \ge \tau_{\text{pregan}}$ (default $0.70$), generates proactive container migration (`PREEMPTIVE_TASK_MIGRATION`) to candidate backup hosts with cooldown logic.

---

## 4. Capability Boundaries and Unsupported Metrics
- **Supported Capabilities**:
  - `RESOURCE_PREDICTION` (Fault / Overload Probability)
  - `PREEMPTIVE_MIGRATION`
  - `CLOSED_LOOP_FAULT_TOLERANCE`
- **Explicitly Unsupported Capabilities**:
  - `CAUSAL_ROOT_CAUSE_ANALYSIS`: **False**
- **Evaluation Contract**: PreGAN is evaluated on proactive detection, execution latency, resource overhead, and migration bandwidth. Root-cause localization metrics (RCA recall, diagnosis latency) are marked strictly as **`NOT_APPLICABLE` / `NR`**. Calls to `diagnose()` raise `NotImplementedError`.

---

## 5. Input Features and Training Procedure
- **Input Features**: 62-dimensional normalized telemetry vector.
- **Training Procedure**: Trained during `fit()` using normal training sequences via mini-batch adversarial optimization. Target attack labels are **never** provided.
