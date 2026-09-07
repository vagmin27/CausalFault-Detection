# Reference Algorithm 3: RCD Methodology Specification

## 1. Paper Reference and Classification
- **Title**: Root Cause Analysis of Failures in Microservices through Causal Discovery
- **Authors & Year**: Ikram et al. (2022), IEEE Transactions on Software Engineering / IEEE/ACM ASE.
- **Classification**: **Adapted Reproduction** (adapted to streaming edge telemetry from original microservice trace/metric graphs).

---

## 2. Algorithm Objective
RCD is a localized causal discovery algorithm designed for root-cause analysis (RCA). Instead of discovering a full global causal DAG over all pairs of system metrics (which is computationally intractable and prone to error in high dimensions), RCD formulates a failure intervention indicator ($F$-node) and performs localized constraint-based conditional independence tests restricted to the neighborhood of $F$.

---

## 3. Core Scientific Distinction from Proposed Causal FT
- **Proposed Causal FT**:
  - Uses a **Domain-Defined DAG** based on networking protocol specifications.
  - Computes **Structural Equation Model (SEM) exogenous parent residuals**.
  - Focuses on real-time streaming causal attribution.
- **Reference Algorithm RCD**:
  - Uses **Data-Driven Localized Causal Discovery**.
  - Formulates an explicit $F$-node intervention indicator (0 = normal, 1 = failure).
  - Employs **Fisher's Z-transform partial correlation conditional independence tests** to orient edges connecting directly into $F$.
  - Focuses strictly on root-cause localization (RCA); does not perform streaming detection or recovery actuation.

---

## 4. Architecture and Algorithm Pipeline

1. **Failure Indicator Formulation ($F$-node)**:
   Given normal telemetry window $\mathcal{D}_{\text{normal}}$ and failure window $\mathcal{D}_{\text{anomalous}}$, assign:
   $$F = 0 \text{ for } x \in \mathcal{D}_{\text{normal}}, \quad F = 1 \text{ for } x \in \mathcal{D}_{\text{anomalous}}$$
2. **Unconditional Screening**:
   Test unconditional independence $X_i \perp\!\!\perp F \mid \emptyset$ using Fisher-Z test ($p > \alpha$). Variables independent of the failure state are immediately pruned.
3. **Localized Conditional Independence Testing**:
   For each surviving candidate $X_i$, test conditional independence given sibling subsets $S \subset \text{Candidates} \setminus \{X_i\}$ ($|S| \le 2$):
   $$\text{corr}(X_i, F \mid S) = 0$$
   If $X_i$ becomes conditionally independent of $F$ given $S$, the edge $X_i - F$ is removed (indicating $X_i$ is merely a secondary correlation or downstream symptom mediated by $S$).
4. **Top-k Root-Cause Ranking**:
   Remaining direct causes of $F$ are ranked in descending order of their Fisher-Z test statistics to output Top-1, Top-3, and Top-5 root causes with p-values and confidence scores.

---

## 5. Capability Boundaries and Unsupported Metrics
- **Supported Capabilities**:
  - `CAUSAL_ROOT_CAUSE_ANALYSIS`
- **Explicitly Unsupported Capabilities**:
  - `STREAMING_DETECTION`: **False**
  - `RESOURCE_PREDICTION`: **False**
  - `PREEMPTIVE_MIGRATION`: **False**
  - `CLOSED_LOOP_FAULT_TOLERANCE`: **False**
- **Evaluation Contract**: RCD is evaluated solely on RCA metrics (diagnosis latency, Top-k recall where domain proxy ground truth is defensible). Detection and recovery metrics are marked strictly as **`NOT_APPLICABLE` / `NR`**. Calls to `detect()` or `recover()` raise `NotImplementedError`.
