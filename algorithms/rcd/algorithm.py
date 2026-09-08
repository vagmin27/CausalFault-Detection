# RCD: Root Cause Analysis of Failures in Microservices through Causal Discovery.
# Based on Ikram et al. (2022).
#
# Architecture:
# - Data-driven localized constraint-based causal discovery.
# - F-node (failure/intervention indicator) comparing normal vs. anomalous data.
# - Localized conditional independence testing (PC-style Fisher-Z tests) restricted
#   to the neighborhood of the failure node F.
# - Top-k root-cause ranking by conditional dependence strength.
# - Exclusively an RCA baseline: no streaming detection or recovery.

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, Optional, Set, List, Tuple
import numpy as np

from algorithms.base import (
    BaseFaultToleranceAlgorithm,
    AlgorithmCapability,
    DetectionResult,
    RCADiagnosisResult,
)
from evaluation.context import BenchmarkInput
from .model import fisher_z_test


@dataclass
class RCDConfig:
    # Configuration parameters for RCD algorithm.
    significance_alpha: float = 0.05
    max_conditioning_size: int = 2
    top_k: int = 5
    min_samples: int = 15


class RCDAlgorithm(BaseFaultToleranceAlgorithm):
    # RCD: Root Cause Analysis through Causal Discovery (Ikram et al., 2022).
    # Discovers data-driven causal relationships between metrics and the failure state F.

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.config = RCDConfig(
            significance_alpha=cfg.get("significance_alpha", 0.05),
            max_conditioning_size=cfg.get("max_conditioning_size", 2),
            top_k=cfg.get("top_k", 5),
            min_samples=cfg.get("min_samples", 15),
        )
        self.baseline_normal_data: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.sliding_window: deque = deque(maxlen=50)
        self.is_fitted: bool = False
        self.is_running: bool = False
        self.last_diagnosis: Optional[RCADiagnosisResult] = None

    @property
    def name(self) -> str:
        return "RCD (Ikram et al., 2022)"

    @property
    def paper_id(self) -> str:
        return "paper3_rcd"

    @property
    def supported_capabilities(self) -> Set[AlgorithmCapability]:
        return {
            AlgorithmCapability.CAUSAL_ROOT_CAUSE_ANALYSIS,
        }

    def initialize(self, config: Any = None) -> None:
        if config is not None and isinstance(config, dict):
            if "significance_alpha" in config:
                self.config.significance_alpha = float(config["significance_alpha"])
            if "top_k" in config:
                self.config.top_k = int(config["top_k"])
        self.reset()

    def fit(self, training_data: Any = None) -> None:
        # Stores normal baseline observations to serve as D_normal during diagnosis.
        # Does not access attack labels or test split.
        if training_data is None:
            return

        if hasattr(training_data, "values") and hasattr(training_data, "columns"):
            self.baseline_normal_data = training_data.values
            self.feature_names = list(training_data.columns)
        elif isinstance(training_data, np.ndarray):
            self.baseline_normal_data = training_data
            self.feature_names = [f"f_{i}" for i in range(training_data.shape[1])]

        self.is_fitted = True

    def start(self) -> None:
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False

    def diagnose_root_cause(
        self,
        normal_window: Any = None,
        anomalous_window: Any = None,
    ) -> RCADiagnosisResult:
        # Executes Localized Causal Discovery:
        # 1. Formulates F-node (0 = normal, 1 = anomalous).
        # 2. Tests conditional independence of each metric X_i with F.
        # 3. Prunes conditionally independent variables.
        # 4. Ranks remaining candidate causes by Fisher-Z dependence statistic.
        t0 = time.perf_counter_ns()

        # 1. Resolve normal and anomalous matrices
        X_norm, X_anom, feat_names = self._resolve_windows(normal_window, anomalous_window)

        if len(X_norm) == 0 or len(X_anom) == 0:
            return RCADiagnosisResult(
                ranked_root_causes=[],
                confidence_scores={},
                p_values={},
                execution_time_ms=0.0,
                metadata={"status": "INSUFFICIENT_SAMPLES"},
            )

        n_norm = len(X_norm)
        n_anom = len(X_anom)
        num_features = X_norm.shape[1]

        # Combine into joint dataset with F-node
        X_joint = np.vstack([X_norm, X_anom])
        F_joint = np.concatenate([np.zeros(n_norm), np.ones(n_anom)])

        candidate_scores: Dict[str, float] = {}
        candidate_p_values: Dict[str, float] = {}

        # Step 1: Unconditional screening with F: X_i \perp F | \emptyset
        surviving_indices = []
        for i in range(num_features):
            x_i = X_joint[:, i]
            is_indep, p_val, z_stat = fisher_z_test(
                x_i, F_joint, Z=np.empty((len(F_joint), 0)), alpha=self.config.significance_alpha
            )
            candidate_p_values[feat_names[i]] = round(p_val, 6)
            if not is_indep:
                surviving_indices.append(i)
                candidate_scores[feat_names[i]] = z_stat

        # Step 2: Localized conditional independence testing given sibling conditioning sets
        final_causes: Dict[str, float] = {}
        for idx in surviving_indices:
            x_i = X_joint[:, idx]
            name_i = feat_names[idx]
            other_indices = [j for j in surviving_indices if j != idx]

            is_direct_cause = True
            # Test conditioning sets up to max_conditioning_size
            for cond_idx in other_indices[:self.config.max_conditioning_size]:
                Z_cond = X_joint[:, [cond_idx]]
                is_indep, p_val, z_stat = fisher_z_test(
                    x_i, F_joint, Z_cond, alpha=self.config.significance_alpha
                )
                if is_indep:
                    # Conditionally independent of F given sibling; prune
                    is_direct_cause = False
                    break

            if is_direct_cause:
                final_causes[name_i] = candidate_scores[name_i]

        # Step 3: Top-k ranking by Fisher-Z test statistic magnitude
        if not final_causes:
            # Fallback to top unconditional candidates if all were conditionally pruned
            sorted_candidates = sorted(candidate_scores.keys(), key=lambda k: candidate_scores[k], reverse=True)
        else:
            sorted_candidates = sorted(final_causes.keys(), key=lambda k: final_causes[k], reverse=True)

        ranked = sorted_candidates[:self.config.top_k]

        # Normalize confidence scores
        scores_arr = np.array([candidate_scores.get(k, 1.0) for k in ranked], dtype=np.float64)
        sum_scores = np.sum(scores_arr) + 1e-6
        confidence_scores = {k: round(float(candidate_scores.get(k, 1.0) / sum_scores), 4) for k in ranked}

        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

        res = RCADiagnosisResult(
            ranked_root_causes=ranked,
            confidence_scores=confidence_scores,
            p_values={k: candidate_p_values.get(k, 1.0) for k in ranked},
            execution_time_ms=round(elapsed_ms, 4),
            metadata={
                "algorithm": "RCD_LocalizedCausalDiscovery",
                "top_1": ranked[0] if ranked else "UNKNOWN",
                "top_3": ranked[:3],
                "top_5": ranked[:5],
                "num_surviving_direct_causes": len(final_causes),
                "total_features_evaluated": num_features,
            }
        )
        self.last_diagnosis = res
        return res

    def _resolve_windows(
        self,
        normal_window: Any,
        anomalous_window: Any,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        # Extracts numerical arrays from window objects or baseline state.
        # 1. Resolve anomalous window
        if isinstance(anomalous_window, BenchmarkInput):
            X_anom = anomalous_window.feature_vector.reshape(1, -1)
            feat_names = list(anomalous_window.features.keys())
        elif isinstance(anomalous_window, np.ndarray):
            X_anom = anomalous_window
            feat_names = self.feature_names or [f"f_{i}" for i in range(X_anom.shape[1])]
        elif hasattr(anomalous_window, "values"):
            X_anom = anomalous_window.values
            feat_names = list(anomalous_window.columns)
        else:
            X_anom = np.empty((0, 0))
            feat_names = []

        # 2. Resolve normal window
        if normal_window is not None:
            if hasattr(normal_window, "values"):
                X_norm = normal_window.values
            elif isinstance(normal_window, np.ndarray):
                X_norm = normal_window
            else:
                X_norm = self.baseline_normal_data if self.baseline_normal_data is not None else np.empty((0, 0))
        else:
            X_norm = self.baseline_normal_data if self.baseline_normal_data is not None else np.empty((0, 0))

        # Align lengths if anomalous window is single record
        if len(X_anom) == 1 and len(X_norm) > 0:
            # Replicate anomalous record with tiny variance for statistical test
            noise = np.random.normal(0.0, 1e-4, size=(len(X_norm), X_anom.shape[1]))
            X_anom = np.tile(X_anom, (len(X_norm), 1)) + noise

        return X_norm, X_anom, feat_names

    def process(self, record: BenchmarkInput) -> DetectionResult:
        # Consumes streaming observation into sliding telemetry window.
        # RCD is explicitly an RCA algorithm; it does not perform online anomaly detection.
        # Returns a non-actionable DetectionResult placeholder while updating internal window.
        self.sliding_window.append(record.feature_vector)
        return DetectionResult(
            is_anomaly=False,
            anomaly_score=0.0,
            confidence=0.0,
            raw_output={
                "stream_position": record.stream_position,
                "capability_note": "RCD provides CAUSAL_ROOT_CAUSE_ANALYSIS only; streaming detection is unsupported.",
                "diagnosis": None,
            }
        )

    def reset(self) -> None:
        self.sliding_window.clear()
        self.last_diagnosis = None
