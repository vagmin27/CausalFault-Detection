"""
BWOAIF: Bilateral-Weighted Online Adaptive Isolation Forest.
Based on Hannák et al. (2023).

Architecture:
- Online Isolation Forest with age-tiered trees.
- Streaming batch updates for continuous concept drift adaptation.
- Bilateral weighting combining exponential age decay and score sensitivity.
- Strictly detection-focused baseline; no recovery or RCA actuation.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Set, List
import numpy as np
import math

from algorithms.base import (
    BaseFaultToleranceAlgorithm,
    AlgorithmCapability,
    DetectionResult,
)
from evaluation.context import BenchmarkInput
from .model import StreamingIsolationTree, c_factor


@dataclass
class BWOAIFConfig:
    """Configuration for BWOAIF streaming anomaly detector."""
    num_trees: int = 25
    sub_sample_size: int = 64
    max_tree_height: int = 8
    batch_size: int = 50
    decay_beta: float = 0.1  # Age decay factor
    threshold: float = 0.60
    random_seed: int = 42


class BWOAIAlgorithm(BaseFaultToleranceAlgorithm):
    """
    BWOAIF streaming anomaly detector (Hannák et al., 2023).
    Adapts to concept drift in streaming telemetry using bilateral-weighted isolation trees.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.config = BWOAIFConfig(
            num_trees=cfg.get("num_trees", 25),
            batch_size=cfg.get("batch_size", 50),
            threshold=cfg.get("threshold", 0.60),
            decay_beta=cfg.get("decay_beta", 0.1),
        )
        self.trees: List[StreamingIsolationTree] = []
        self.current_batch_data: List[np.ndarray] = []
        self.current_batch_index: int = 0
        self.c_psi: float = c_factor(self.config.sub_sample_size)
        self.is_fitted: bool = False
        self.is_running: bool = False

    @property
    def name(self) -> str:
        return "BWOAIF (Hannák et al., 2023)"

    @property
    def paper_id(self) -> str:
        return "paper2_bwoaif"

    @property
    def supported_capabilities(self) -> Set[AlgorithmCapability]:
        return {
            AlgorithmCapability.STREAMING_DETECTION,
        }

    def initialize(self, config: Any = None) -> None:
        if config is not None and isinstance(config, dict):
            if "threshold" in config:
                self.config.threshold = float(config["threshold"])
            if "num_trees" in config:
                self.config.num_trees = int(config["num_trees"])
        self.reset()

    def fit(self, training_data: Any = None) -> None:
        """
        Initializes the initial forest using normal training observations.
        Does not access labels or future test records.
        """
        if training_data is None:
            return

        if hasattr(training_data, "values"):
            X = training_data.values
        elif isinstance(training_data, np.ndarray):
            X = training_data
        else:
            return

        np.random.seed(self.config.random_seed)
        self.trees = []
        for _ in range(self.config.num_trees):
            tree = StreamingIsolationTree(
                max_height=self.config.max_tree_height,
                sub_sample_size=self.config.sub_sample_size,
            )
            tree.fit(X, current_batch=0)
            self.trees.append(tree)

        self.is_fitted = True

    def start(self) -> None:
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False

    def _compute_bilateral_weights(self) -> np.ndarray:
        """
        Computes composite weights W_i = w_time(i) * w_score(i) for all trees.
        """
        weights = []
        for tree in self.trees:
            # 1. Time decay weight
            age_diff = max(0, self.current_batch_index - tree.age_batch)
            w_time = math.exp(-self.config.decay_beta * age_diff)
            # 2. Performance weight
            w_score = tree.anomaly_performance_weight
            weights.append(w_time * w_score)

        w_arr = np.array(weights, dtype=np.float64)
        sum_w = np.sum(w_arr)
        if sum_w <= 0.0:
            return np.ones(len(self.trees)) / len(self.trees)
        return w_arr / sum_w

    def detect(self, record: BenchmarkInput) -> DetectionResult:
        """
        Computes online anomaly score for streaming observation BenchmarkInput.
        """
        vec = record.feature_vector

        # Fallback initialization if fit was not called
        if not self.trees:
            dummy_mat = np.tile(vec, (self.config.sub_sample_size, 1))
            self.fit(dummy_mat)

        # 1. Compute path lengths across all trees
        paths = np.array([tree.path_length(vec) for tree in self.trees], dtype=np.float64)

        # 2. Bilateral weighting
        weights = self._compute_bilateral_weights()
        weighted_avg_path = float(np.sum(weights * paths))

        # 3. Anomaly score: s = 2^(-E(h) / c(psi))
        if self.c_psi > 0.0:
            score = float(2.0 ** (-weighted_avg_path / self.c_psi))
        else:
            score = 0.5

        score = max(0.0, min(1.0, score))
        is_anomaly = (score >= self.config.threshold)
        confidence = float(min(1.0, abs(score - self.config.threshold) * 2.0))

        # 4. Stream batch buffering and online tree replacement
        self.current_batch_data.append(vec.copy())
        if len(self.current_batch_data) >= self.config.batch_size:
            self._update_forest()

        return DetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=round(score, 4),
            predicted_class="ANOMALY" if is_anomaly else "NORMAL",
            confidence=round(confidence, 4),
            raw_output={
                "weighted_path_length": round(weighted_avg_path, 4),
                "threshold": self.config.threshold,
                "current_batch": self.current_batch_index,
                "stream_position": record.stream_position,
            }
        )

    def _update_forest(self) -> None:
        """Replaces the oldest tree with a newly fitted tree on the recent streaming batch."""
        if not self.current_batch_data:
            return

        batch_mat = np.array(self.current_batch_data)
        self.current_batch_index += 1

        # Find oldest tree (minimum age_batch)
        oldest_idx = int(np.argmin([t.age_batch for t in self.trees]))

        new_tree = StreamingIsolationTree(
            max_height=self.config.max_tree_height,
            sub_sample_size=self.config.sub_sample_size,
        )
        new_tree.fit(batch_mat, current_batch=self.current_batch_index)
        self.trees[oldest_idx] = new_tree

        self.current_batch_data.clear()

    def process(self, record: BenchmarkInput) -> DetectionResult:
        """Evaluates single streaming record."""
        return self.detect(record)

    def reset(self) -> None:
        self.current_batch_data.clear()
        self.current_batch_index = 0
        self.trees = []
        self.is_fitted = False
