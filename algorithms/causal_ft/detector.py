"""
Online Streaming Fault and Anomaly Detector.

Processes strictly observable BenchmarkInput records one-by-one.
Computes anomaly scores using baseline reconstruction distance and online
exponential moving deviation without access to ground truth labels or future records.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np
import math

from algorithms.base import DetectionResult
from evaluation.context import BenchmarkInput


@dataclass
class DetectorConfig:
    """Configuration for streaming detector."""
    threshold: float = 0.65
    ewma_alpha: float = 0.2
    min_variance_epsilon: float = 1e-4
    max_score_cap: float = 5.0


class StreamingCausalDetector:
    """
    Streaming online anomaly detector for edge telemetry.
    Maintains bounded online state and operates strictly on observable features.
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.is_fitted: bool = False
        self.baseline_mean: Optional[np.ndarray] = None
        self.baseline_std: Optional[np.ndarray] = None
        self.dim: int = 62

        # Online running state
        self.ewma_mean: Optional[np.ndarray] = None
        self.stream_count: int = 0

    def fit(self, training_data: np.ndarray) -> None:
        """
        Fit detector baseline on training split feature vectors.
        Operates strictly on unlabeled training telemetry.
        """
        if training_data is None or len(training_data) == 0:
            return

        self.dim = training_data.shape[1]
        self.baseline_mean = np.mean(training_data, axis=0)
        self.baseline_std = np.std(training_data, axis=0) + self.config.min_variance_epsilon
        self.is_fitted = True

    def detect(self, record: BenchmarkInput) -> DetectionResult:
        """
        Evaluate a single streaming observation BenchmarkInput.
        Returns standardized DetectionResult.
        """
        vec = record.feature_vector
        if vec is None or len(vec) == 0:
            return DetectionResult(is_anomaly=False, anomaly_score=0.0, confidence=1.0)

        dim = len(vec)
        if not self.is_fitted or self.baseline_mean is None or len(self.baseline_mean) != dim:
            # Cold-start online estimation fallback
            self.baseline_mean = np.zeros(dim)
            self.baseline_std = np.ones(dim)
            self.is_fitted = True

        # 1. Standardized z-deviation from baseline
        z_scores = np.abs((vec - self.baseline_mean) / self.baseline_std)
        normalized_deviation = float(np.mean(z_scores))

        # 2. Online EWMA temporal drift update
        if self.ewma_mean is None:
            self.ewma_mean = vec.copy()
        else:
            self.ewma_mean = (1.0 - self.config.ewma_alpha) * self.ewma_mean + self.config.ewma_alpha * vec

        ewma_deviation = float(np.mean(np.abs((vec - self.ewma_mean) / self.baseline_std)))

        # 3. Composite score normalized to [0, 1]
        raw_score = 0.7 * normalized_deviation + 0.3 * ewma_deviation
        # Sigmoid scaling around threshold
        scaled_score = 1.0 / (1.0 + math.exp(-1.5 * (raw_score - 1.5)))
        scaled_score = max(0.0, min(1.0, scaled_score))

        is_anomaly = (scaled_score >= self.config.threshold)
        confidence = abs(scaled_score - self.config.threshold) * 2.0
        confidence = max(0.1, min(1.0, confidence))

        self.stream_count += 1

        return DetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=round(scaled_score, 4),
            predicted_class="FAULT" if is_anomaly else "NORMAL",
            confidence=round(confidence, 4),
            raw_output={
                "stream_position": record.stream_position,
                "timestamp": record.timestamp,
                "normalized_deviation": round(normalized_deviation, 4),
                "ewma_deviation": round(ewma_deviation, 4),
            }
        )

    def reset(self) -> None:
        """Reset running online EWMA state while preserving fitted baseline."""
        self.ewma_mean = None
        self.stream_count = 0
