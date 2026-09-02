# River Online Streaming Anomaly Detector.
#
# Uses River 0.26.1 `anomaly.HalfSpaceTrees` algorithm for real-time online
# anomaly/fault scoring on streaming TelemetryRecord instances.
#
# Sequential streaming architecture:
#     Record 1 -> score_one -> learn_one
#     Record 2 -> score_one -> learn_one
#     Record 3 -> score_one -> learn_one

import logging
from typing import Dict, Any, List, Optional
# pyrefly: ignore [missing-import]
from river import anomaly
from data.telemetry import TelemetryRecord

logger = logging.getLogger(__name__)


class RiverFaultDetector:
    # Online streaming fault detector powered by River's Half-Space Trees.

    def __init__(
        self,
        n_trees: int = 10,
        height: int = 8,
        window_size: int = 250,
        anomaly_threshold: float = 0.65,
        seed: int = 42,
    ):
        self.n_trees = n_trees
        self.height = height
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.seed = seed

        # Initialize River HalfSpaceTrees streaming model
        self.model = anomaly.HalfSpaceTrees(
            n_trees=self.n_trees,
            height=self.height,
            window_size=self.window_size,
            seed=self.seed,
        )

        self.processed_count: int = 0
        self.score_history: List[float] = []

    def process_record(self, record: TelemetryRecord) -> Dict[str, Any]:
        # Process a single TelemetryRecord observation sequentially.
        # 1. Extract feature dictionary
        # 2. Compute streaming anomaly score using score_one()
        # 3. Update streaming model using learn_one()
        # 4. Determine is_fault threshold decision
        x = record.get_available_features()


        # Score observation before updating (unseen streaming evaluation)
        raw_score = self.model.score_one(x)
        if len(self.score_history) < 10000:
            self.score_history.append(raw_score)

        # Update streaming model with current observation
        self.model.learn_one(x)
        self.processed_count += 1

        # Dynamic warm-up check: lower threshold initially or use configured threshold
        effective_threshold = self.anomaly_threshold
        if self.processed_count < 30:
            effective_threshold = 0.85  # Higher threshold during initial model warm-up

        is_fault = bool(raw_score >= effective_threshold)

        return {
            "is_fault": is_fault,
            "anomaly_score": round(float(raw_score), 4),
            "timestamp": record.timestamp,
            "edge_node_id": record.edge_node_id,
            "detector": "River_HalfSpaceTrees",
        }

    def get_stats(self) -> Dict[str, Any]:
        avg_score = sum(self.score_history) / max(1, len(self.score_history))
        return {
            "detector": "River_HalfSpaceTrees",
            "processed_records": self.processed_count,
            "average_anomaly_score": round(avg_score, 4),
            "threshold": self.anomaly_threshold,
        }
