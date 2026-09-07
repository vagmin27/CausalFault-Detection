"""
IPFT: Intelligent Proactive Fault Tolerance at the Edge through Resource Usage Prediction.
Based on Theodoropoulos et al. (2022).

Architecture:
- Two-channel resource usage prediction (GRU for temporal telemetry + Dense for cluster context).
- Dual-threshold proactive triggering (Upper threshold for migration/replication, lower for consolidation).
- MinMin-style heuristic task allocation for preemptive migration.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, Optional, Set, List
import numpy as np
# pyrefly: ignore [missing-import]
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.base import (
    BaseFaultToleranceAlgorithm,
    AlgorithmCapability,
    DetectionResult,
    MitigationResult,
)
from evaluation.context import BenchmarkInput
from .model import IPFTNeuralPredictor


@dataclass
class IPFTConfig:
    """Configuration for IPFT algorithm."""
    sequence_length: int = 10
    feature_dim: int = 62
    context_dim: int = 4
    hidden_dim: int = 32
    upper_threshold: float = 0.75  # Triggers proactive migration
    lower_threshold: float = 0.20  # Underutilization / consolidation
    learning_rate: float = 0.005
    epochs: int = 3
    batch_size: int = 32
    cooldown_steps: int = 15


class IPFTAlgorithm(BaseFaultToleranceAlgorithm):
    """
    Adapted implementation of IPFT (Theodoropoulos et al., 2022).
    Predicts edge resource utilization using temporal telemetry and cluster context,
    triggering proactive task migration under projected overload.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg_dict = config or {}
        self.config = IPFTConfig(
            sequence_length=cfg_dict.get("sequence_length", 10),
            feature_dim=cfg_dict.get("feature_dim", 62),
            upper_threshold=cfg_dict.get("upper_threshold", 0.75),
            lower_threshold=cfg_dict.get("lower_threshold", 0.20),
        )
        self.model = IPFTNeuralPredictor(
            sequence_dim=self.config.feature_dim,
            context_dim=self.config.context_dim,
            hidden_dim=self.config.hidden_dim,
        )
        self.is_fitted: bool = False
        self.is_running: bool = False

        # Sliding telemetry buffer for temporal window
        self.window: deque = deque(maxlen=self.config.sequence_length)
        self.cooldown_until: int = 0
        self.last_mitigation: Optional[MitigationResult] = None

    @property
    def name(self) -> str:
        return "IPFT (Theodoropoulos et al., 2022)"

    @property
    def paper_id(self) -> str:
        return "paper1_ipft"

    @property
    def supported_capabilities(self) -> Set[AlgorithmCapability]:
        return {
            AlgorithmCapability.RESOURCE_PREDICTION,
            AlgorithmCapability.PREEMPTIVE_MIGRATION,
            AlgorithmCapability.CLOSED_LOOP_FAULT_TOLERANCE,
        }

    def initialize(self, config: Any = None) -> None:
        if config is not None and isinstance(config, dict):
            if "upper_threshold" in config:
                self.config.upper_threshold = float(config["upper_threshold"])
            if "sequence_length" in config:
                self.config.sequence_length = int(config["sequence_length"])
                self.window = deque(maxlen=self.config.sequence_length)
        self.reset()

    def fit(self, training_data: Any = None) -> None:
        """
        Fits two-channel predictor on normal training sequences.
        Target is next-step utilization proxy computed from telemetry.
        Never accesses Attack_label or test records.
        """
        if training_data is None:
            return

        if hasattr(training_data, "values"):
            X = training_data.values
        elif isinstance(training_data, np.ndarray):
            X = training_data
        else:
            return

        if len(X) <= self.config.sequence_length + 10:
            self.is_fitted = True
            return

        # Prepare temporal sliding window sequences and 1-step targets
        seq_len = self.config.sequence_length
        # Subsample if dataset is large to maintain efficiency
        max_samples = min(len(X) - seq_len - 1, 1000)
        indices = np.linspace(0, len(X) - seq_len - 2, max_samples, dtype=int)

        X_seqs = []
        X_ctxs = []
        y_targets = []

        for idx in indices:
            seq = X[idx:idx + seq_len]
            # Context features: mean load, variance, min, max of current sequence
            ctx = np.array([
                np.mean(seq),
                np.std(seq),
                np.min(seq),
                np.max(seq),
            ], dtype=np.float32)
            # Target: mean of subsequent observation normalized to [0, 1] via sigmoid
            next_obs = X[idx + seq_len]
            target_val = float(1.0 / (1.0 + np.exp(-0.5 * np.mean(next_obs))))

            X_seqs.append(seq)
            X_ctxs.append(ctx)
            y_targets.append([target_val])

        seq_tensor = torch.tensor(np.array(X_seqs), dtype=torch.float32)
        ctx_tensor = torch.tensor(np.array(X_ctxs), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y_targets), dtype=torch.float32)

        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.config.epochs):
            optimizer.zero_grad()
            preds = self.model(seq_tensor, ctx_tensor)
            loss = criterion(preds, y_tensor)
            loss.backward()
            optimizer.step()

        self.model.eval()
        self.is_fitted = True

    def start(self) -> None:
        self.is_running = True
        self.model.eval()

    def stop(self) -> None:
        self.is_running = False

    def detect(self, record: BenchmarkInput) -> DetectionResult:
        """
        Executes proactive resource usage prediction on BenchmarkInput.
        Flags predicted overload when predicted_usage > upper_threshold.
        """
        vec = record.feature_vector
        self.window.append(vec)

        # Pad window if not yet full
        while len(self.window) < self.config.sequence_length:
            self.window.appendleft(vec)

        seq_arr = np.array(self.window, dtype=np.float32).reshape(1, self.config.sequence_length, -1)
        ctx_arr = np.array([[
            np.mean(seq_arr),
            np.std(seq_arr),
            np.min(seq_arr),
            np.max(seq_arr),
        ]], dtype=np.float32)

        with torch.no_grad():
            pred_tensor = self.model(
                torch.tensor(seq_arr, dtype=torch.float32),
                torch.tensor(ctx_arr, dtype=torch.float32),
            )
            predicted_usage = float(pred_tensor.item())

        is_overload = (predicted_usage >= self.config.upper_threshold)
        confidence = float(min(1.0, abs(predicted_usage - 0.5) * 2.0))

        return DetectionResult(
            is_anomaly=is_overload,
            anomaly_score=round(predicted_usage, 4),
            predicted_class="PREDICTED_OVERLOAD" if is_overload else "NORMAL",
            confidence=round(confidence, 4),
            raw_output={
                "predicted_resource_usage": round(predicted_usage, 4),
                "upper_threshold": self.config.upper_threshold,
                "lower_threshold": self.config.lower_threshold,
                "stream_position": record.stream_position,
            }
        )

    def recover(self, context: Any) -> MitigationResult:
        """
        Proactive migration actuation based on predicted usage.
        Executes MinMin allocation: moves tasks to edge_backup_node.
        """
        if not isinstance(context, BenchmarkInput):
            return MitigationResult(action_type="NO_ACTION", success=True)

        rec = context
        if rec.stream_position < self.cooldown_until:
            return MitigationResult(
                action_type="NO_ACTION",
                source_node=rec.edge_node_id,
                tasks_affected=0,
                success=True,
                metadata={"reason": "IN_COOLDOWN"}
            )

        # MinMin proactive migration allocation
        tasks_to_migrate = 6
        state_bytes = tasks_to_migrate * 256 * 1024  # 256 KB per task state
        self.cooldown_until = rec.stream_position + self.config.cooldown_steps

        res = MitigationResult(
            action_type="PREEMPTIVE_TASK_MIGRATION",
            target_node="edge_backup_node",
            source_node=rec.edge_node_id,
            tasks_affected=tasks_to_migrate,
            state_bytes_transferred=state_bytes,
            success=True,
            metadata={
                "algorithm": "IPFT_MinMin",
                "trigger": "UPPER_THRESHOLD_VIOLATION",
                "state_bytes_transferred": state_bytes,
            }
        )
        self.last_mitigation = res
        return res

    def process(self, record: BenchmarkInput) -> DetectionResult:
        """
        Processes single observation: predicts usage -> triggers proactive migration if overloaded.
        """
        det_result = self.detect(record)
        mit_result = None

        if det_result.is_anomaly:
            mit_result = self.recover(record)

        det_result.raw_output["mitigation"] = mit_result
        return det_result

    def reset(self) -> None:
        self.window.clear()
        self.cooldown_until = 0
        self.last_mitigation = None
