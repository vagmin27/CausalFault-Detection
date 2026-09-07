"""
PreGAN: Preemptive Migration Prediction Network for Proactive Fault-Tolerant Edge Computing.
Based on Tuli et al. (2022).

Architecture:
- Feature attention (GAT-style spatial correlation).
- GRU for temporal sequence modeling.
- Prototypical embedding space for nominal vs. fault states.
- Preemptive migration generator and self-supervised discriminator.
- Proactive closed-loop task migration before host failure.
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
from .model import PreGANGenerator, PreGANDiscriminator


@dataclass
class PreGANConfig:
    """Configuration for PreGAN algorithm."""
    sequence_length: int = 10
    feature_dim: int = 62
    hidden_dim: int = 32
    prototype_dim: int = 16
    threshold: float = 0.70
    learning_rate: float = 0.003
    epochs: int = 3
    cooldown_steps: int = 15


class PreGANAlgorithm(BaseFaultToleranceAlgorithm):
    """
    Adapted implementation of PreGAN (Tuli et al., 2022).
    Generates preemptive task migration decisions using GAT-GRU prototypical embeddings.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.config = PreGANConfig(
            sequence_length=cfg.get("sequence_length", 10),
            feature_dim=cfg.get("feature_dim", 62),
            threshold=cfg.get("threshold", 0.70),
        )
        self.generator = PreGANGenerator(
            feature_dim=self.config.feature_dim,
            hidden_dim=self.config.hidden_dim,
            prototype_dim=self.config.prototype_dim,
        )
        self.discriminator = PreGANDiscriminator(
            prototype_dim=self.config.prototype_dim,
        )
        self.window: deque = deque(maxlen=self.config.sequence_length)
        self.cooldown_until: int = 0
        self.is_fitted: bool = False
        self.is_running: bool = False
        self.last_mitigation: Optional[MitigationResult] = None

    @property
    def name(self) -> str:
        return "PreGAN (Tuli et al., 2022)"

    @property
    def paper_id(self) -> str:
        return "paper4_pregan"

    @property
    def supported_capabilities(self) -> Set[AlgorithmCapability]:
        return {
            AlgorithmCapability.RESOURCE_PREDICTION,
            AlgorithmCapability.PREEMPTIVE_MIGRATION,
            AlgorithmCapability.CLOSED_LOOP_FAULT_TOLERANCE,
        }

    def initialize(self, config: Any = None) -> None:
        if config is not None and isinstance(config, dict):
            if "threshold" in config:
                self.config.threshold = float(config["threshold"])
            if "sequence_length" in config:
                self.config.sequence_length = int(config["sequence_length"])
                self.window = deque(maxlen=self.config.sequence_length)
        self.reset()

    def fit(self, training_data: Any = None) -> None:
        """
        Adversarial / prototypical training of generator and discriminator on normal sequences.
        Does not access attack labels or test records.
        """
        if training_data is None:
            return

        if hasattr(training_data, "values"):
            X = training_data.values
        elif isinstance(training_data, np.ndarray):
            X = training_data
        else:
            return

        seq_len = self.config.sequence_length
        if len(X) <= seq_len + 10:
            self.is_fitted = True
            return

        # Prepare normal training sequences
        max_samples = min(len(X) - seq_len, 800)
        indices = np.linspace(0, len(X) - seq_len - 1, max_samples, dtype=int)
        seqs = np.array([X[i:i + seq_len] for i in indices], dtype=np.float32)
        tensor_x = torch.tensor(seqs, dtype=torch.float32)

        opt_g = optim.Adam(self.generator.parameters(), lr=self.config.learning_rate)
        opt_d = optim.Adam(self.discriminator.parameters(), lr=self.config.learning_rate)
        bce_loss = nn.BCELoss()

        self.generator.train()
        self.discriminator.train()

        # Adversarial training loop
        batch_size = 32
        n_batches = min(len(tensor_x) // batch_size, 10)

        for _ in range(self.config.epochs):
            for b in range(n_batches):
                batch_x = tensor_x[b * batch_size:(b + 1) * batch_size]

                # Train Generator
                opt_g.zero_grad()
                fault_prob, mig_probs, embedding = self.generator(batch_x)
                # Normal sequences have target 0 for fault probability
                zero_targets = torch.zeros_like(fault_prob)
                loss_g_pred = bce_loss(fault_prob, zero_targets)
                # Adversarial objective: fool discriminator
                validity = self.discriminator(embedding, mig_probs)
                ones = torch.ones_like(validity)
                loss_g_adv = bce_loss(validity, ones)
                loss_g = loss_g_pred + 0.1 * loss_g_adv
                loss_g.backward()
                opt_g.step()

                # Train Discriminator
                opt_d.zero_grad()
                real_alloc = torch.tensor([[1.0, 0.0]], dtype=torch.float32).repeat(len(batch_x), 1)
                real_validity = self.discriminator(embedding.detach(), real_alloc)
                fake_validity = self.discriminator(embedding.detach(), mig_probs.detach())
                loss_d = bce_loss(real_validity, ones) + bce_loss(fake_validity, zero_targets)
                loss_d.backward()
                opt_d.step()

        self.generator.eval()
        self.discriminator.eval()
        self.is_fitted = True

    def start(self) -> None:
        self.is_running = True
        self.generator.eval()

    def stop(self) -> None:
        self.is_running = False

    def detect(self, record: BenchmarkInput) -> DetectionResult:
        """
        Evaluates preemptive fault probability using GAT-GRU generator.
        """
        vec = record.feature_vector
        self.window.append(vec)

        while len(self.window) < self.config.sequence_length:
            self.window.appendleft(vec)

        seq_arr = np.array(self.window, dtype=np.float32).reshape(1, self.config.sequence_length, -1)
        tensor_in = torch.tensor(seq_arr, dtype=torch.float32)

        with torch.no_grad():
            fault_prob_tensor, mig_probs_tensor, _ = self.generator(tensor_in)
            fault_prob = float(fault_prob_tensor.item())
            target_node_idx = int(torch.argmax(mig_probs_tensor, dim=-1).item())

        is_fault_predicted = (fault_prob >= self.config.threshold)
        confidence = float(min(1.0, abs(fault_prob - self.config.threshold) * 2.0))

        return DetectionResult(
            is_anomaly=is_fault_predicted,
            anomaly_score=round(fault_prob, 4),
            predicted_class="PREDICTED_FAULT" if is_fault_predicted else "NORMAL",
            confidence=round(confidence, 4),
            raw_output={
                "fault_probability": round(fault_prob, 4),
                "target_host_index": target_node_idx,
                "threshold": self.config.threshold,
                "stream_position": record.stream_position,
            }
        )

    def recover(self, context: Any) -> MitigationResult:
        """
        Executes preemptive task migration generated by PreGAN.
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

        tasks_to_migrate = 8
        state_bytes = tasks_to_migrate * 512 * 1024  # 512 KB per task container state
        self.cooldown_until = rec.stream_position + self.config.cooldown_steps

        res = MitigationResult(
            action_type="PREEMPTIVE_TASK_MIGRATION",
            target_node="edge_backup_node",
            source_node=rec.edge_node_id,
            tasks_affected=tasks_to_migrate,
            state_bytes_transferred=state_bytes,
            success=True,
            metadata={
                "algorithm": "PreGAN_Generator",
                "trigger": "PROTOTYPICAL_FAULT_PREDICTION",
                "state_bytes_transferred": state_bytes,
            }
        )
        self.last_mitigation = res
        return res

    def process(self, record: BenchmarkInput) -> DetectionResult:
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
