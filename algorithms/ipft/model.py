"""
Neural Network Architecture for IPFT Proactive Resource Usage Prediction.
Based on Theodoropoulos et al. (2022):
"Intelligent Proactive Fault Tolerance at the Edge through Resource Usage Prediction"

Two-channel architecture:
- Channel 1 (Temporal): GRU processing historical local telemetry sequence.
- Channel 2 (Global Context): Dense layers processing cluster/host summary context.
- Head: Fusion layer predicting multi-step ahead resource utilization.
"""

from typing import Tuple
import torch
import torch.nn as nn


class IPFTNeuralPredictor(nn.Module):
    """
    Two-channel GRU + Dense prediction model for edge resource telemetry.
    Predicts normalized future resource utilization (0.0 to 1.0).
    """

    def __init__(
        self,
        sequence_dim: int = 62,
        context_dim: int = 4,
        hidden_dim: int = 32,
        num_gru_layers: int = 1,
    ):
        super().__init__()
        self.sequence_dim = sequence_dim
        self.context_dim = context_dim

        # Channel 1: Temporal sequence processing
        self.gru = nn.GRU(
            input_size=sequence_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
        )

        # Channel 2: Global/cluster context processing
        self.context_fc = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.ReLU(),
        )

        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Bound predicted utilization to [0, 1]
        )

    def forward(
        self,
        seq_tensor: torch.Tensor,
        context_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            seq_tensor: (batch, seq_len, sequence_dim)
            context_tensor: (batch, context_dim)
        Returns:
            predicted_usage: (batch, 1) in range [0, 1]
        """
        gru_out, _ = self.gru(seq_tensor)
        temporal_repr = gru_out[:, -1, :]  # Last hidden state (batch, hidden_dim)

        context_repr = self.context_fc(context_tensor)  # (batch, 16)

        combined = torch.cat([temporal_repr, context_repr], dim=-1)
        pred = self.fusion(combined)
        return pred
