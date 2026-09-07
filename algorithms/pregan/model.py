"""
PreGAN Neural Network Models (GAT + GRU + Prototypical Generator).
Based on Tuli et al. (2022):
"PreGAN: Preemptive Migration Prediction Network for Proactive Fault-Tolerant Edge Computing"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAttentionLayer(nn.Module):
    """Self-attention across telemetry feature channels (GAT-style spatial correlation)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.attn = nn.Linear(out_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, in_features)
        h = F.leaky_relu(self.fc(x))
        scores = F.softmax(self.attn(h), dim=-1)
        return h * scores


class PreGANGenerator(nn.Module):
    """
    Generator network combining GAT feature attention, GRU temporal modeling,
    and prototypical embedding to predict preemptive migration need and target host.
    """

    def __init__(
        self,
        feature_dim: int = 62,
        hidden_dim: int = 32,
        num_heads: int = 2,
        prototype_dim: int = 16,
    ):
        super().__init__()
        self.feature_attention = FeatureAttentionLayer(feature_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Prototypical embedding projection
        self.prototype_proj = nn.Linear(hidden_dim, prototype_dim)

        # Preemptive fault prediction head
        self.fault_predictor = nn.Sequential(
            nn.Linear(prototype_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Target host migration decision head (scores migration candidate nodes)
        self.migration_head = nn.Sequential(
            nn.Linear(prototype_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # Target node selection distribution
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, feature_dim)
        attn_out = self.feature_attention(x)
        gru_out, _ = self.gru(attn_out)
        last_step = gru_out[:, -1, :]  # (batch, hidden_dim)

        embedding = F.relu(self.prototype_proj(last_step))
        fault_prob = self.fault_predictor(embedding)
        migration_probs = self.migration_head(embedding)

        return fault_prob, migration_probs, embedding


class PreGANDiscriminator(nn.Module):
    """Discriminator / critic evaluating validity of generated host allocations."""

    def __init__(self, prototype_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(prototype_dim + 2, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, embedding: torch.Tensor, migration_alloc: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([embedding, migration_alloc], dim=-1)
        return self.net(combined)
