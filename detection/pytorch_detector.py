# PyTorch Telemetry Autoencoder Anomaly Detector.
#
# Implements an Autoencoder neural network for real-time anomaly detection based on
# reconstruction error. Features separate training (offline baseline) and inference phases.

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.optim as optim
from data.telemetry import TelemetryRecord

logger = logging.getLogger(__name__)


class TelemetryAutoencoder(nn.Module):
    # Symmetric Autoencoder Neural Network.

    def __init__(self, input_dim: int = 7, latent_dim: int = 3):
        super(TelemetryAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 12),
            nn.ReLU(),
            nn.Linear(12, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


class PyTorchFaultDetector:
    # PyTorch Anomaly Detector wrapping TelemetryAutoencoder with model persistence
    # and real-time streaming inference.

    def __init__(
        self,
        input_dim: int = 7,
        latent_dim: int = 3,
        threshold: float = 0.15,
        lr: float = 0.005,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.threshold = threshold
        self.lr = lr

        self.model = TelemetryAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.is_trained: bool = False
        self.feature_names = TelemetryRecord.get_feature_names()

    def train_on_baseline(
        self,
        records: List[TelemetryRecord],
        epochs: int = 40,
        batch_size: int = 16,
    ):
        # Train Autoencoder on normal baseline telemetry records.
        # Filter for normal records if label available
        normal_recs = [r for r in records if r.fault_label == 0]
        if not normal_recs:
            normal_recs = records

        logger.info(f"Training PyTorch Autoencoder on {len(normal_recs)} normal baseline records...")

        data_matrix = np.array([r.to_feature_vector(self.feature_names) for r in normal_recs], dtype=np.float32)

        # Basic standardization normalization
        self.mean = np.mean(data_matrix, axis=0)
        self.std = np.std(data_matrix, axis=0) + 1e-6
        norm_data = (data_matrix - self.mean) / self.std

        tensor_data = torch.from_numpy(norm_data)
        dataset = torch.utils.data.TensorDataset(tensor_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataloader:
                x_batch = batch[0]
                self.optimizer.zero_grad()
                recon = self.model(x_batch)
                loss = self.criterion(recon, x_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(x_batch)

        self.is_trained = True
        logger.info("PyTorch Autoencoder training complete.")

    def process_record(self, record: TelemetryRecord) -> Dict[str, Any]:
        # Perform real-time inference on a single TelemetryRecord.
        # Calculate reconstruction MSE error.
        if not self.is_trained:
            # Quick self-calibration on first record if not pre-trained
            dummy_vec = np.array(record.to_feature_vector(self.feature_names), dtype=np.float32)
            self.mean = dummy_vec
            self.std = np.ones_like(dummy_vec)
            self.is_trained = True

        vec = np.array(record.to_feature_vector(self.feature_names), dtype=np.float32)
        norm_vec = (vec - self.mean) / self.std

        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(norm_vec).unsqueeze(0)
            recon = self.model(x_tensor)
            mse_error = torch.mean((recon - x_tensor) ** 2).item()

        is_fault = bool(mse_error >= self.threshold)

        return {
            "is_fault": is_fault,
            "anomaly_score": round(float(mse_error), 4),
            "timestamp": record.timestamp,
            "edge_node_id": record.edge_node_id,
            "detector": "PyTorch_Autoencoder",
        }

    def save_checkpoint(self, path: str):
        # Save PyTorch model checkpoint to disk.
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "mean": getattr(self, "mean", None),
            "std": getattr(self, "std", None),
            "threshold": self.threshold,
            "is_trained": self.is_trained,
        }, path)
        logger.info(f"Saved PyTorch detector checkpoint to {path}")

    def load_checkpoint(self, path: str):
        # Load PyTorch model checkpoint from disk.
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.mean = checkpoint.get("mean", None)
            self.std = checkpoint.get("std", None)
            self.threshold = checkpoint.get("threshold", self.threshold)
            self.is_trained = checkpoint.get("is_trained", True)
            logger.info(f"Loaded PyTorch detector checkpoint from {path}")

