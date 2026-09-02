# N-BaIoT Dataset Adapter.
#
# N-BaIoT is a benchmark generalization dataset containing IoT botnet attack telemetry
# captured across commercial IoT devices (cameras, doorbells, sensors).
#
# Dataset Adapter Policy:
# If the dataset file is not present at --data-path, a clear FileNotFoundError is raised
# directing the user to provide the valid dataset file path.

import os
import pandas as pd
from typing import Generator, Optional
import logging
from .telemetry import TelemetryRecord, DataSource

logger = logging.getLogger(__name__)


class NBaIoTAdapter(DataSource):
    # Adapter for streaming telemetry observations from N-BaIoT dataset CSV files.

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path

    def get_dataset_name(self) -> str:
        return "N-BaIoT"

    def stream_telemetry(self) -> Generator[TelemetryRecord, None, None]:
        if not self.data_path or not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"[N-BaIoT Adapter Error] Dataset file or directory not found at: '{self.data_path}'.\n"
                f"Please download the N-BaIoT dataset CSV (e.g. 1.benign.csv or mirai_attacks.csv) "
                f"and specify its path via '--data-path <path_to_csv>'."
            )

        logger.info(f"Loading N-BaIoT dataset from: {self.data_path}")

        try:
            chunk_size = 1000
            for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
                for idx, row in chunk.iterrows():
                    record = self._map_row_to_record(idx, row)
                    yield record
        except Exception as e:
            logger.error(f"Error streaming N-BaIoT dataset: {e}")
            raise e

    def _map_row_to_record(self, idx: int, row: pd.Series) -> TelemetryRecord:
        # Map N-BaIoT extracted statistical features (MI, HH, HpH stream features)
        # to TelemetryRecord metrics.
        # N-BaIoT features include packet arrival rates, jitter, sizes across time windows
        # e.g., MI_dir_L5_weight, MI_dir_L5_mean, HH_L5_magnitude, etc.
        ts = float(idx)
        dev_id = "NBaIoT_Device_1"
        edge_id = "NBaIoT_Edge_1"

        # Map stream packet weight / rate as workload / throughput proxies
        workload = row.get("MI_dir_L5_weight", row.get("workload", None))
        tp = row.get("MI_dir_L5_mean", row.get("throughput", None))
        lat = row.get("HH_L5_std", row.get("latency", None))
        pkt_loss = row.get("H_L5_variance", row.get("packet_loss", None))

        # Check label column if present or infer from filename
        is_benign = "benign" in str(self.data_path).lower()
        label_val = row.get("label", 0 if is_benign else 1)
        try:
            fault_label = int(label_val)
        except (ValueError, TypeError):
            fault_label = 0 if is_benign else 1

        orig_label = "BENIGN" if fault_label == 0 else "BOTNET_ATTACK"

        return TelemetryRecord(
            timestamp=float(ts),
            device_id=dev_id,
            edge_node_id=edge_id,
            cpu_utilization=None,
            memory_utilization=None,
            network_utilization=None,
            latency=float(lat) if (pd.notnull(lat) and isinstance(lat, (int, float))) else None,
            packet_loss=float(pkt_loss) if (pd.notnull(pkt_loss) and isinstance(pkt_loss, (int, float))) else None,
            throughput=float(tp) if (pd.notnull(tp) and isinstance(tp, (int, float))) else None,
            workload=float(workload) if (pd.notnull(workload) and isinstance(workload, (int, float))) else None,
            fault_label=fault_label,
            fault_type="BOTNET_ANOMALY" if fault_label == 1 else "NONE",
            original_label=orig_label,
        )

