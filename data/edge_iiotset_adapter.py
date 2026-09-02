# Edge-IIoTset Dataset Adapter.
#
# Edge-IIoTset is a primary Edge-IIoT validation dataset containing cyber-physical
# system telemetry and attack vectors across multi-layer Edge-IoT architectures.
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


class EdgeIIoTsetAdapter(DataSource):
    # Adapter for streaming telemetry observations from Edge-IIoTset dataset files.

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path

    def get_dataset_name(self) -> str:
        return "Edge-IIoTset"

    def stream_telemetry(self) -> Generator[TelemetryRecord, None, None]:
        if not self.data_path or not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"[Edge-IIoTset Adapter Error] Dataset file not found at: '{self.data_path}'.\n"
                f"Please download the Edge-IIoTset CSV (e.g. Edge-IIoTset_dataset.csv) "
                f"and specify its path via '--data-path <path_to_csv>'."
            )

        logger.info(f"Loading Edge-IIoTset dataset from: {self.data_path}")

        try:
            chunk_size = 1000
            for chunk in pd.read_csv(self.data_path, chunksize=chunk_size, low_memory=False):
                for idx, row in chunk.iterrows():
                    record = self._map_row_to_record(idx, row)
                    yield record
        except Exception as e:
            logger.error(f"Error streaming Edge-IIoTset dataset: {e}")
            raise e

    def _map_row_to_record(self, idx: int, row: pd.Series) -> TelemetryRecord:
        # Map Edge-IIoTset row fields to unified TelemetryRecord format.
        ts = float(row.get("frame.time_epoch", idx))
        dev_id = str(row.get("ip.src_host", row.get("src_ip", "IIoT_Device_1")))
        edge_id = str(row.get("ip.dst_host", row.get("dst_ip", "IIoT_Edge_1")))

        lat = row.get("tcp.time_delta", row.get("latency", None))
        pkt_loss = row.get("tcp.analysis.lost_segment", row.get("packet_loss", None))
        tp = row.get("tcp.len", row.get("throughput", None))
        workload = row.get("http.request.method", row.get("workload", None))
        w_val = 1.0 if pd.notnull(workload) else 0.0

        label_val = row.get("Attack_label", row.get("label", 0))
        try:
            fault_label = int(label_val)
        except (ValueError, TypeError):
            fault_label = 1 if str(label_val).lower() not in ["normal", "0"] else 0

        orig_label = str(row.get("Attack_type", "ATTACK" if fault_label == 1 else "NORMAL"))

        return TelemetryRecord(
            timestamp=float(ts) if pd.notnull(ts) else float(idx),
            device_id=dev_id,
            edge_node_id=edge_id,
            cpu_utilization=None,  # Not directly measured in network pcap derived IIoTset
            memory_utilization=None,
            network_utilization=None,
            latency=float(lat) * 1000.0 if (pd.notnull(lat) and isinstance(lat, (int, float))) else None,
            packet_loss=float(pkt_loss) if (pd.notnull(pkt_loss) and isinstance(pkt_loss, (int, float))) else None,
            throughput=float(tp) if (pd.notnull(tp) and isinstance(tp, (int, float))) else None,
            workload=w_val,
            fault_label=fault_label,
            fault_type="CYBER_PHYSICAL_ATTACK" if fault_label == 1 else "NONE",
            original_label=orig_label,
        )

