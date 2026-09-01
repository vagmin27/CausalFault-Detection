"""
TON_IoT Dataset Adapter.

TON_IoT is the primary real-world dataset for evaluating anomaly/attack detection
in Heterogeneous IoT and Edge Networks.

Note: TON_IoT is a cybersecurity telemetry dataset (attacks vs normal traffic).
While cyber attacks cause abnormal telemetry behavior (high latency, packet loss, CPU spikes),
they are distinct from physical hardware faults. The adapter preserves the original attack label
in `original_label` and maps binary anomaly status to `fault_label`.

Dataset Adapter Policy:
If the dataset file is not present at --data-path, a clear FileNotFoundError is raised
directing the user to provide the valid dataset file path.
"""

import os
import pandas as pd
from typing import Generator, Optional
import logging
from .telemetry import TelemetryRecord, DataSource

logger = logging.getLogger(__name__)


class TONIoTAdapter(DataSource):
    """
    Adapter for streaming telemetry observations from TON_IoT dataset files.
    """

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path

    def get_dataset_name(self) -> str:
        return "TON_IoT"

    def stream_telemetry(self) -> Generator[TelemetryRecord, None, None]:
        if not self.data_path or not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"[TON_IoT Adapter Error] Dataset file or directory not found at: '{self.data_path}'.\n"
                f"Please download the TON_IoT dataset (e.g. Train_Test_Network.csv or IoT_Telemetry.csv) "
                f"and specify its absolute path via '--data-path <path_to_csv>'."
            )

        logger.info(f"Loading TON_IoT dataset from: {self.data_path}")

        # Stream row by row using pandas chunking
        try:
            chunk_size = 1000
            for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
                for idx, row in chunk.iterrows():
                    record = self._map_row_to_record(idx, row)
                    yield record
        except Exception as e:
            logger.error(f"Error streaming TON_IoT dataset: {e}")
            raise e

    def _map_row_to_record(self, idx: int, row: pd.Series) -> TelemetryRecord:
        """
        Dynamically map TON_IoT CSV fields to standard TelemetryRecord schema.
        Handles common column names found across TON_IoT subsets (Network, IoT device telemetry).
        """
        # Timestamp
        ts = float(row.get("ts", row.get("timestamp", idx)))

        # Identifiers
        dev_id = str(row.get("src_ip", row.get("device_id", row.get("type", "TON_Device_1"))))
        edge_id = str(row.get("dst_ip", row.get("edge_node_id", "Edge_Node_1")))

        # Telemetry metrics mapping (where available)
        cpu = row.get("CPU_Usage", row.get("cpu_utilization", None))
        mem = row.get("Memory_Usage", row.get("memory_utilization", None))
        net = row.get("Network_Usage", row.get("network_utilization", None))
        lat = row.get("duration", row.get("latency", None))
        pkt_loss = row.get("missed_bytes", row.get("packet_loss", None))
        tp = row.get("src_bytes", row.get("throughput", None))
        workload = row.get("src_pkts", row.get("workload", None))

        # Labels (Attack / Anomaly binary label & detailed category)
        label_val = row.get("label", row.get("type", 0))
        try:
            fault_label = int(label_val)
        except (ValueError, TypeError):
            fault_label = 1 if str(label_val).lower() not in ["normal", "0", "false"] else 0

        orig_label = str(row.get("type", "ATTACK" if fault_label == 1 else "NORMAL"))

        return TelemetryRecord(
            timestamp=float(ts) if pd.notnull(ts) else float(idx),
            device_id=dev_id,
            edge_node_id=edge_id,
            cpu_utilization=float(cpu) if pd.notnull(cpu) else None,
            memory_utilization=float(mem) if pd.notnull(mem) else None,
            network_utilization=float(net) if pd.notnull(net) else None,
            latency=float(lat) if pd.notnull(lat) else None,
            packet_loss=float(pkt_loss) if pd.notnull(pkt_loss) else None,
            throughput=float(tp) if pd.notnull(tp) else None,
            workload=float(workload) if pd.notnull(workload) else None,
            fault_label=fault_label,
            fault_type="SECURITY_ATTACK" if fault_label == 1 else "NONE",
            original_label=orig_label,
        )
