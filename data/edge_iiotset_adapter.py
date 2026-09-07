# Edge-IIoTset Dataset Adapter.
#
# Adapter for streaming telemetry observations from Edge-IIoTset dataset files
# (both preprocessed and raw CSV chunks).

import os
import glob
import logging
from typing import Generator, Optional, List
import pandas as pd
from .telemetry import TelemetryRecord, DataSource

logger = logging.getLogger(__name__)


class EdgeIIoTsetAdapter(DataSource):
    """
    Adapter for streaming telemetry observations from Edge-IIoTset dataset files.
    Supports streaming from a single CSV or an entire directory of processed CSV files.
    """

    def __init__(self, data_path: Optional[str] = None, max_records: Optional[int] = None):
        self.data_path = data_path or "data/processed/test"
        self.max_records = max_records

    def get_dataset_name(self) -> str:
        return "Edge-IIoTset"

    def _resolve_files(self) -> List[str]:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"[Edge-IIoTset Adapter Error] Dataset path not found at: '{self.data_path}'.\n"
                f"Please run 'python scripts/preprocess_dataset.py' to generate processed splits "
                f"or specify a valid file/directory path via '--data-path'."
            )
        if os.path.isdir(self.data_path):
            csv_files = sorted(glob.glob(os.path.join(self.data_path, "*.csv")))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in dataset directory: '{self.data_path}'")
            return csv_files
        return [self.data_path]

    def stream_telemetry(self) -> Generator[TelemetryRecord, None, None]:
        files = self._resolve_files()
        logger.info(f"Loading Edge-IIoTset dataset from {len(files)} file(s) in: {self.data_path}")

        count = 0
        chunk_size = 5000

        for file_path in files:
            source_file = os.path.basename(file_path)
            try:
                for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                    for idx, row in chunk.iterrows():
                        record = self._map_row_to_record(idx, row, source_file)
                        yield record
                        count += 1
                        if self.max_records is not None and count >= self.max_records:
                            return
            except Exception as e:
                logger.error(f"Error streaming Edge-IIoTset file {file_path}: {e}")
                raise e

    def _map_row_to_record(self, idx: int, row: pd.Series, source_file: str) -> TelemetryRecord:
        # 1. Timestamp resolution
        ts_val = row.get("timestamp", row.get("frame.time", None))
        if pd.notnull(ts_val):
            try:
                ts = pd.to_datetime(ts_val).timestamp()
            except Exception:
                ts = float(idx)
        else:
            ts = float(idx)

        # 2. Host identities
        src_host = str(row.get("ip.src_host", "192.168.0.128"))
        dst_host = str(row.get("ip.dst_host", "192.168.0.101"))

        # 3. Target labels (isolated)
        label_val = row.get("Attack_label", 0)
        try:
            fault_label = int(label_val)
        except (ValueError, TypeError):
            fault_label = 0 if str(label_val).lower() in ["0", "normal", "false"] else 1

        orig_label = str(row.get("Attack_type", "Normal" if fault_label == 0 else "Attack"))

        # 4. Extract numerical features into raw_features dictionary
        # Explicit behavioral network telemetry - not falsely labeled as physical measurements
        non_feature_cols = {
            "timestamp", "frame.time", "ip.src_host", "ip.dst_host",
            "Attack_label", "Attack_type", "device_id", "edge_node_id"
        }
        raw_feats = {}
        for col, val in row.items():
            if col not in non_feature_cols and pd.notnull(val):
                try:
                    num_val = float(val)
                    raw_feats[col] = num_val
                except (ValueError, TypeError):
                    pass

        return TelemetryRecord(
            timestamp=ts,
            device_id=src_host,
            edge_node_id=dst_host,
            cpu_utilization=None,      # Not physically measured in network pcap
            memory_utilization=None,
            network_utilization=None,
            latency=None,             # Behavioral features kept in raw_features
            packet_loss=None,
            throughput=None,
            workload=1.0 if fault_label == 1 else 0.0,
            fault_label=fault_label,
            fault_type="CYBER_PHYSICAL_ATTACK" if fault_label == 1 else "NONE",
            original_label=orig_label,
            dataset_name="Edge-IIoTset",
            source_file=source_file,
            device_type=src_host,
            raw_features=raw_feats,
        )
