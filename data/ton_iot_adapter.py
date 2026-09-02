
# TON_IoT Dataset Streaming Adapter.

# Provides real-time streaming ingestion for the UNSW Canberra TON_IoT dataset subsets.
# Discovers and streams observations sequentially from IoT device CSV files:
# - IoT_Fridge.csv
# - IoT_GPS_Tracker.csv
# - IoT_Garage_Door.csv
# - IoT_Modbus.csv
# - IoT_Motion_Light.csv
# - IoT_Weather.csv

# Dataset Adapter Policy:
#     1. High-Performance Vectorized Ingestion: Uses chunked vectorized string cleaning and dict streaming.
#     2. No Artificial Telemetry: Physical system metrics unavailable in the dataset
#     (cpu_utilization, memory_utilization, packet_loss) remain strictly None.
#     3. Raw Feature Retention: Device-specific sensor telemetry (temperatures, pressure, registers, GPS, states)
#     is preserved inside TelemetryRecord.raw_features and extracted for downstream models.
#     4. Ground-Truth Isolation: Original 'label' (0/1) and 'type' (attack category) are assigned to
#     fault_label and original_label for evaluation ONLY.


import os
import glob
import logging
from typing import Generator, Optional, List, Dict, Any, Tuple
import pandas as pd
from datetime import datetime

from .telemetry import TelemetryRecord, DataSource

logger = logging.getLogger(__name__)


def _safe_str(val: Any) -> str:
    if pd.isnull(val):
        return ""
    return str(val).strip().lower()


class TONIoTAdapter(DataSource):
    
    # High-Performance Streaming Adapter for TON_IoT device telemetry datasets.
    

    def __init__(
        self,
        data_path: Optional[str] = None,
        chunk_size: int = 50000,
        max_records: Optional[int] = None,
    ):
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.max_records = max_records
        self.processed_count = 0
        self.ts_cache: Dict[str, float] = {}

    def get_dataset_name(self) -> str:
        return "TON_IoT"

    def _discover_files(self) -> List[str]:
        # Discover CSV files in specified data path.
        if not self.data_path or not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"[TON_IoT Adapter Error] Dataset path not found: '{self.data_path}'.\n"
                f"Please place TON_IoT CSV files in 'data/datasets/Processed_IoT_dataset/'."
            )

        if os.path.isfile(self.data_path):
            return [self.data_path]
        elif os.path.isdir(self.data_path):
            csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
            csv_files.sort()
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in dataset directory: '{self.data_path}'")
            return csv_files
        else:
            raise FileNotFoundError(f"Invalid dataset path: '{self.data_path}'")

    def stream_telemetry(self) -> Generator[TelemetryRecord, None, None]:
        
        # Stream TelemetryRecords sequentially across discovered TON_IoT device CSV files.
        
        files = self._discover_files()
        logger.info(f"TON_IoT Adapter discovered {len(files)} dataset files: {[os.path.basename(f) for f in files]}")
        self.processed_count = 0

        for fpath in files:
            fname = os.path.basename(fpath)
            device_type = fname.replace(".csv", "")
            logger.info(f"Streaming dataset file: {fname} (Device: {device_type})")

            last_state: Dict[str, Any] = {}

            try:
                for chunk in pd.read_csv(fpath, chunksize=self.chunk_size, low_memory=False):
                    rows_dict = chunk.to_dict(orient="records")
                    for row in rows_dict:
                        record, last_state = self._map_row_to_record(row, fname, device_type, self.processed_count, last_state)
                        self.processed_count += 1
                        yield record

                        if self.max_records is not None and self.processed_count >= self.max_records:
                            logger.info(f"Reached max_records limit ({self.max_records}). Stopping stream.")
                            return
            except Exception as e:
                logger.error(f"Error streaming file '{fname}': {e}")
                raise e

        logger.info(f"Completed streaming {self.processed_count} total records from TON_IoT dataset.")

    def _map_row_to_record(
        self,
        row: Dict[str, Any],
        source_file: str,
        device_type: str,
        global_idx: int,
        last_state: Dict[str, Any],
    ) -> Tuple[TelemetryRecord, Dict[str, Any]]:
        
        # Map a single pre-sanitized CSV row dict to TelemetryRecord without fabricating missing physical telemetry.
        
        # 1. Fast Cached Timestamp Parsing
        date_str = _safe_str(row.get("date"))
        time_str = _safe_str(row.get("time"))
        if not date_str or date_str == "nan":
            date_str = last_state.get("date", "")
        if not time_str or time_str == "nan":
            time_str = last_state.get("time", "")

        timestamp_val = float(global_idx)
        if date_str and time_str:
            dt_key = f"{date_str} {time_str}"
            if dt_key in self.ts_cache:
                timestamp_val = self.ts_cache[dt_key]
            else:
                try:
                    dt_obj = datetime.strptime(dt_key, "%d-%b-%y %H:%M:%S")
                    timestamp_val = dt_obj.timestamp()
                    self.ts_cache[dt_key] = timestamp_val
                    last_state["date"] = date_str
                    last_state["time"] = time_str
                except Exception:
                    timestamp_val = float(global_idx)

        # 2. Extract Identifiers
        dev_id = device_type
        edge_node_id = f"Edge_{device_type}"

        # 3. Extract Raw Device-Specific Features
        raw_features: Dict[str, Any] = {}

        if "IoT_Fridge" in device_type:
            val = row.get("fridge_temperature")
            if pd.notnull(val):
                try:
                    raw_features["fridge_temperature"] = float(val)
                except (ValueError, TypeError):
                    pass
            tc = _safe_str(row.get("temp_condition"))
            if tc:
                raw_features["temp_condition_high"] = 1.0 if "high" in tc else 0.0

        elif "IoT_GPS_Tracker" in device_type:
            lat = row.get("latitude")
            lon = row.get("longitude")
            if pd.notnull(lat):
                try:
                    raw_features["latitude"] = float(lat)
                except (ValueError, TypeError):
                    pass
            if pd.notnull(lon):
                try:
                    raw_features["longitude"] = float(lon)
                except (ValueError, TypeError):
                    pass

        elif "IoT_Garage_Door" in device_type:
            ds = _safe_str(row.get("door_state"))
            ss = _safe_str(row.get("sphone_signal"))
            if not ds or ds == "nan":
                ds = last_state.get("door_state", "closed")
            if not ss or ss == "nan":
                ss = last_state.get("sphone_signal", "false")
            last_state["door_state"] = ds
            last_state["sphone_signal"] = ss
            raw_features["door_state_open"] = 1.0 if "open" in ds else 0.0
            raw_features["sphone_signal_true"] = 1.0 if "true" in ss else 0.0

        elif "IoT_Modbus" in device_type:
            for fc in ["FC1_Read_Input_Register", "FC2_Read_Discrete_Value", "FC3_Read_Holding_Register", "FC4_Read_Coil"]:
                val = row.get(fc)
                if pd.notnull(val):
                    try:
                        raw_features[fc] = float(val)
                    except (ValueError, TypeError):
                        pass

        elif "IoT_Motion_Light" in device_type:
            ms = row.get("motion_status")
            if pd.notnull(ms):
                try:
                    raw_features["motion_status"] = float(ms)
                except (ValueError, TypeError):
                    pass
            ls = _safe_str(row.get("light_status"))
            if ls:
                raw_features["light_status_on"] = 1.0 if "on" in ls else 0.0

        elif "IoT_Weather" in device_type:
            t_val = row.get("temperature")
            p_val = row.get("pressure")
            h_val = row.get("humidity")
            if pd.notnull(t_val):
                try:
                    raw_features["temperature"] = float(t_val)
                except (ValueError, TypeError):
                    pass
            if pd.notnull(p_val):
                try:
                    raw_features["pressure"] = float(p_val)
                except (ValueError, TypeError):
                    pass
            if pd.notnull(h_val):
                try:
                    raw_features["humidity"] = float(h_val)
                except (ValueError, TypeError):
                    pass

        # 4. Extract Ground-Truth Labels (Evaluation Only)
        raw_lbl = row.get("label", 0)
        try:
            fault_label = int(raw_lbl) if pd.notnull(raw_lbl) else 0
        except (ValueError, TypeError):
            fault_label = 0

        attack_type = _safe_str(row.get("type"))
        if not attack_type or attack_type == "nan":
            attack_type = "normal"
        if fault_label == 1 and attack_type == "normal":
            attack_type = "attack"

        record = TelemetryRecord(
            timestamp=timestamp_val,
            device_id=dev_id,
            edge_node_id=edge_node_id,
            cpu_utilization=None,      # MISSING in TON_IoT (No fabrication)
            memory_utilization=None,   # MISSING in TON_IoT (No fabrication)
            network_utilization=None,  # MISSING in TON_IoT (No fabrication)
            latency=None,              # MISSING in TON_IoT (No fabrication)
            packet_loss=None,          # MISSING in TON_IoT (No fabrication)
            throughput=None,           # MISSING in TON_IoT (No fabrication)
            workload=None,             # MISSING in TON_IoT (No fabrication)
            fault_label=fault_label,
            fault_type=attack_type.upper(),
            original_label=attack_type,
            dataset_name="TON_IoT",
            source_file=source_file,
            device_type=device_type,
            raw_features=raw_features,
        )

        return record, last_state
