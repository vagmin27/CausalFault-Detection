# Common Benchmark Data Harness for Edge-IIoTset Telemetry Streaming.
#
# Ensures:
# - Identical observation stream delivered to all algorithms.
# - Complete isolation of ground-truth labels from algorithm observable inputs.
# - Deterministic ordering and chunked memory-efficient streaming.
# - Standardized 1,000-record warm-up delineation.
# - Benchmark manifest generation for full experimental reproducibility.

import os
import glob
import json
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Generator, List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from .context import BenchmarkInput, EventGroundTruth, BenchmarkContext

logger = logging.getLogger("DataHarness")


# -----------------------------------------------------------------------------
# Explicit Mapping: Cyber-Physical Attack Types to Benchmark Fault Categories
# -----------------------------------------------------------------------------
ATTACK_TO_FAULT_CATEGORY: Dict[str, Tuple[str, Optional[str]]] = {
    "NORMAL": ("NORMAL_OPERATION", None),
    "DDOS_ICMP": ("NETWORK_FLOOD_FAULT", "icmp.checksum"),
    "DDOS_UDP": ("NETWORK_FLOOD_FAULT", "udp.time_delta"),
    "DDOS_TCP": ("NETWORK_FLOOD_FAULT", "tcp_active_flags_count"),
    "DDOS_HTTP": ("NETWORK_FLOOD_FAULT", "http.content_length"),
    "SQL_INJECTION": ("APPLICATION_EXPLOIT_FAULT", "http.request.uri.query"),
    "XSS": ("APPLICATION_EXPLOIT_FAULT", "http.request.method"),
    "UPLOADING": ("APPLICATION_EXPLOIT_FAULT", "http.content_length"),
    "BACKDOOR": ("COMPROMISED_PROCESS_FAULT", "tcp.dstport"),
    "PASSWORD": ("RECONNAISSANCE_BURST_FAULT", "tcp_active_flags_count"),
    "VULNERABILITY_SCANNER": ("RECONNAISSANCE_BURST_FAULT", "is_well_known_dstport"),
    "PORT_SCANNING": ("RECONNAISSANCE_BURST_FAULT", "tcp.dstport"),
    "FINGERPRINTING": ("RECONNAISSANCE_BURST_FAULT", "tcp.flags"),
    "RANSOMWARE": ("RESOURCE_EXHAUSTION_FAULT", "payload_len"),
    "MITM": ("TRAFFIC_HIJACK_FAULT", "arp.opcode"),
}


@dataclass
class StreamConfig:
    # Canonical test stream specification ensuring 100% identical evaluation input.
    dataset_name: str = "edge_iiotset"
    split: str = "test"
    data_dir: str = "data/processed/test"
    artifacts_dir: str = "data/processed/artifacts"
    warmup_count: int = 1000
    max_records: Optional[int] = None
    batch_size: int = 5000
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "data_dir": self.data_dir,
            "artifacts_dir": self.artifacts_dir,
            "warmup_count": self.warmup_count,
            "max_records": self.max_records,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }


class CommonDataHarness:
    # Standardized streaming harness feeding all five approaches under identical conditions.
    # Guarantees strict separation of observable features from ground truth.

    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self.feature_names: List[str] = self._load_feature_names()
        self.resolved_files: List[str] = self._resolve_files()
        self.feature_hash: str = self._compute_feature_hash()

    def _load_feature_names(self) -> List[str]:
        path = os.path.join(self.config.artifacts_dir, "feature_names.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        # Fallback for synthetic/smoke testing
        return []

    def _compute_feature_hash(self) -> str:
        s = ",".join(self.feature_names).encode("utf-8")
        return hashlib.sha256(s).hexdigest()[:16]

    def _resolve_files(self) -> List[str]:
        path = self.config.data_dir
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data path does not exist: {path}")
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*.csv")))
            if not files:
                raise FileNotFoundError(f"No CSV files found in {path}")
            return files
        return [path]

    def generate_manifest(self, output_dir: str = "results/raw") -> Dict[str, Any]:
        # Generates results/raw/benchmark_manifest.json documenting stream parameters.
        os.makedirs(output_dir, exist_ok=True)
        manifest = {
            "dataset_name": self.config.dataset_name,
            "split": self.config.split,
            "feature_count": len(self.feature_names),
            "feature_hash": self.feature_hash,
            "feature_names": self.feature_names,
            "warmup_records": self.config.warmup_count,
            "max_records": self.config.max_records,
            "source_files": [os.path.basename(f) for f in self.resolved_files],
            "random_seed": self.config.seed,
            "manifest_hash": "",
        }
        manifest_str = json.dumps(manifest, sort_keys=True).encode("utf-8")
        manifest["manifest_hash"] = hashlib.sha256(manifest_str).hexdigest()[:16]

        out_path = os.path.join(output_dir, "benchmark_manifest.json")
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def stream_contexts(self) -> Generator[BenchmarkContext, None, None]:
        # Streams standardized BenchmarkContext objects.
        # Yields (observable_input, ground_truth) cleanly separated.
        position = 0
        max_recs = self.config.max_records
        warmup_limit = self.config.warmup_count

        for file_path in self.resolved_files:
            reader = pd.read_csv(file_path, chunksize=self.config.batch_size, low_memory=False)
            try:
                for chunk in reader:
                    # Dynamically set feature names if not loaded from artifacts (e.g. in test fixture)
                    if not self.feature_names:
                        non_feats = {"timestamp", "ip.src_host", "ip.dst_host", "Attack_label", "Attack_type"}
                        self.feature_names = [c for c in chunk.columns if c not in non_feats]
                        self.feature_hash = self._compute_feature_hash()

                    for _, row in chunk.iterrows():
                        # 1. Parse timestamps
                        ts_str = str(row.get("timestamp", ""))
                        try:
                            ts = pd.to_datetime(ts_str).timestamp()
                        except Exception:
                            ts = float(position)

                        dev_id = str(row.get("ip.src_host", "192.168.0.128"))
                        edge_id = str(row.get("ip.dst_host", "192.168.0.101"))

                        # 2. Extract strictly observable feature vector
                        feat_dict = {}
                        feat_vec = np.zeros(len(self.feature_names), dtype=np.float64)
                        for i, fname in enumerate(self.feature_names):
                            val = float(row.get(fname, 0.0))
                            feat_dict[fname] = val
                            feat_vec[i] = val

                        obs_input = BenchmarkInput(
                            stream_position=position,
                            timestamp=ts,
                            timestamp_str=ts_str,
                            device_id=dev_id,
                            edge_node_id=edge_id,
                            features=feat_dict,
                            feature_vector=feat_vec,
                        )

                        # 3. Extract ground-truth (ISOLATED from algorithm input)
                        lbl_val = int(row.get("Attack_label", 0))
                        atk_type = str(row.get("Attack_type", "Normal")).strip()
                        atk_key = atk_type.upper().replace(" ", "_").replace("-", "_")

                        category, root_cause = ATTACK_TO_FAULT_CATEGORY.get(
                            atk_key, ("GENERIC_ANOMALY", None)
                        )

                        gt = EventGroundTruth(
                            stream_position=position,
                            timestamp=ts,
                            device_id=dev_id,
                            edge_node_id=edge_id,
                            is_fault=(lbl_val == 1),
                            fault_label=lbl_val,
                            raw_attack_type=atk_type,
                            canonical_fault_category=category,
                            root_cause_metric=root_cause,
                        )

                        is_warmup = (position < warmup_limit)

                        ctx = BenchmarkContext(
                            stream_position=position,
                            observable_input=obs_input,
                            ground_truth=gt,
                            is_warmup=is_warmup,
                        )

                        yield ctx
                        position += 1

                        if max_recs is not None and position >= max_recs:
                            return
            finally:
                if hasattr(reader, "close"):
                    reader.close()
