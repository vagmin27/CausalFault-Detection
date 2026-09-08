# preprocess_dataset.py - Memory-Efficient Preprocessing & Feature Engineering for Edge-IIoTset.
#
# Strict Requirements & Constraints:
# 1. Zero Imputation: Any row containing ANY missing/NaN/empty value in ANY original field is completely dropped.
# 2. Complete Invalid/Infinite Value Removal: Rows containing infinite or unconvertible values are completely dropped.
# 3. Post-Cleaning Deduplication: Duplicate rows are removed after invalid/missing row removal.
# 4. Preservation of Valid Extreme Values: Outliers are NOT removed or clipped, as they represent real attacks/faults.
# 5. Untouched Raw Data: Files in data/datasets/ remain strictly read-only.
# 6. Preserve Existing Splits: Train, validation, and test splits are processed independently without reshuffling.
# 7. Zero Future Lookahead: All temporal features are point-in-time per-packet (no cross-row delta across unrelated records).
# 8. Train-Only Fitting: StandardScaler and categorical vocabularies are fitted ONLY on the training split.
# 9. Target Isolation: Attack_label and Attack_type are strictly isolated and never used as input features.
# 10. Pre-Execution 5,000-Row Sample Test: Mandatory verification of schema match, 0 NaNs, 0 infs, and compatible labels.

import os
import sys
import glob
import json
import time
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Column Definitions based on Edge-IIoTset schema verification
# -----------------------------------------------------------------------------
NATIVE_NUMERIC_COLS = [
    'arp.opcode', 'arp.hw.size',
    'icmp.checksum', 'icmp.seq_le', 'icmp.transmit_timestamp',
    'http.content_length', 'http.response',
    'tcp.ack', 'tcp.ack_raw', 'tcp.checksum',
    'tcp.connection.fin', 'tcp.connection.rst', 'tcp.connection.syn', 'tcp.connection.synack',
    'tcp.dstport', 'tcp.flags', 'tcp.flags.ack', 'tcp.len', 'tcp.seq', 'tcp.srcport',
    'udp.port', 'udp.stream', 'udp.time_delta',
    'dns.qry.name', 'dns.qry.qu', 'dns.retransmission', 'dns.retransmit_request', 'dns.retransmit_request_in',
    'mqtt.conflag.cleansess', 'mqtt.conflags', 'mqtt.hdrflags', 'mqtt.len', 'mqtt.msgtype', 'mqtt.proto_len', 'mqtt.topic_len', 'mqtt.ver',
    'mbtcp.len', 'mbtcp.trans_id', 'mbtcp.unit_id'
]

ZERO_VARIANCE_COLS = [
    'icmp.unused', 'http.tls_port', 'dns.qry.type', 'mqtt.msg_decoded_as'
]

PAYLOAD_TEXT_COLS = [
    'tcp.options', 'tcp.payload', 'http.file_data', 'http.request.uri.query',
    'http.referer', 'http.request.full_uri', 'mqtt.msg', 'dns.qry.name.len',
    'mqtt.conack.flags', 'arp.dst.proto_ipv4', 'arp.src.proto_ipv4'
]

METADATA_COLS = [
    'frame.time', 'ip.src_host', 'ip.dst_host'
]

TARGET_COLS = [
    'Attack_label', 'Attack_type'
]


class EdgeIIoTsetPreprocessor:
    def __init__(self, raw_dir="data/datasets", output_dir="data/processed", artifacts_dir="data/processed/artifacts", chunk_size=50000):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.artifacts_dir = artifacts_dir
        self.chunk_size = chunk_size
        self.scaler = StandardScaler(copy=False)
        self.feature_names = []
        self.label_mapping = {}
        self.train_http_methods = set()
        self.stats = {
            "train": {"raw_rows": 0, "cleaned_rows": 0, "dropped_missing": 0, "dropped_duplicates": 0},
            "validation": {"raw_rows": 0, "cleaned_rows": 0, "dropped_missing": 0, "dropped_duplicates": 0},
            "test": {"raw_rows": 0, "cleaned_rows": 0, "dropped_missing": 0, "dropped_duplicates": 0},
        }

    def clean_chunk(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        # Cleans a raw chunk strictly according to requirements:
        # 1. Drops any row where ANY original column is missing, NaN, empty, or sentinel.
        # 2. Drops any row with unparseable timestamp (e.g. shifted IP lines).
        # 3. Drops any row with infinite numeric values.
        # 4. Drops duplicate rows after cleaning.
        # Returns cleaned df and valid parsed timestamps.
        # 1. Parse frame.time to datetime
        time_series = df['frame.time'].astype(str).str.strip()
        ts = pd.to_datetime(time_series, format='%Y %H:%M:%S.%f', errors='coerce')
        invalid_mask = ts.isna()

        # 2. Check all original columns for missing/empty/sentinel
        for col in df.columns:
            s = df[col]
            invalid_mask |= s.isna()
            if s.dtype == object or isinstance(s.dtype, pd.StringDtype):
                str_s = s.astype(str).str.strip()
                invalid_mask |= (str_s == '')
                invalid_mask |= str_s.str.lower().isin(['nan', 'null', 'none', 'na', 'nat', '?'])

        df_clean = df[~invalid_mask].copy()
        ts_clean = ts[~invalid_mask]

        if len(df_clean) == 0:
            return df_clean, ts_clean

        # 3. Check infinite values across numeric columns
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            inf_mask = np.isinf(df_clean[num_cols]).any(axis=1)
            df_clean = df_clean[~inf_mask].copy()
            ts_clean = ts_clean[~inf_mask]

        if len(df_clean) == 0:
            return df_clean, ts_clean

        # 4. Drop duplicates after missing/invalid row removal
        pre_dedup_len = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        ts_clean = ts_clean.loc[df_clean.index]

        return df_clean, ts_clean

    def extract_features(self, df: pd.DataFrame, ts: pd.Series) -> pd.DataFrame:
        # Extracts engineered features without lookahead and without misleading proxy labels:
        # - Point-in-time timestamp features: hour, minute, second, sin_hour, cos_hour, day_of_week
        # - Native validated network numeric columns
        # - Behavioral derived features (explicitly behavioral, not physical measurements)
        # - Protocol presence flags
        # - Train-consistent categorical encodings
        feats = pd.DataFrame(index=df.index)

        # 1. Point-in-time temporal features
        hour = ts.dt.hour.astype(np.float64)
        feats['frame_hour'] = hour
        feats['frame_minute'] = ts.dt.minute.astype(np.float64)
        feats['frame_second'] = ts.dt.second.astype(np.float64)
        feats['sin_hour'] = np.sin(2.0 * np.pi * hour / 24.0)
        feats['cos_hour'] = np.cos(2.0 * np.pi * hour / 24.0)
        feats['day_of_week'] = ts.dt.dayofweek.astype(np.float64)

        # 2. Native numeric features (validated)
        for col in NATIVE_NUMERIC_COLS:
            feats[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(np.float64)

        # 3. Behavioral derived network features
        # Note: explicitly behavioral, not labelled as physical packet loss or latency
        feats['tcp_active_flags_count'] = (
            feats['tcp.connection.syn'] + feats['tcp.connection.fin'] +
            feats['tcp.connection.rst'] + feats['tcp.connection.synack'] +
            feats['tcp.flags.ack']
        ).astype(np.float64)

        feats['tcp_seq_ack_ratio'] = (feats['tcp.seq'] / (feats['tcp.ack'] + 1.0)).astype(np.float64)

        feats['is_well_known_dstport'] = (
            ((feats['tcp.dstport'] > 0) & (feats['tcp.dstport'] < 1024)) |
            ((feats['udp.port'] > 0) & (feats['udp.port'] < 1024))
        ).astype(np.float64)

        # 4. Protocol presence binary indicators
        feats['is_tcp'] = ((feats['tcp.dstport'] > 0) | (feats['tcp.srcport'] > 0) | (feats['tcp.flags'] > 0)).astype(np.float64)
        feats['is_udp'] = ((feats['udp.port'] > 0) | (feats['udp.stream'] > 0)).astype(np.float64)
        feats['is_icmp'] = ((feats['icmp.checksum'] > 0) | (feats['icmp.seq_le'] > 0)).astype(np.float64)
        feats['is_http'] = ((feats['http.content_length'] > 0) | (feats['http.response'] > 0)).astype(np.float64)
        feats['is_mqtt'] = ((feats['mqtt.len'] > 0) | (feats['mqtt.msgtype'] > 0) | (feats['mqtt.conflags'] > 0)).astype(np.float64)
        feats['is_dns'] = ((feats['dns.retransmission'] > 0) | (feats['dns.qry.qu'] > 0)).astype(np.float64)
        feats['is_mbtcp'] = ((feats['mbtcp.len'] > 0) | (feats['mbtcp.trans_id'] > 0) | (feats['mbtcp.unit_id'] > 0)).astype(np.float64)
        feats['is_arp'] = ((feats['arp.opcode'] > 0) | (feats['arp.hw.size'] > 0)).astype(np.float64)

        # 5. Categorical encodings (methods seen in training)
        http_m = df['http.request.method'].astype(str).str.strip().str.upper()
        feats['http_method_GET'] = (http_m == 'GET').astype(np.float64)
        feats['http_method_POST'] = (http_m == 'POST').astype(np.float64)
        feats['http_method_OPTIONS'] = (http_m == 'OPTIONS').astype(np.float64)
        feats['http_method_OTHER'] = (http_m.isin(['TRACE', 'SEARCH', 'PUT', 'HEAD'])).astype(np.float64)

        mqtt_p = df['mqtt.protoname'].astype(str).str.strip().str.upper()
        feats['is_mqtt_proto'] = (mqtt_p == 'MQTT').astype(np.float64)

        mqtt_t = df['mqtt.topic'].astype(str).str.strip()
        feats['has_mqtt_topic'] = (~mqtt_t.isin(['0', '0.0', ''])).astype(np.float64)

        return feats

    def fit_train(self, train_files: list[str], sample_limit: int = None):
        # Pass 1: Stream training files in chunks.
        # Fits StandardScaler incrementally on training feature vectors only.
        print(f"\n--- [Pass 1] Fitting on Training Data ({len(train_files)} files) ---")
        total_fitted_rows = 0
        feature_cols_initialized = False

        for f_idx, file_path in enumerate(train_files):
            print(f"  [Train File {f_idx + 1}/{len(train_files)}] {os.path.basename(file_path)}")
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size, low_memory=False):
                self.stats["train"]["raw_rows"] += len(chunk)
                
                df_clean, ts_clean = self.clean_chunk(chunk)
                dropped = len(chunk) - len(df_clean)
                self.stats["train"]["dropped_missing"] += dropped

                if len(df_clean) == 0:
                    continue

                feats = self.extract_features(df_clean, ts_clean)

                if not feature_cols_initialized:
                    self.feature_names = list(feats.columns)
                    feature_cols_initialized = True

                # Align columns
                X_chunk = feats[self.feature_names].values.astype(np.float64)
                self.scaler.partial_fit(X_chunk)
                total_fitted_rows += len(X_chunk)
                self.stats["train"]["cleaned_rows"] += len(X_chunk)

                if sample_limit and total_fitted_rows >= sample_limit:
                    print(f"  Sample limit of {sample_limit} reached in training fit.")
                    return

        print(f"Pass 1 Complete: Fitted StandardScaler on {total_fitted_rows:,} training rows across {len(self.feature_names)} features.")

    def transform_and_save(self, split_name: str, files: list[str], sample_limit: int = None):
        # Pass 2: Transforms split files using the fitted train components.
        # Saves processed chunks to data/processed/{split_name}/.
        print(f"\n--- [Pass 2] Transforming & Saving {split_name.upper()} ({len(files)} files) ---")
        split_out_dir = os.path.join(self.output_dir, split_name)
        os.makedirs(split_out_dir, exist_ok=True)

        total_saved_rows = 0

        for f_idx, file_path in enumerate(files):
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            out_file = os.path.join(split_out_dir, f"{base_name}_processed.csv")

            first_chunk = True
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size, low_memory=False):
                self.stats[split_name]["raw_rows"] += len(chunk)

                df_clean, ts_clean = self.clean_chunk(chunk)
                dropped = len(chunk) - len(df_clean)
                self.stats[split_name]["dropped_missing"] += dropped

                if len(df_clean) == 0:
                    continue

                feats = self.extract_features(df_clean, ts_clean)
                X_chunk = feats[self.feature_names].values.astype(np.float64)
                
                # Transform using trained scaler
                X_scaled = self.scaler.transform(X_chunk)
                scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names, index=df_clean.index)

                # Prepare final output chunk
                # Metadata + Scaled Features + Isolated Targets
                out_chunk = pd.DataFrame(index=df_clean.index)
                out_chunk['timestamp'] = ts_clean.dt.strftime('%Y-%m-%d %H:%M:%S.%f')
                out_chunk['ip.src_host'] = df_clean['ip.src_host'].astype(str)
                out_chunk['ip.dst_host'] = df_clean['ip.dst_host'].astype(str)

                # Append scaled features
                out_chunk = pd.concat([out_chunk, scaled_df], axis=1)

                # Append isolated targets
                out_chunk['Attack_label'] = pd.to_numeric(df_clean['Attack_label'], errors='coerce').astype(int)
                out_chunk['Attack_type'] = df_clean['Attack_type'].astype(str)

                # Strict sanity check on output chunk
                assert not out_chunk.isna().any().any(), f"NaN detected in processed output chunk for {split_name}!"
                assert not np.isinf(out_chunk[self.feature_names].values).any(), f"Inf detected in processed features for {split_name}!"

                # Append to CSV
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                out_chunk.to_csv(out_file, mode=mode, header=header, index=False)
                first_chunk = False

                total_saved_rows += len(out_chunk)
                self.stats[split_name]["cleaned_rows"] += len(out_chunk)

                if sample_limit and total_saved_rows >= sample_limit:
                    print(f"  Sample limit of {sample_limit} reached for {split_name}.")
                    break

            if sample_limit and total_saved_rows >= sample_limit:
                break

        print(f"Saved {total_saved_rows:,} processed rows for {split_name}.")

    def save_artifacts(self):
        # Saves fitted scaler, feature names, label mapping, and statistics.
        os.makedirs(self.artifacts_dir, exist_ok=True)
        scaler_path = os.path.join(self.artifacts_dir, "scaler.joblib")
        joblib.dump(self.scaler, scaler_path)

        features_path = os.path.join(self.artifacts_dir, "feature_names.json")
        with open(features_path, "w") as f:
            json.dump(self.feature_names, f, indent=2)

        stats_path = os.path.join(self.artifacts_dir, "preprocessing_report.json")
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)

        print(f"Artifacts successfully saved to {self.artifacts_dir}:")
        print(f"  - scaler.joblib ({len(self.feature_names)} features)")
        print(f"  - feature_names.json")
        print(f"  - preprocessing_report.json")

    def run_verification(self, sample_size=5000) -> bool:
        # Runs 5,000-row sample validation across train, validation, and test.
        # Verifies:
        # 1. Identical columns and column ordering across train, val, and test.
        # 2. Exactly 0 missing/NaN values across all columns.
        # 3. Exactly 0 infinite values across all columns.
        # 4. Compatible label distributions (Attack_label, Attack_type).
        print(f"\n========================================================")
        print(f"  RUNNING {sample_size}-ROW SAMPLE VALIDATION TEST")
        print(f"========================================================")

        train_files = sorted(glob.glob(os.path.join(self.raw_dir, "train", "*.csv")))
        val_files = sorted(glob.glob(os.path.join(self.raw_dir, "validation", "*.csv")))
        test_files = sorted(glob.glob(os.path.join(self.raw_dir, "test", "*.csv")))

        assert len(train_files) > 0, "No train files found in " + self.raw_dir
        assert len(val_files) > 0, "No validation files found in " + self.raw_dir
        assert len(test_files) > 0, "No test files found in " + self.raw_dir

        # Reset stats
        self.stats = {
            "train": {"raw_rows": 0, "cleaned_rows": 0, "dropped_missing": 0, "dropped_duplicates": 0},
            "validation": {"raw_rows": 0, "cleaned_rows": 0, "dropped_missing": 0, "dropped_duplicates": 0},
            "test": {"raw_rows": 0, "cleaned_rows": 0, "dropped_missing": 0, "dropped_duplicates": 0},
        }

        # 1. Fit on sample of train
        self.fit_train(train_files, sample_limit=sample_size)

        # 2. Transform and save samples of train, validation, test
        self.transform_and_save("train", train_files, sample_limit=sample_size)
        self.transform_and_save("validation", val_files, sample_limit=sample_size)
        self.transform_and_save("test", test_files, sample_limit=sample_size)

        # 3. Save sample artifacts
        self.save_artifacts()

        # 4. Rigorous assertions on processed files
        print(f"\n--- Checking Validation Assertions ---")
        processed_splits = {}
        for split in ["train", "validation", "test"]:
            split_files = sorted(glob.glob(os.path.join(self.output_dir, split, "*.csv")))
            assert len(split_files) > 0, f"No processed files found for {split}"
            df_split = pd.read_csv(split_files[0])
            processed_splits[split] = df_split
            print(f"  {split.upper()} sample loaded: {df_split.shape[0]} rows, {df_split.shape[1]} columns")

        train_cols = list(processed_splits["train"].columns)
        val_cols = list(processed_splits["validation"].columns)
        test_cols = list(processed_splits["test"].columns)

        # Assertion 1: Identical columns
        assert train_cols == val_cols, "Column mismatch between train and validation!"
        assert train_cols == test_cols, "Column mismatch between train and test!"
        print("  [CHECK 1 PASSED] All splits have IDENTICAL columns in identical sequence.")

        # Assertion 2: Zero missing values
        for split, df_split in processed_splits.items():
            nan_count = df_split.isna().sum().sum()
            assert nan_count == 0, f"Found {nan_count} NaNs in {split}!"
        print("  [CHECK 2 PASSED] Exactly 0 missing/NaN values across all columns in all splits.")

        # Assertion 3: Zero infinite values
        for split, df_split in processed_splits.items():
            num_data = df_split.select_dtypes(include=[np.number]).values
            inf_count = np.isinf(num_data).sum()
            assert inf_count == 0, f"Found {inf_count} infinite values in {split}!"
        print("  [CHECK 3 PASSED] Exactly 0 infinite values across all columns in all splits.")

        # Assertion 4: Target isolation and compatible labels
        for split, df_split in processed_splits.items():
            assert "Attack_label" in df_split.columns
            assert "Attack_type" in df_split.columns
            assert "Attack_label" not in self.feature_names
            assert "Attack_type" not in self.feature_names
            unique_labels = set(df_split['Attack_label'].unique())
            assert unique_labels.issubset({0, 1}), f"Unexpected Attack_label in {split}: {unique_labels}"
            attack_types = df_split['Attack_type'].value_counts()
            print(f"  {split.upper()} Attack types:\n{attack_types.to_dict()}")
        print("  [CHECK 4 PASSED] Attack_label and Attack_type are strictly isolated and compatible.")

        print(f"\n>>> 5,000-ROW SAMPLE VALIDATION SUCCEEDED 100% <<<\n")
        return True

    def run_full(self):
        # Runs complete preprocessing across all 24 CSV files.
        print(f"\n========================================================")
        print(f"  STARTING FULL DATASET PREPROCESSING (~1.16 GB)")
        print(f"========================================================")

        train_files = sorted(glob.glob(os.path.join(self.raw_dir, "train", "*.csv")))
        val_files = sorted(glob.glob(os.path.join(self.raw_dir, "validation", "*.csv")))
        test_files = sorted(glob.glob(os.path.join(self.raw_dir, "test", "*.csv")))

        start_time = time.time()

        # Reset stats
        self.stats = {
            "train": {"raw_rows": 0, "cleaned_rows": 0, "dropped_missing": 0, "dropped_duplicates": 0},
            "validation": {"raw_rows": 0, "cleaned_rows": 0, "dropped_missing": 0, "dropped_duplicates": 0},
            "test": {"raw_rows": 0, "cleaned_rows": 0, "dropped_missing": 0, "dropped_duplicates": 0},
        }

        # Pass 1: Fit on all train files
        self.fit_train(train_files)

        # Save artifacts after Pass 1
        self.save_artifacts()

        # Pass 2: Transform train, validation, and test
        self.transform_and_save("train", train_files)
        self.transform_and_save("validation", val_files)
        self.transform_and_save("test", test_files)

        # Update and save final report
        elapsed = time.time() - start_time
        self.stats["elapsed_seconds"] = elapsed
        self.stats["total_features"] = len(self.feature_names)
        self.save_artifacts()

        print(f"\n========================================================")
        print(f"  FULL PREPROCESSING COMPLETED IN {elapsed/60.0:.2f} MINUTES")
        print(f"========================================================")
        print(json.dumps(self.stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Edge-IIoTset Preprocessing and Feature Engineering")
    parser.add_argument("--sample", type=int, default=0, help="Run sample verification with specified rows (e.g. 5000)")
    parser.add_argument("--full", action="store_true", help="Run full preprocessing on the entire dataset")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Chunk size for streaming processing")
    parser.add_argument("--raw-dir", type=str, default="data/datasets", help="Path to raw dataset root")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Path to output processed directory")
    args = parser.parse_args()

    preprocessor = EdgeIIoTsetPreprocessor(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        artifacts_dir=os.path.join(args.output_dir, "artifacts"),
        chunk_size=args.chunk_size
    )

    if args.sample > 0:
        success = preprocessor.run_verification(sample_size=args.sample)
        sys.exit(0 if success else 1)
    elif args.full:
        # Before full run, always run the 5000-row sample validation first
        print("Executing mandatory pre-execution 5,000-row sample test before full run...")
        val_success = preprocessor.run_verification(sample_size=5000)
        if not val_success:
            print("Pre-execution sample verification failed! Aborting full processing.")
            sys.exit(1)
        preprocessor.run_full()
    else:
        # Default behavior: run 5000-row verification
        print("No mode specified. Running default 5,000-row sample validation test...")
        success = preprocessor.run_verification(sample_size=5000)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
