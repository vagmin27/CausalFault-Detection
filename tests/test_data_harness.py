"""
Unit and integration tests for CommonDataHarness, StreamConfig, label isolation,
timing boundaries, manifest generation, and dry-run execution.
"""

import unittest
import os
import tempfile
import json
import time
import numpy as np

from evaluation.data_harness import CommonDataHarness, StreamConfig, ATTACK_TO_FAULT_CATEGORY
from evaluation.context import BenchmarkInput, EventGroundTruth, BenchmarkContext
from evaluation.instrumentation import SystemInstrumentation
from evaluation.results_schema import (
    BenchmarkResultRecord,
    ResultsCollection,
    MeasurementType,
    Comparability,
)
from evaluation.reproducibility import (
    capture_reproducibility_environment,
    save_reproducibility_record,
)
from evaluation.metrics import calculate_detection_accuracy, build_result_record
from algorithms.base import (
    BaseFaultToleranceAlgorithm,
    AlgorithmCapability,
    DetectionResult,
)


class DryRunDummyAlgorithm(BaseFaultToleranceAlgorithm):
    """Simple dummy algorithm for harness dry-run only. Produces no research claims."""

    @property
    def name(self) -> str:
        return "Dry Run Dummy Algorithm"

    @property
    def paper_id(self) -> str:
        return "dry_run_dummy"

    @property
    def supported_capabilities(self):
        return {AlgorithmCapability.STREAMING_DETECTION}

    def initialize(self, config):
        self.processed_count = 0
        self.initialized = True

    def reset(self):
        self.processed_count = 0

    def detect(self, record: BenchmarkInput) -> DetectionResult:
        self.processed_count += 1
        # Simple dummy logic: score based on arbitrary feature
        score = 0.8 if record.features.get("tcp.dstport", 0.0) > 0.0 else 0.1
        return DetectionResult(is_anomaly=(score > 0.5), anomaly_score=score)


class TestDataHarness(unittest.TestCase):

    def setUp(self):
        self.fixture_path = os.path.join("tests", "fixtures", "smoke_test_telemetry.csv")
        self.assertTrue(os.path.exists(self.fixture_path), "Smoke test fixture missing!")
        self.config = StreamConfig(
            dataset_name="synthetic_fixture",
            split="test",
            data_dir=self.fixture_path,
            warmup_count=20,  # 20 warmup, 80 evaluation in 100-row fixture
            max_records=100,
            batch_size=50,
            seed=42,
        )
        self.harness = CommonDataHarness(self.config)

    def test_label_isolation_and_no_leakage(self):
        """Verify Attack_label and Attack_type are strictly invisible to BenchmarkInput."""
        contexts = list(self.harness.stream_contexts())
        self.assertEqual(len(contexts), 100)

        for ctx in contexts:
            obs = ctx.observable_input
            gt = ctx.ground_truth

            # Observable input must NOT contain Attack_label or Attack_type
            self.assertNotIn("Attack_label", obs.features)
            self.assertNotIn("Attack_type", obs.features)
            self.assertFalse(hasattr(obs, "Attack_label"))
            self.assertFalse(hasattr(obs, "Attack_type"))
            self.assertFalse(hasattr(obs, "is_fault"))

            # Ground truth must contain them
            self.assertIn(gt.fault_label, [0, 1])
            self.assertIn(gt.raw_attack_type, ["Normal", "DDoS_ICMP"])
            self.assertIsInstance(gt.is_fault, bool)

    def test_deterministic_ordering_and_stream_identity(self):
        """Verify identical record ordering across repeated iterations."""
        stream1 = list(self.harness.stream_contexts())
        stream2 = list(self.harness.stream_contexts())

        self.assertEqual(len(stream1), len(stream2))
        for c1, c2 in zip(stream1, stream2):
            self.assertEqual(c1.stream_position, c2.stream_position)
            self.assertEqual(c1.observable_input.timestamp, c2.observable_input.timestamp)
            np.testing.assert_array_equal(
                c1.observable_input.feature_vector,
                c2.observable_input.feature_vector
            )

    def test_warmup_delineation(self):
        """Verify warm-up records are strictly flagged (is_warmup=True for first 20 records)."""
        contexts = list(self.harness.stream_contexts())
        warmup_ctxs = [c for c in contexts if c.is_warmup]
        eval_ctxs = [c for c in contexts if not c.is_warmup]

        self.assertEqual(len(warmup_ctxs), 20)
        self.assertEqual(len(eval_ctxs), 80)
        self.assertTrue(all(c.stream_position < 20 for c in warmup_ctxs))
        self.assertTrue(all(c.stream_position >= 20 for c in eval_ctxs))

    def test_manifest_generation(self):
        """Verify results/raw/benchmark_manifest.json accurately describes stream."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = self.harness.generate_manifest(output_dir=tmpdir)
            manifest_path = os.path.join(tmpdir, "benchmark_manifest.json")
            self.assertTrue(os.path.exists(manifest_path))

            self.assertEqual(manifest["dataset_name"], "synthetic_fixture")
            self.assertEqual(manifest["split"], "test")
            self.assertEqual(manifest["warmup_records"], 20)
            self.assertGreater(manifest["feature_count"], 0)
            self.assertTrue(len(manifest["manifest_hash"]) > 0)

    def test_reproducibility_environment(self):
        """Verify reproducibility capture contains OS, Python, packages, and hardware."""
        env = capture_reproducibility_environment(config=self.config)
        self.assertIn("python", env)
        self.assertIn("operating_system", env)
        self.assertIn("hardware", env)
        self.assertIn("installed_packages", env)
        self.assertIn("git", env)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "env.json")
            save_reproducibility_record(env, out_path)
            self.assertTrue(os.path.exists(out_path))

    def test_not_applicable_metric_representation(self):
        """Verify NOT_APPLICABLE comparability and NR measurement type representation."""
        rec_na = build_result_record(
            algorithm="BWOAIF",
            paper_id="paper2_bwoaif",
            metric="Resources",
            submetric="Migration Bandwidth Overhead",
            value=None,
            unit="MB",
            measurement_type=MeasurementType.NR,
            comparability=Comparability.NOT_COMPARABLE,
            notes="BWOAIF does not perform migration; marked NOT_COMPARABLE, not 0.",
        )
        self.assertEqual(rec_na.measurement_type, MeasurementType.NR)
        self.assertEqual(rec_na.comparability, Comparability.NOT_COMPARABLE)
        self.assertIsNone(rec_na.value)

    def test_dry_run_pipeline_execution(self):
        """
        Complete Dry-Run Execution:
        load stream -> BenchmarkInput -> ground truth separation -> invoke DummyAlgorithm
        -> collect instrumentation -> create standardized result records.
        """
        algo = DryRunDummyAlgorithm()
        algo.initialize(self.config)
        algo.start()

        instrumentation = SystemInstrumentation(slo_threshold_ms=50.0)
        instrumentation.start_session()

        eval_y_true = []
        eval_y_pred = []
        eval_scores = []

        for ctx in self.harness.stream_contexts():
            obs = ctx.observable_input
            gt = ctx.ground_truth

            # Algorithm only receives obs
            t_start = time.perf_counter_ns()
            det_res = algo.process(obs)
            t_elapsed_ms = (time.perf_counter_ns() - t_start) / 1_000_000.0

            # Record instrumentation
            instrumentation.throughput_counter.record_observation(1)
            instrumentation.bandwidth_accountant.add_telemetry_bytes(250)
            instrumentation.action_counter.record_latency(t_elapsed_ms)

            # Exclude warm-up from performance metrics
            if not ctx.is_warmup:
                eval_y_true.append(gt.fault_label)
                eval_y_pred.append(1 if det_res.is_anomaly else 0)
                eval_scores.append(det_res.anomaly_score)

        algo.stop()
        summary = instrumentation.end_session()

        self.assertEqual(algo.processed_count, 100)
        self.assertEqual(len(eval_y_true), 80)
        self.assertGreater(summary["throughput_rec_sec"], 0.0)

        metrics = calculate_detection_accuracy(eval_y_true, eval_y_pred, eval_scores)
        self.assertIn("f1", metrics)

        # Standardized result records
        records = [
            build_result_record(
                algorithm=algo.name,
                paper_id=algo.paper_id,
                metric="Performance",
                submetric="Detection F1",
                value=metrics["f1"],
                unit="%",
                measurement_type=MeasurementType.MEASURED,
                comparability=Comparability.DIRECT,
            ),
            build_result_record(
                algorithm=algo.name,
                paper_id=algo.paper_id,
                metric="Resources",
                submetric="Processing Throughput",
                value=summary["throughput_rec_sec"],
                unit="rec/sec",
                measurement_type=MeasurementType.MEASURED,
                comparability=Comparability.DIRECT,
            ),
            build_result_record(
                algorithm=algo.name,
                paper_id=algo.paper_id,
                metric="Resources",
                submetric="Migration Bandwidth Overhead",
                value=None,
                unit="MB",
                measurement_type=MeasurementType.NR,
                comparability=Comparability.NOT_COMPARABLE,
                notes="Dry run dummy does not perform migration.",
            ),
        ]

        collection = ResultsCollection()
        collection.extend(records)
        df = collection.to_dataframe()
        self.assertEqual(len(df), 3)
        self.assertIn("measurement_type", df.columns)
        self.assertIn("comparability", df.columns)


if __name__ == "__main__":
    unittest.main()
