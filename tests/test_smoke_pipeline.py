# Unit test verifying smoke-test dataset ingestion and evaluation instrumentation.

import unittest
import os
from data.edge_iiotset_adapter import EdgeIIoTsetAdapter
from evaluation.instrumentation import SystemInstrumentation
from evaluation.metrics import calculate_detection_accuracy, build_result_record
from evaluation.results_schema import MeasurementType, Comparability, ResultsCollection


class TestSmokePipeline(unittest.TestCase):

    def test_smoke_telemetry_streaming_and_metrics(self):
        fixture_path = os.path.join("tests", "fixtures", "smoke_test_telemetry.csv")
        self.assertTrue(os.path.exists(fixture_path), "Smoke fixture not found!")

        adapter = EdgeIIoTsetAdapter(data_path=fixture_path)
        records = list(adapter.stream_telemetry())
        self.assertEqual(len(records), 100)

        # Profile execution of streaming records
        instrumentation = SystemInstrumentation(slo_threshold_ms=50.0)
        instrumentation.start_session()

        y_true = []
        y_pred = []
        scores = []

        for r in records:
            instrumentation.throughput_counter.record_observation(1)
            instrumentation.bandwidth_accountant.add_telemetry_bytes(250)
            instrumentation.action_counter.record_latency(15.0)

            y_true.append(r.fault_label)
            # Dummy threshold on workload/score
            score = 0.8 if r.fault_label == 1 else 0.1
            scores.append(score)
            y_pred.append(1 if score > 0.5 else 0)

        summary = instrumentation.end_session()

        self.assertEqual(summary["total_actions"], 0)
        self.assertEqual(summary["slo_violation_rate"], 0.0)
        self.assertGreater(summary["throughput_rec_sec"], 0.0)
        self.assertGreater(summary["bandwidth_kb_sec"], 0.0)

        det = calculate_detection_accuracy(y_true, y_pred, scores)
        self.assertEqual(det["precision"], 100.0)
        self.assertEqual(det["recall"], 100.0)
        self.assertEqual(det["f1"], 100.0)

        # Build standardized result record
        rec = build_result_record(
            algorithm="Smoke Test Detector",
            paper_id="smoke_test",
            metric="Performance",
            submetric="Detection F1",
            value=det["f1"],
            unit="%",
            measurement_type=MeasurementType.MEASURED,
            comparability=Comparability.DIRECT,
            notes="Smoke test run over synthetic fixture",
        )
        self.assertEqual(rec.value, 100.0)
        self.assertEqual(rec.measurement_type, MeasurementType.MEASURED)

        collection = ResultsCollection()
        collection.add_record(rec)
        self.assertEqual(len(collection.to_dataframe()), 1)


if __name__ == "__main__":
    unittest.main()
