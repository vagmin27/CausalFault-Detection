# Unit Tests for Centralized Metric Calculations and Result Schema.
#
# Tests all 10 Headline Evaluation Parameters:
# 1. Latency / Delay (MTTD, Diagnosis, Recovery, End-to-End)
# 2. Execution / Response Time (Inference ms/rec, Response time, Overhead ratio)
# 3. Accuracy (Detection F1/ROC-AUC, RCA Top-k Recall, Recovery Success Rate)
# 4. Computational Capacity & Throughput (Hardware Profile, Records/sec)
# 5. Resource Utilization (CPU %, Peak RSS MB)
# 6. Bandwidth (Streaming KB/s, Migration MB)
# 7. Energy Consumption (Estimated Joules & Watt-hours)
# 8. Operational Recovery Cost (SLO violation rate, Normalized penalty score)
# 9. Reliability & Availability (Availability %, Task reliability %, MTTF/MTTR)
# 10. Scalability (Scaling exponent alpha)
# 11. Results Schema (MeasurementType, Comparability, NR handling, CSV serialization)

import unittest
import math
import tempfile
import os
import pandas as pd
import numpy as np

from evaluation.metrics import (
    calculate_mean_std,
    calculate_confidence_interval,
    calculate_detection_latency_mttd,
    calculate_diagnosis_latency,
    calculate_recovery_latency,
    calculate_end_to_end_latency,
    calculate_per_record_inference_latency,
    calculate_service_response_time,
    calculate_algorithmic_overhead_ratio,
    calculate_detection_accuracy,
    calculate_rca_localization_accuracy,
    calculate_recovery_success_rate,
    get_hardware_capacity_profile,
    calculate_processing_throughput,
    calculate_average_cpu_utilization,
    calculate_peak_ram_footprint,
    calculate_telemetry_streaming_bandwidth,
    calculate_migration_bandwidth_overhead,
    calculate_estimated_energy_joules,
    calculate_estimated_energy_watt_hours,
    calculate_slo_violation_rate,
    calculate_operational_recovery_cost,
    calculate_service_availability,
    calculate_task_reliability,
    calculate_mttf_mttr,
    calculate_complexity_scaling_exponent,
    build_result_record,
)
from evaluation.results_schema import (
    BenchmarkResultRecord,
    ResultsCollection,
    MeasurementType,
    Comparability,
)
from evaluation.config import ExperimentConfig
from evaluation.instrumentation import (
    ExecutionTimer,
    ResourceMonitor,
    ThroughputCounter,
    BandwidthAccountant,
    EnergyEstimator,
    ActionCounter,
)


class TestMetricCalculations(unittest.TestCase):

    def test_statistical_mean_std_and_ci(self):
        # 5 runs sample
        values = [10.0, 12.0, 11.0, 10.5, 11.5]
        mean, std = calculate_mean_std(values)
        self.assertAlmostEqual(mean, 11.0, places=4)
        self.assertGreater(std, 0.0)

        ci = calculate_confidence_interval(values, confidence=0.95)
        self.assertEqual(len(ci), 2)
        self.assertLess(ci[0], mean)
        self.assertGreater(ci[1], mean)

        # Single value edge case
        m_single, s_single = calculate_mean_std([42.0])
        self.assertEqual(m_single, 42.0)
        self.assertEqual(s_single, 0.0)

        # Empty list edge case
        m_empty, s_empty = calculate_mean_std([])
        self.assertEqual(m_empty, 0.0)
        self.assertEqual(s_empty, 0.0)

    def test_latency_delay_metrics(self):
        # Fault onsets at 100, 200, 300; detects at 120, 230, 310
        onsets = [100.0, 200.0, 300.0]
        detects = [120.0, 230.0, 310.0]
        diagnoses = [150.0, 260.0, 340.0]
        recoveries = [180.0, 290.0, 370.0]

        mttd = calculate_detection_latency_mttd(onsets, detects)
        self.assertAlmostEqual(mttd, (20.0 + 30.0 + 10.0) / 3.0, places=2)

        t_diag = calculate_diagnosis_latency(detects, diagnoses)
        self.assertAlmostEqual(t_diag, 30.0, places=2)

        t_act = calculate_recovery_latency(diagnoses, recoveries)
        self.assertAlmostEqual(t_act, 30.0, places=2)

        t_e2e = calculate_end_to_end_latency(onsets, recoveries)
        self.assertAlmostEqual(t_e2e, 80.0, places=2)

    def test_execution_time_metrics(self):
        inferences = [0.05, 0.06, 0.04, 0.05]
        per_rec = calculate_per_record_inference_latency(inferences)
        self.assertAlmostEqual(per_rec, 0.05, places=3)

        responses = [120.0, 150.0, 110.0]
        resp_time = calculate_service_response_time(responses)
        self.assertAlmostEqual(resp_time, 126.6667, places=2)

        overhead = calculate_algorithmic_overhead_ratio(10.0, 100.0)
        self.assertEqual(overhead, 0.10)

    def test_accuracy_detection_rca_recovery(self):
        # 1. Detection accuracy
        y_true = [0, 0, 1, 1, 1, 0, 1, 0]
        y_pred = [0, 0, 1, 0, 1, 0, 1, 1]  # 3 TP, 1 FP, 1 FN, 3 TN
        scores = [0.1, 0.2, 0.9, 0.4, 0.8, 0.3, 0.85, 0.7]

        det_metrics = calculate_detection_accuracy(y_true, y_pred, scores)
        self.assertGreater(det_metrics["precision"], 0.0)
        self.assertGreater(det_metrics["recall"], 0.0)
        self.assertGreater(det_metrics["f1"], 0.0)
        self.assertGreater(det_metrics["roc_auc"], 0.5)

        # 2. RCA Top-k Recall
        ground_truth = ["cpu_overload", "mem_leak", "net_congestion"]
        rankings = [
            ["cpu_overload", "disk_io", "net_congestion"],      # hit top 1
            ["dns_flood", "mem_leak", "cpu_overload"],          # hit top 2 (top 3)
            ["io_throttle", "packet_drop", "mem_leak", "net_congestion"] # hit top 4 (top 5)
        ]
        rca_metrics = calculate_rca_localization_accuracy(ground_truth, rankings)
        self.assertAlmostEqual(rca_metrics["top_1_recall"], 33.33, places=1)
        self.assertAlmostEqual(rca_metrics["top_3_recall"], 66.67, places=1)
        self.assertAlmostEqual(rca_metrics["top_5_recall"], 100.0, places=1)

        # 3. Recovery success rate
        rec_rate = calculate_recovery_success_rate(total_actions=20, successful_actions=18)
        self.assertEqual(rec_rate, 90.0)

    def test_computational_capacity_and_throughput(self):
        cfg = ExperimentConfig()
        profile = get_hardware_capacity_profile(cfg)
        self.assertIn("cpu_cores_logical", profile)
        self.assertIn("ram_total_gb", profile)

        throughput = calculate_processing_throughput(num_records=10000, total_wall_time_sec=2.5)
        self.assertEqual(throughput, 4000.0)

    def test_resource_utilization_and_bandwidth(self):
        cpu_samples = [15.0, 25.0, 20.0]
        avg_cpu = calculate_average_cpu_utilization(cpu_samples)
        self.assertEqual(avg_cpu, 20.0)

        ram_bytes = 256 * 1024 * 1024  # 256 MB
        peak_ram = calculate_peak_ram_footprint(ram_bytes)
        self.assertEqual(peak_ram, 256.0)

        # Bandwidth
        stream_bw = calculate_telemetry_streaming_bandwidth(telemetry_bytes=1024 * 1024, elapsed_sec=10.0)
        self.assertAlmostEqual(stream_bw, 102.4, places=1)

        mig_bw = calculate_migration_bandwidth_overhead(migration_bytes=50 * 1024 * 1024)
        self.assertEqual(mig_bw, 50.0)

    def test_energy_estimation(self):
        # 50% CPU over 10 seconds: P = 2.7 + (6.4 - 2.7) * 0.5 = 4.55 W; Energy = 45.5 Joules
        joules = calculate_estimated_energy_joules(avg_cpu_percent=50.0, elapsed_sec=10.0, idle_watts=2.7, peak_watts=6.4)
        self.assertAlmostEqual(joules, 45.5, places=2)

        wh = calculate_estimated_energy_watt_hours(joules)
        self.assertAlmostEqual(wh, 45.5 / 3600.0, places=4)

    def test_operational_recovery_cost_and_reliability(self):
        slo_rate = calculate_slo_violation_rate(total_tasks=1000, violated_tasks=40)
        self.assertEqual(slo_rate, 4.0)

        op_cost = calculate_operational_recovery_cost(
            total_actions=20,
            total_tasks=1000,
            slo_violations=40,
            overhead_ratio=0.05
        )
        self.assertGreater(op_cost, 0.0)

        avail = calculate_service_availability(total_operational_sec=1000.0, total_downtime_sec=20.0)
        self.assertEqual(avail, 98.0)

        rel = calculate_task_reliability(total_tasks=500, successful_tasks=475)
        self.assertEqual(rel, 95.0)

        mttf, mttr = calculate_mttf_mttr(healthy_time_sec=900.0, repair_time_sec=30.0, fault_count=3)
        self.assertEqual(mttf, 300.0)
        self.assertEqual(mttr, 10.0)

    def test_scalability_exponent(self):
        # Linear scaling: T = c * Scale^1.0 -> alpha = 1.0
        scales = [10.0, 20.0, 40.0, 80.0]
        times = [1.0, 2.0, 4.0, 8.0]
        alpha = calculate_complexity_scaling_exponent(scales, times)
        self.assertAlmostEqual(alpha, 1.0, places=2)

        # Quadratic scaling: T = c * Scale^2.0 -> alpha = 2.0
        times_quad = [1.0, 4.0, 16.0, 64.0]
        alpha_quad = calculate_complexity_scaling_exponent(scales, times_quad)
        self.assertAlmostEqual(alpha_quad, 2.0, places=2)

    def test_results_schema_and_nr_handling(self):
        # 1. Measured record with repeated runs
        rec_measured = build_result_record(
            algorithm="Causal FT",
            paper_id="proposed_causal_ft",
            metric="Performance",
            submetric="Detection Latency",
            value=25.0,
            unit="ms",
            measurement_type=MeasurementType.MEASURED,
            comparability=Comparability.DIRECT,
            run_values=[24.0, 25.5, 26.0, 24.5, 25.0],
        )
        self.assertEqual(rec_measured.measurement_type, MeasurementType.MEASURED)
        self.assertAlmostEqual(rec_measured.mean, 25.0, places=2)
        self.assertIsNotNone(rec_measured.std)
        self.assertIsNotNone(rec_measured.confidence_interval)

        # 2. Not Reported (NR) paper record
        rec_nr = build_result_record(
            algorithm="Paper 2 – BWOAIF",
            paper_id="paper2_bwoaif",
            metric="System Quality",
            submetric="Mean Time To Failure",
            value=None,
            unit="s",
            measurement_type=MeasurementType.NR,
            comparability=Comparability.NOT_COMPARABLE,
            notes="BWOAIF is a standalone detector; MTTF is not reported by authors.",
        )
        self.assertEqual(rec_nr.measurement_type, MeasurementType.NR)
        self.assertIsNone(rec_nr.value)
        self.assertIsNone(rec_nr.mean)
        self.assertIsNone(rec_nr.std)
        self.assertIsNone(rec_nr.confidence_interval)

        # 3. Results collection export
        collection = ResultsCollection()
        collection.add_record(rec_measured)
        collection.add_record(rec_nr)

        df = collection.to_dataframe()
        self.assertEqual(len(df), 2)
        self.assertIn("measurement_type", df.columns)
        self.assertIn("comparability", df.columns)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            collection.save_csv(tmp_path)
            loaded_df = pd.read_csv(tmp_path)
            self.assertEqual(len(loaded_df), 2)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
