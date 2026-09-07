"""
Unit and Integration Tests for Proposed Causal Fault-Tolerance Pipeline.

Validates:
1. BaseFaultToleranceAlgorithm lifecycle and contract compatibility
2. Model training and baseline SEM fitting
3. Streaming processing with no ground-truth leakage
4. Real-time detection output schemas
5. Causal root-cause analysis and ranking
6. Synthetic A -> B -> C causal validation test
7. Adaptive recovery policy and simulated edge state
8. Bounded state memory management
9. Reset functionality
10. End-to-end smoke pipeline with SystemInstrumentation
"""

import unittest
import numpy as np
import networkx as nx

from algorithms.base import (
    BaseFaultToleranceAlgorithm,
    AlgorithmCapability,
    DetectionResult,
    RCADiagnosisResult,
    MitigationResult,
)
from algorithms.causal_ft import (
    CausalFaultTolerancePipeline,
    StreamingCausalDetector,
    CausalInferenceEngine,
    CausalGraphSpecification,
    AdaptiveCausalRecoveryPolicy,
    RecoveryActionType,
    CausalFTState,
)
from evaluation.context import BenchmarkInput, EventGroundTruth, BenchmarkContext
from evaluation.instrumentation import SystemInstrumentation


class TestCausalFaultTolerance(unittest.TestCase):
    """Test suite for the Proposed Causal Fault-Tolerance architecture."""

    def setUp(self):
        np.random.seed(42)
        self.feature_names = [
            "tcp.flags", "tcp_active_flags_count", "tcp_seq_ack_ratio",
            "udp.port", "udp.stream", "udp.time_delta",
            "http.content_length", "http.response"
        ]
        self.pipeline = CausalFaultTolerancePipeline(config={"detection_threshold": 0.60})

    def _create_mock_input(
        self,
        position: int,
        feature_dict: dict,
        edge_node_id: str = "192.168.0.101"
    ) -> BenchmarkInput:
        vec = np.array([feature_dict.get(k, 0.0) for k in self.feature_names], dtype=np.float64)
        return BenchmarkInput(
            stream_position=position,
            timestamp=1000.0 + position,
            timestamp_str="2026-09-08 00:00:00",
            device_id="sensor_01",
            edge_node_id=edge_node_id,
            features=feature_dict,
            feature_vector=vec,
        )

    def test_base_contract_and_capabilities(self):
        """Verify inheritance, lifecycle methods, and advertised capabilities."""
        self.assertIsInstance(self.pipeline, BaseFaultToleranceAlgorithm)
        self.assertEqual(self.pipeline.paper_id, "proposed_causal_ft")
        self.assertTrue(self.pipeline.supports(AlgorithmCapability.STREAMING_DETECTION))
        self.assertTrue(self.pipeline.supports(AlgorithmCapability.CAUSAL_ROOT_CAUSE_ANALYSIS))
        self.assertTrue(self.pipeline.supports(AlgorithmCapability.CLOSED_LOOP_FAULT_TOLERANCE))
        self.assertFalse(self.pipeline.supports(AlgorithmCapability.RESOURCE_PREDICTION))

    def test_training_and_fit(self):
        """Verify model fitting on normal training data without labels."""
        N = 200
        D = len(self.feature_names)
        X_normal = np.random.normal(loc=0.0, scale=1.0, size=(N, D))

        # Fit pipeline
        self.pipeline.fit(X_normal)
        self.assertTrue(self.pipeline.detector.is_fitted)
        self.assertTrue(self.pipeline.causal_engine.is_fitted)
        self.assertEqual(len(self.pipeline.detector.baseline_mean), D)

    def test_streaming_detection_no_leakage(self):
        """Verify detector produces valid DetectionResult with zero ground-truth knowledge."""
        normal_feats = {k: 0.05 for k in self.feature_names}
        inp = self._create_mock_input(0, normal_feats)

        # Algorithm receives ONLY BenchmarkInput
        res = self.pipeline.detect(inp)
        self.assertIsInstance(res, DetectionResult)
        self.assertIn("anomaly_score", res.__dict__)
        self.assertIn("is_anomaly", res.__dict__)
        self.assertFalse(res.is_anomaly)
        self.assertLess(res.anomaly_score, 0.60)

    def test_synthetic_causal_chain_validation(self):
        """
        CAUSAL VALIDATION TEST (Requirement 16):
        Synthetic chain: A -> B -> C
        Verify that structural residual analysis distinguishes root cause A from
        downstream symptoms B and C, even when B and C have higher absolute values.
        """
        graph_spec = CausalGraphSpecification()
        graph_spec.add_causal_edge("A", "B", weight=2.0, is_domain=False)
        graph_spec.add_causal_edge("B", "C", weight=1.5, is_domain=False)

        engine = CausalInferenceEngine(causal_graph=graph_spec)

        # Baseline training data where B = 2*A, C = 1.5*B
        N = 300
        A_train = np.random.normal(0.0, 1.0, size=N)
        B_train = 2.0 * A_train + np.random.normal(0.0, 0.05, size=N)
        C_train = 1.5 * B_train + np.random.normal(0.0, 0.05, size=N)
        X_train = np.column_stack([A_train, B_train, C_train])

        engine.fit_structural_equations(X_train, feature_names=["A", "B", "C"])

        # Case 1: Root cause shock introduced at A (A=10 -> B=20, C=30)
        # Downstream variables B and C have higher values, but A is the root cause.
        shock_A_input = BenchmarkInput(
            stream_position=1,
            timestamp=1001.0,
            timestamp_str="",
            device_id="dev",
            edge_node_id="edge",
            features={"A": 10.0, "B": 20.0, "C": 30.0},
            feature_vector=np.array([10.0, 20.0, 30.0]),
        )
        diag_A = engine.diagnose(shock_A_input, top_k=3)
        self.assertEqual(diag_A.ranked_root_causes[0], "A",
                         "Causal engine failed to identify A as root cause in A->B->C propagation.")

        # Case 2: Direct localized fault introduced at C only (A=0.1, B=0.2, C=15.0)
        shock_C_input = BenchmarkInput(
            stream_position=2,
            timestamp=1002.0,
            timestamp_str="",
            device_id="dev",
            edge_node_id="edge",
            features={"A": 0.1, "B": 0.2, "C": 15.0},
            feature_vector=np.array([0.1, 0.2, 15.0]),
        )
        diag_C = engine.diagnose(shock_C_input, top_k=3)
        self.assertEqual(diag_C.ranked_root_causes[0], "C",
                         "Causal engine failed to isolate localized failure at C.")

    def test_synthetic_fork_confounding_validation(self):
        """
        SYNTHETIC FORK/CONFOUNDING TEST:
        Z -> X, Z -> Y with NO direct edge between X and Y.
        Verify that a shock in common cause Z does not cause the engine to falsely infer
        X -> Y or Y -> X, and correctly identifies Z as the common root cause.
        """
        graph_spec = CausalGraphSpecification()
        graph_spec.add_causal_edge("Z", "X", weight=2.0, is_domain=False)
        graph_spec.add_causal_edge("Z", "Y", weight=3.0, is_domain=False)

        # Confirm no direct edge exists between X and Y
        self.assertFalse(graph_spec.dag.has_edge("X", "Y"))
        self.assertFalse(graph_spec.dag.has_edge("Y", "X"))

        engine = CausalInferenceEngine(causal_graph=graph_spec)

        # Baseline training data where X = 2*Z, Y = 3*Z (X and Y are strongly correlated)
        N = 300
        Z_train = np.random.normal(0.0, 1.0, size=N)
        X_train = 2.0 * Z_train + np.random.normal(0.0, 0.05, size=N)
        Y_train = 3.0 * Z_train + np.random.normal(0.0, 0.05, size=N)
        train_mat = np.column_stack([Z_train, X_train, Y_train])

        engine.fit_structural_equations(train_mat, feature_names=["Z", "X", "Y"])

        # Shock introduced at common cause Z (Z=10 -> X=20, Y=30)
        # Even though X and Y are correlated and have large values, Z is the origin
        shock_Z_input = BenchmarkInput(
            stream_position=10,
            timestamp=1010.0,
            timestamp_str="",
            device_id="dev",
            edge_node_id="edge",
            features={"Z": 10.0, "X": 20.0, "Y": 30.0},
            feature_vector=np.array([10.0, 20.0, 30.0]),
        )
        diag = engine.diagnose(shock_Z_input, top_k=3)

        # 1. Z must be identified as top root cause
        self.assertEqual(diag.ranked_root_causes[0], "Z",
                         "Causal engine failed to identify common cause Z as root cause in fork structure.")

        # 2. Residuals for X and Y should be near 0 because they are explained by parent Z
        res_X = diag.metadata["residuals"]["X"]
        res_Y = diag.metadata["residuals"]["Y"]
        self.assertLess(res_X, 0.5, f"Residual of X should be near 0, got {res_X}")
        self.assertLess(res_Y, 0.5, f"Residual of Y should be near 0, got {res_Y}")

        # 3. Structural graph must preserve absence of spurious edges between siblings
        self.assertNotIn("X", engine.causal_graph.get_parents("Y"))
        self.assertNotIn("Y", engine.causal_graph.get_parents("X"))

    def test_adaptive_recovery_escalation(self):
        """Verify adaptive escalation from rate-limiting to preemptive migration on consecutive faults."""
        state = CausalFTState(window_size=50)
        policy = AdaptiveCausalRecoveryPolicy(state)

        node_id = "192.168.0.101"
        diag = RCADiagnosisResult(
            ranked_root_causes=["icmp.checksum", "tcp.flags"],
            confidence_scores={"icmp.checksum": 0.85},
            execution_time_ms=0.5,
        )

        # Fault 1: Rate limiting
        inp1 = self._create_mock_input(1, {"icmp.checksum": 5.0}, edge_node_id=node_id)
        act1 = policy.decide_recovery(inp1, diag)
        self.assertEqual(act1.action_type, RecoveryActionType.RATE_LIMIT_TRAFFIC.value)

        # Fault 2: Isolation
        inp2 = self._create_mock_input(10, {"icmp.checksum": 6.0}, edge_node_id=node_id)
        act2 = policy.decide_recovery(inp2, diag)
        self.assertEqual(act2.action_type, RecoveryActionType.ISOLATE_PORT_FLOW.value)

        # Fault 3: Preemptive migration
        inp3 = self._create_mock_input(20, {"icmp.checksum": 7.0}, edge_node_id=node_id)
        act3 = policy.decide_recovery(inp3, diag)
        self.assertEqual(act3.action_type, RecoveryActionType.PREEMPTIVE_TASK_MIGRATION.value)
        self.assertEqual(act3.target_node, "edge_backup_node")
        self.assertGreater(act3.state_bytes_transferred, 0)

    def test_bounded_state_memory(self):
        """Verify that streaming 500 records maintains fixed bounded ring buffer size."""
        pipeline = CausalFaultTolerancePipeline(config={"window_size": 30})
        for i in range(150):
            feats = {k: float(i % 5) for k in self.feature_names}
            inp = self._create_mock_input(i, feats)
            pipeline.process(inp)

        self.assertLessEqual(len(pipeline.state.telemetry_buffer), 30)
        self.assertLessEqual(len(pipeline.state.anomaly_scores), 30)
        self.assertLessEqual(len(pipeline.state.detected_faults), 30)

    def test_reset_functionality(self):
        """Verify complete state and counter cleanup on reset()."""
        inp = self._create_mock_input(0, {k: 10.0 for k in self.feature_names})
        self.pipeline.process(inp)
        self.assertIsNotNone(self.pipeline.last_diagnosis)

        self.pipeline.reset()
        self.assertIsNone(self.pipeline.last_diagnosis)
        self.assertIsNone(self.pipeline.last_mitigation)
        self.assertEqual(len(self.pipeline.state.telemetry_buffer), 0)

    def test_end_to_end_smoke_pipeline(self):
        """
        Smoke test (Requirement 17):
        Stream records through pipeline with SystemInstrumentation and verify:
        input -> detection -> causal analysis -> recovery decision -> instrumentation.
        """
        pipeline = CausalFaultTolerancePipeline(config={"detection_threshold": 0.50})
        instrumentation = SystemInstrumentation()

        instrumentation.start_session()
        for i in range(25):
            # Introduce anomaly at record 15
            val = 15.0 if i >= 15 else 0.05
            feats = {k: val for k in self.feature_names}
            inp = self._create_mock_input(i, feats)

            with instrumentation.timer:
                res = pipeline.process(inp)

            instrumentation.throughput_counter.record_observation(1)
            instrumentation.bandwidth_accountant.add_telemetry_bytes(len(inp.feature_vector) * 8)

            if res.is_anomaly:
                diag = res.raw_output.get("diagnosis")
                mit = res.raw_output.get("mitigation")
                self.assertIsNotNone(diag)
                self.assertIsNotNone(mit)
                if mit.state_bytes_transferred > 0:
                    instrumentation.bandwidth_accountant.add_migration_state_bytes(mit.state_bytes_transferred)
                instrumentation.action_counter.record_action("MIGRATION" if "MIGRATION" in mit.action_type else "REBALANCE")

        summary = instrumentation.end_session()

        self.assertGreater(summary["throughput_rec_sec"], 0.0)
        self.assertIn("peak_rss_mb", summary)
        self.assertIn("estimated_energy_joules", summary)
        self.assertGreaterEqual(summary["total_actions"], 1)


if __name__ == "__main__":
    unittest.main()
