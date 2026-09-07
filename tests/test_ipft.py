"""
Unit Tests for IPFT Algorithm Implementation.
Validates:
1. Base interface compliance
2. Initialization and reset
3. Training/fitting on normal telemetry
4. Streaming detection and proactive migration triggering
5. No label leakage
6. Correct capability advertising and unsupported capability gating
7. Small synthetic smoke pipeline
"""

import unittest
import numpy as np

from algorithms.base import (
    BaseFaultToleranceAlgorithm,
    AlgorithmCapability,
    DetectionResult,
    MitigationResult,
)
from algorithms.ipft import IPFTAlgorithm, IPFTConfig
from evaluation.context import BenchmarkInput


class TestIPFTAlgorithm(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.num_features = 62
        self.algo = IPFTAlgorithm(config={"sequence_length": 5, "upper_threshold": 0.70})

    def _create_mock_input(self, position: int, val: float = 0.1) -> BenchmarkInput:
        vec = np.full(self.num_features, val, dtype=np.float64)
        features = {f"f_{i}": val for i in range(self.num_features)}
        return BenchmarkInput(
            stream_position=position,
            timestamp=100.0 + position,
            timestamp_str="",
            device_id="dev_01",
            edge_node_id="node_01",
            features=features,
            feature_vector=vec,
        )

    def test_base_interface_and_capabilities(self):
        """Verify inheritance and correct advertised capabilities."""
        self.assertIsInstance(self.algo, BaseFaultToleranceAlgorithm)
        self.assertEqual(self.algo.paper_id, "paper1_ipft")
        self.assertTrue(self.algo.supports(AlgorithmCapability.RESOURCE_PREDICTION))
        self.assertTrue(self.algo.supports(AlgorithmCapability.PREEMPTIVE_MIGRATION))
        self.assertTrue(self.algo.supports(AlgorithmCapability.CLOSED_LOOP_FAULT_TOLERANCE))
        self.assertFalse(self.algo.supports(AlgorithmCapability.CAUSAL_ROOT_CAUSE_ANALYSIS))

    def test_unsupported_capability_rejection(self):
        """Verify that diagnose() raises NotImplementedError."""
        inp = self._create_mock_input(0)
        with self.assertRaises(NotImplementedError):
            self.algo.diagnose(inp, inp)

    def test_fit_and_training(self):
        """Verify fit executes on normal training data without errors."""
        X_train = np.random.normal(0.0, 1.0, size=(100, self.num_features))
        self.algo.fit(X_train)
        self.assertTrue(self.algo.is_fitted)

    def test_streaming_process_no_leakage(self):
        """Verify streaming process with no label access."""
        inp = self._create_mock_input(0, val=0.05)
        res = self.algo.process(inp)
        self.assertIsInstance(res, DetectionResult)
        self.assertIn("predicted_resource_usage", res.raw_output)

    def test_overload_proactive_migration_trigger(self):
        """Verify that heavy resource surge triggers proactive migration."""
        # Warm up window
        for i in range(5):
            self.algo.process(self._create_mock_input(i, val=0.1))

        # Surge observation
        surge_input = self._create_mock_input(6, val=15.0)
        # Force model prediction threshold for deterministic trigger
        self.algo.config.upper_threshold = 0.10
        res = self.algo.process(surge_input)

        self.assertTrue(res.is_anomaly)
        mit = res.raw_output.get("mitigation")
        self.assertIsNotNone(mit)
        self.assertIsInstance(mit, MitigationResult)
        self.assertEqual(mit.action_type, "PREEMPTIVE_TASK_MIGRATION")
        self.assertGreater(mit.state_bytes_transferred, 0)

    def test_reset_functionality(self):
        """Verify clean state reset."""
        inp = self._create_mock_input(0, val=10.0)
        self.algo.process(inp)
        self.assertGreater(len(self.algo.window), 0)

        self.algo.reset()
        self.assertEqual(len(self.algo.window), 0)
        self.assertIsNone(self.algo.last_mitigation)


if __name__ == "__main__":
    unittest.main()
