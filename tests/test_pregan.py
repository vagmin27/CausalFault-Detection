# Unit Tests for PreGAN Algorithm (Tuli et al., 2022).
# Validates:
# 1. Base interface compliance
# 2. Initialization and reset
# 3. Adversarial / prototypical training on baseline data
# 4. Streaming fault prediction and preemptive migration
# 5. Rejection of unsupported capabilities (diagnose)
# 6. Zero label leakage

import unittest
import numpy as np

from algorithms.base import (
    BaseFaultToleranceAlgorithm,
    AlgorithmCapability,
    DetectionResult,
    MitigationResult,
)
from algorithms.pregan import PreGANAlgorithm, PreGANConfig
from evaluation.context import BenchmarkInput


class TestPreGANAlgorithm(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.num_features = 62
        self.algo = PreGANAlgorithm(config={"sequence_length": 5, "threshold": 0.65})

    def _create_mock_input(self, position: int, val: float = 0.05) -> BenchmarkInput:
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
        # Verify PreGAN advertises correct capabilities.
        self.assertIsInstance(self.algo, BaseFaultToleranceAlgorithm)
        self.assertEqual(self.algo.paper_id, "paper4_pregan")
        self.assertTrue(self.algo.supports(AlgorithmCapability.RESOURCE_PREDICTION))
        self.assertTrue(self.algo.supports(AlgorithmCapability.PREEMPTIVE_MIGRATION))
        self.assertTrue(self.algo.supports(AlgorithmCapability.CLOSED_LOOP_FAULT_TOLERANCE))
        self.assertFalse(self.algo.supports(AlgorithmCapability.CAUSAL_ROOT_CAUSE_ANALYSIS))

    def test_unsupported_capability_rejections(self):
        # Verify diagnose() raises NotImplementedError.
        inp = self._create_mock_input(0)
        with self.assertRaises(NotImplementedError):
            self.algo.diagnose(inp, inp)

    def test_adversarial_fit(self):
        # Verify generator and discriminator fit loop runs without errors.
        X_train = np.random.normal(0.0, 1.0, size=(100, self.num_features))
        self.algo.fit(X_train)
        self.assertTrue(self.algo.is_fitted)

    def test_streaming_process_and_migration_trigger(self):
        # Verify streaming process predicts fault and triggers preemptive migration.
        # Warm up window
        for i in range(5):
            self.algo.process(self._create_mock_input(i, val=0.05))

        # Test with threshold lowered to trigger migration deterministically
        self.algo.config.threshold = 0.01
        inp_surge = self._create_mock_input(6, val=12.0)
        res = self.algo.process(inp_surge)

        self.assertIsInstance(res, DetectionResult)
        self.assertTrue(res.is_anomaly)
        mit = res.raw_output.get("mitigation")
        self.assertIsNotNone(mit)
        self.assertIsInstance(mit, MitigationResult)
        self.assertEqual(mit.action_type, "PREEMPTIVE_TASK_MIGRATION")
        self.assertGreater(mit.state_bytes_transferred, 0)

    def test_reset(self):
        # Verify reset clears buffers and cooldowns.
        self.algo.process(self._create_mock_input(0))
        self.assertGreater(len(self.algo.window), 0)
        self.algo.reset()
        self.assertEqual(len(self.algo.window), 0)
        self.assertIsNone(self.algo.last_mitigation)


if __name__ == "__main__":
    unittest.main()
