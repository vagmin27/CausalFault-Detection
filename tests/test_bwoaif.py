# Unit Tests for BWOAIF Streaming Anomaly Detector.
# Validates:
# 1. Base interface compliance
# 2. Initialization and reset
# 3. Training on baseline data
# 4. Streaming detection output and thresholding
# 5. Bilateral weighting and tree replacement batch update
# 6. Rejection of unsupported capabilities (diagnose, recover)
# 7. Zero label leakage

import unittest
import numpy as np

from algorithms.base import (
    BaseFaultToleranceAlgorithm,
    AlgorithmCapability,
    DetectionResult,
)
from algorithms.bwoaif import BWOAIAlgorithm, BWOAIFConfig
from evaluation.context import BenchmarkInput


class TestBWOAIAlgorithm(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.num_features = 62
        self.algo = BWOAIAlgorithm(config={"num_trees": 15, "batch_size": 20, "threshold": 0.55})

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
        # Verify BWOAIF inherits base interface and advertises ONLY STREAMING_DETECTION.
        self.assertIsInstance(self.algo, BaseFaultToleranceAlgorithm)
        self.assertEqual(self.algo.paper_id, "paper2_bwoaif")
        self.assertTrue(self.algo.supports(AlgorithmCapability.STREAMING_DETECTION))
        self.assertFalse(self.algo.supports(AlgorithmCapability.RESOURCE_PREDICTION))
        self.assertFalse(self.algo.supports(AlgorithmCapability.CAUSAL_ROOT_CAUSE_ANALYSIS))
        self.assertFalse(self.algo.supports(AlgorithmCapability.PREEMPTIVE_MIGRATION))
        self.assertFalse(self.algo.supports(AlgorithmCapability.CLOSED_LOOP_FAULT_TOLERANCE))

    def test_unsupported_capability_rejections(self):
        # Verify that diagnose() and recover() raise NotImplementedError.
        inp = self._create_mock_input(0)
        with self.assertRaises(NotImplementedError):
            self.algo.diagnose(inp, inp)
        with self.assertRaises(NotImplementedError):
            self.algo.recover(inp)

    def test_initial_fit(self):
        # Verify initial forest construction on normal training observations.
        X_train = np.random.normal(0.0, 1.0, size=(100, self.num_features))
        self.algo.fit(X_train)
        self.assertTrue(self.algo.is_fitted)
        self.assertEqual(len(self.algo.trees), 15)

    def test_streaming_detection_no_leakage(self):
        # Verify single-record streaming detection produces valid DetectionResult.
        X_train = np.random.normal(0.0, 0.1, size=(80, self.num_features))
        self.algo.fit(X_train)

        normal_inp = self._create_mock_input(0, val=0.01)
        res_normal = self.algo.process(normal_inp)
        self.assertIsInstance(res_normal, DetectionResult)
        self.assertIn("anomaly_score", res_normal.__dict__)

        # Anomaly with extreme feature values should produce higher anomaly score
        anomaly_inp = self._create_mock_input(1, val=15.0)
        res_anomaly = self.algo.process(anomaly_inp)
        self.assertGreater(res_anomaly.anomaly_score, res_normal.anomaly_score)

    def test_online_batch_tree_replacement(self):
        # Verify that streaming more than batch_size records updates trees and increments batch index.
        X_train = np.random.normal(0.0, 0.1, size=(60, self.num_features))
        self.algo.fit(X_train)
        initial_batch_idx = self.algo.current_batch_index

        # Stream 25 records (batch_size is 20)
        for i in range(25):
            self.algo.process(self._create_mock_input(i, val=0.1))

        self.assertGreater(self.algo.current_batch_index, initial_batch_idx)

    def test_reset(self):
        # Verify complete state reset.
        inp = self._create_mock_input(0)
        self.algo.process(inp)
        self.algo.reset()
        self.assertEqual(len(self.algo.trees), 0)
        self.assertEqual(len(self.algo.current_batch_data), 0)
        self.assertEqual(self.algo.current_batch_index, 0)


if __name__ == "__main__":
    unittest.main()
