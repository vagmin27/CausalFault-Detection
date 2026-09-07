"""
Unit Tests for RCD Algorithm (Ikram et al., 2022).
Validates:
1. Base interface compliance
2. Initialization and reset
3. Fitting baseline normal data
4. Localized causal discovery and F-node testing
5. Top-k root-cause ranking
6. Rejection of unsupported capabilities (detect, recover)
7. Synthetic root-cause isolation test
"""

import unittest
import numpy as np

from algorithms.base import (
    BaseFaultToleranceAlgorithm,
    AlgorithmCapability,
    RCADiagnosisResult,
)
from algorithms.rcd import RCDAlgorithm, RCDConfig
from evaluation.context import BenchmarkInput


class TestRCDAlgorithm(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.feature_names = ["net_load", "cpu_usage", "mem_usage", "req_rate", "latency"]
        self.algo = RCDAlgorithm(config={"significance_alpha": 0.05, "top_k": 3})

    def test_base_interface_and_capabilities(self):
        """Verify RCD advertises ONLY CAUSAL_ROOT_CAUSE_ANALYSIS."""
        self.assertIsInstance(self.algo, BaseFaultToleranceAlgorithm)
        self.assertEqual(self.algo.paper_id, "paper3_rcd")
        self.assertTrue(self.algo.supports(AlgorithmCapability.CAUSAL_ROOT_CAUSE_ANALYSIS))
        self.assertFalse(self.algo.supports(AlgorithmCapability.STREAMING_DETECTION))
        self.assertFalse(self.algo.supports(AlgorithmCapability.RESOURCE_PREDICTION))
        self.assertFalse(self.algo.supports(AlgorithmCapability.PREEMPTIVE_MIGRATION))
        self.assertFalse(self.algo.supports(AlgorithmCapability.CLOSED_LOOP_FAULT_TOLERANCE))

    def test_unsupported_capability_rejections(self):
        """Verify detect() and recover() raise NotImplementedError."""
        dummy_input = BenchmarkInput(
            stream_position=0, timestamp=0.0, timestamp_str="",
            device_id="", edge_node_id="", features={},
            feature_vector=np.zeros(5),
        )
        with self.assertRaises(NotImplementedError):
            self.algo.detect(dummy_input)
        with self.assertRaises(NotImplementedError):
            self.algo.recover(dummy_input)

    def test_fit_stores_baseline(self):
        """Verify fit stores normal training data."""
        X_norm = np.random.normal(0.0, 1.0, size=(50, len(self.feature_names)))
        self.algo.fit(X_norm)
        self.assertTrue(self.algo.is_fitted)
        self.assertEqual(self.algo.baseline_normal_data.shape, (50, 5))

    def test_localized_causal_discovery_synthetic_rca(self):
        """
        Verify RCD identifies the true injected fault source:
        Baseline: All features ~ N(0, 1)
        Anomalous: Specific feature (cpu_usage) experiences massive shock (mean=15.0),
        while others remain normal. RCD must identify cpu_usage as #1 root cause.
        """
        N = 60
        D = len(self.feature_names)
        X_norm = np.random.normal(0.0, 1.0, size=(N, D))

        # In anomalous window, feature index 1 ("cpu_usage") is shocked
        X_anom = np.random.normal(0.0, 1.0, size=(N, D))
        X_anom[:, 1] += 12.0  # Injected failure cause at cpu_usage

        # Fit normal baseline
        import pandas as pd
        df_norm = pd.DataFrame(X_norm, columns=self.feature_names)
        df_anom = pd.DataFrame(X_anom, columns=self.feature_names)
        self.algo.fit(df_norm)

        # Execute RCA diagnosis
        diag = self.algo.diagnose_root_cause(normal_window=df_norm, anomalous_window=df_anom)

        self.assertIsInstance(diag, RCADiagnosisResult)
        self.assertGreater(len(diag.ranked_root_causes), 0)
        self.assertEqual(diag.ranked_root_causes[0], "cpu_usage",
                         "RCD failed to isolate shocked variable 'cpu_usage' as top root cause.")
        self.assertIn("cpu_usage", diag.confidence_scores)
        self.assertGreater(diag.confidence_scores["cpu_usage"], 0.0)

    def test_reset(self):
        """Verify reset clears diagnosis cache."""
        self.algo.last_diagnosis = RCADiagnosisResult(ranked_root_causes=["test"])
        self.algo.reset()
        self.assertIsNone(self.algo.last_diagnosis)


if __name__ == "__main__":
    unittest.main()
