# Unit tests for algorithms/base.py interfaces, capability checking, and result types.

import unittest
from typing import Set

from algorithms.base import (
    BaseFaultToleranceAlgorithm,
    AlgorithmCapability,
    DetectionResult,
    RCADiagnosisResult,
    MitigationResult,
)


class DummyDetector(BaseFaultToleranceAlgorithm):
    # Concrete dummy implementation supporting only streaming detection.

    @property
    def name(self) -> str:
        return "Dummy Detector"

    @property
    def paper_id(self) -> str:
        return "dummy_detector"

    @property
    def supported_capabilities(self) -> Set[AlgorithmCapability]:
        return {AlgorithmCapability.STREAMING_DETECTION}

    def initialize(self, config):
        self.initialized = True

    def reset(self):
        self.reset_called = True

    def detect(self, record):
        return DetectionResult(is_anomaly=False, anomaly_score=0.15)


class TestAlgorithmsBase(unittest.TestCase):

    def test_capabilities_and_methods(self):
        algo = DummyDetector()
        algo.initialize(None)
        self.assertTrue(algo.initialized)

        algo.start()
        res_proc = algo.process({"x": 1.0})
        self.assertFalse(res_proc.is_anomaly)
        algo.stop()
        algo.initialize(None)
        self.assertTrue(algo.initialized)

        # Test capability querying
        self.assertTrue(algo.supports(AlgorithmCapability.STREAMING_DETECTION))
        self.assertFalse(algo.supports(AlgorithmCapability.CAUSAL_ROOT_CAUSE_ANALYSIS))
        self.assertFalse(algo.supports(AlgorithmCapability.PREEMPTIVE_MIGRATION))

        # Test supported observation processing
        det = algo.process_observation({"x": 1.0})
        self.assertIsInstance(det, DetectionResult)
        self.assertFalse(det.is_anomaly)
        self.assertEqual(det.anomaly_score, 0.15)

        # Test that unsupported capability raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            algo.diagnose_root_cause(None, None)

        with self.assertRaises(NotImplementedError):
            algo.execute_mitigation(None)

        algo.reset()
        self.assertTrue(algo.reset_called)

    def test_rca_and_mitigation_dataclasses(self):
        rca = RCADiagnosisResult(
            ranked_root_causes=["cpu_utilization", "memory_utilization"],
            confidence_scores={"cpu_utilization": 0.95},
            execution_time_ms=12.5,
        )
        self.assertEqual(len(rca.ranked_root_causes), 2)
        self.assertEqual(rca.ranked_root_causes[0], "cpu_utilization")

        mit = MitigationResult(
            action_type="MIGRATION",
            target_node="Edge_Node_2",
            tasks_affected=5,
            success=True,
        )
        self.assertEqual(mit.action_type, "MIGRATION")
        self.assertEqual(mit.tasks_affected, 5)
        self.assertTrue(mit.success)


if __name__ == "__main__":
    unittest.main()
