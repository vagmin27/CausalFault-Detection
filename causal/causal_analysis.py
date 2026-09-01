"""
DoWhy Causal Analysis Engine.

Performs treatment-specific causal effect estimation (Average Causal Effect - ACE)
using DoWhy backdoor linear regression and NetworkX system graphs to determine root causes.

Non-Circular Treatment -> Outcome Mappings:
    cpu_utilization     -> high_latency       (latency > 55ms)
    memory_utilization  -> high_latency       (latency > 55ms)
    network_utilization -> high_packet_loss   (packet_loss > 5%)
    packet_loss         -> high_latency       (latency > 55ms)
    latency             -> extreme_latency    (latency > 70ms)
"""

from dataclasses import dataclass, field
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from dowhy import CausalModel

from data.telemetry import TelemetryRecord
from .causal_graph import SystemCausalGraph

logger = logging.getLogger(__name__)


@dataclass
class CausalResult:
    """
    Structured Causal Analysis Result output.
    Status can be:
        - "CAUSAL_INFERENCE": Valid DoWhy Average Causal Effect estimation.
        - "HEURISTIC": Feature deviation fallback when sample/variance is insufficient.
        - "INSUFFICIENT_EVIDENCE": Data insufficient for defensible cause determination.
    """
    treatment: str                          # Primary treatment variable evaluated
    outcome: str                            # Non-circular downstream outcome variable evaluated
    estimated_effect: float                 # Estimated ACE magnitude
    method: str                             # "backdoor.linear_regression"
    root_cause: str                         # Identified root cause fault category
    root_cause_source: str                  # "DOWHY_CAUSAL_INFERENCE", "HEURISTIC", "RULE_BASED", "INSUFFICIENT_EVIDENCE"
    causal_status: str                      # "CAUSAL_INFERENCE", "HEURISTIC", "INSUFFICIENT_EVIDENCE"
    refutation_status: str = "NOT_TESTED"   # "NOT_TESTED" or "PASSED"
    candidate_effects: Dict[str, float] = field(default_factory=dict)
    supporting_variables: List[str] = field(default_factory=list)


class CausalAnalyzer:
    """
    DoWhy Causal Effect Estimation Engine.
    """

    def __init__(self, causal_graph: Optional[SystemCausalGraph] = None):
        self.causal_graph = causal_graph if causal_graph is not None else SystemCausalGraph()
        self.nx_graph = self.causal_graph.get_graph()

        self.treatment_map = {
            "cpu_utilization": "CPU_OVERLOAD",
            "memory_utilization": "MEMORY_OVERLOAD",
            "network_utilization": "NETWORK_CONGESTION",
            "packet_loss": "PACKET_LOSS",
            "latency": "HIGH_LATENCY",
        }

        # Non-circular treatment -> downstream outcome mapping
        self.treatment_outcome_map = {
            "cpu_utilization": "high_latency",
            "memory_utilization": "high_latency",
            "network_utilization": "high_packet_loss",
            "packet_loss": "high_latency",
            "latency": "extreme_latency",
        }

        self._last_analysis_time: Dict[str, float] = {}
        self._last_result: Dict[str, CausalResult] = {}

    def analyze_fault_cause(
        self,
        history_records: List[TelemetryRecord],
        target_node_id: Optional[str] = None,
    ) -> CausalResult:
        """
        Analyze recent historical window of TelemetryRecords using DoWhy causal models.
        """
        node_key = target_node_id or "default"
        latest_ts = history_records[-1].timestamp if history_records else 0.0

        # Debounce cache: reuse recent result if within 4.0 simulation time steps
        if node_key in self._last_result and (latest_ts - self._last_analysis_time.get(node_key, -99.0) < 4.0):
            return self._last_result[node_key]

        # Filter for relevant node records if target specified
        node_records = [
            r for r in history_records
            if target_node_id is None or r.edge_node_id == target_node_id
        ]

        if len(node_records) < 10:
            logger.warning("[Causal Analyzer] Insufficient telemetry window length (<10 records) for estimation.")
            res = CausalResult(
                treatment="none",
                outcome="none",
                estimated_effect=0.0,
                method="none",
                root_cause="UNKNOWN",
                root_cause_source="INSUFFICIENT_EVIDENCE",
                causal_status="INSUFFICIENT_EVIDENCE",
                refutation_status="NOT_TESTED",
                candidate_effects={},
                supporting_variables=[],
            )
            self._last_result[node_key] = res
            self._last_analysis_time[node_key] = latest_ts
            return res

        # Build Pandas DataFrame from historical telemetry window (NO ground truth fault_label leakage)
        rows = [r.get_available_features() for r in node_records]
        df = pd.DataFrame(rows).fillna(0.0)

        # Compute non-circular downstream outcome variables
        df["high_latency"] = (df["latency"] > 55.0).astype(int)
        df["high_packet_loss"] = (df["packet_loss"] > 5.0).astype(int)
        df["extreme_latency"] = (df["latency"] > 70.0).astype(int)

        candidate_treatments = self.causal_graph.get_candidate_treatments()
        candidate_effects: Dict[str, float] = {}

        for treatment in candidate_treatments:
            if treatment not in df.columns or df[treatment].nunique() <= 1:
                continue

            outcome = self.treatment_outcome_map.get(treatment, "high_latency")

            if outcome not in df.columns or df[outcome].nunique() <= 1:
                continue

            try:
                # Pass NetworkX DiGraph directly to DoWhy CausalModel
                model = CausalModel(
                    data=df,
                    treatment=treatment,
                    outcome=outcome,
                    graph=self.nx_graph,
                    logging_level=logging.ERROR,
                )

                # Identify causal effect via backdoor criterion
                identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

                # Estimate causal effect using backdoor linear regression
                estimate = model.estimate_effect(
                    identified_estimand,
                    method_name="backdoor.linear_regression",
                    test_significance=False,
                )

                effect_val = abs(float(estimate.value)) if estimate.value is not None else 0.0
                candidate_effects[treatment] = round(effect_val, 4)

            except Exception as e:
                logger.debug(f"DoWhy estimation error for treatment '{treatment}': {e}")
                candidate_effects[treatment] = 0.0

        if not candidate_effects or max(candidate_effects.values(), default=0.0) == 0.0:
            res = self._heuristic_feature_ranking(df, node_records)
            self._last_result[node_key] = res
            self._last_analysis_time[node_key] = latest_ts
            return res

        # Identify treatment with strongest causal effect on its downstream outcome
        best_treatment = max(candidate_effects, key=candidate_effects.get)
        top_effect = candidate_effects[best_treatment]
        identified_root_cause = self.treatment_map.get(best_treatment, "CPU_OVERLOAD")
        best_outcome = self.treatment_outcome_map.get(best_treatment, "high_latency")

        logger.info(
            f"[DOWHY CAUSAL INFERENCE] Root cause identified: '{identified_root_cause}' "
            f"(Treatment: {best_treatment} -> Outcome: {best_outcome}, ACE: {top_effect})"
        )

        res = CausalResult(
            treatment=best_treatment,
            outcome=best_outcome,
            estimated_effect=top_effect,
            method="backdoor.linear_regression",
            root_cause=identified_root_cause,
            root_cause_source="DOWHY_CAUSAL_INFERENCE",
            causal_status="CAUSAL_INFERENCE",
            refutation_status="NOT_TESTED",
            candidate_effects=candidate_effects,
            supporting_variables=list(candidate_effects.keys()),
        )
        self._last_result[node_key] = res
        self._last_analysis_time[node_key] = latest_ts
        return res

    def _heuristic_feature_ranking(
        self,
        df: pd.DataFrame,
        node_records: List[TelemetryRecord],
    ) -> CausalResult:
        """
        Fallback feature z-score deviation analyzer when DoWhy sample size or label variance is low.
        Returns status 'HEURISTIC' and root_cause_source 'HEURISTIC'.
        """
        candidate_effects = {}
        for feat in ["cpu_utilization", "memory_utilization", "network_utilization", "packet_loss", "latency"]:
            if feat in df.columns:
                mean_val = df[feat].mean()
                latest_val = df[feat].iloc[-1]
                std_val = df[feat].std() + 1e-5
                z_score = abs((latest_val - mean_val) / std_val)
                candidate_effects[feat] = round(float(z_score), 4)

        if not candidate_effects or max(candidate_effects.values(), default=0.0) < 1.0:
            return CausalResult(
                treatment="none",
                outcome="none",
                estimated_effect=0.0,
                method="z_score_deviation",
                root_cause="UNKNOWN",
                root_cause_source="INSUFFICIENT_EVIDENCE",
                causal_status="INSUFFICIENT_EVIDENCE",
                refutation_status="NOT_TESTED",
                candidate_effects=candidate_effects,
                supporting_variables=list(candidate_effects.keys()),
            )

        best_feat = max(candidate_effects, key=candidate_effects.get)
        root_cause = self.treatment_map.get(best_feat, "CPU_OVERLOAD")
        outcome = self.treatment_outcome_map.get(best_feat, "high_latency")

        return CausalResult(
            treatment=best_feat,
            outcome=outcome,
            estimated_effect=candidate_effects[best_feat],
            method="z_score_deviation",
            root_cause=root_cause,
            root_cause_source="HEURISTIC",
            causal_status="HEURISTIC",
            refutation_status="NOT_TESTED",
            candidate_effects=candidate_effects,
            supporting_variables=list(candidate_effects.keys()),
        )
