# Formal Results Schema and Data Representation for Dual-Track Evaluation.

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any, List
import time
import json
import pandas as pd


class MeasurementType(str, Enum):
    # Origin and nature of the recorded metric value.
    PAPER_REPORTED = "PAPER_REPORTED"
    MEASURED = "MEASURED"
    REPRODUCED = "REPRODUCED"
    ESTIMATED = "ESTIMATED"
    NR = "NR"


class Comparability(str, Enum):
    # Academic comparability status relative to the other approaches.
    DIRECT = "DIRECT"
    CONDITIONAL = "CONDITIONAL"
    NOT_COMPARABLE = "NOT_COMPARABLE"


@dataclass
class BenchmarkResultRecord:
    # Standardized result record schema.
    # Every evaluation metric emitted by any algorithm or extracted from papers
    # must strictly conform to this schema.
    algorithm: str
    paper_id: str
    metric: str
    submetric: str
    value: Optional[float]
    unit: str
    mean: Optional[float] = None
    std: Optional[float] = None
    confidence_interval: Optional[List[float]] = None  # [lower_95, upper_95]
    dataset: str = "edge_iiotset"
    dataset_split: str = "test"
    run_id: str = "run_001"
    seed: Optional[int] = 42
    measurement_type: MeasurementType = MeasurementType.MEASURED
    comparability: Comparability = Comparability.DIRECT
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    configuration: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None

    def __post_init__(self):
        # Validate measurement type
        if isinstance(self.measurement_type, str):
            self.measurement_type = MeasurementType(self.measurement_type)
        if isinstance(self.comparability, str):
            self.comparability = Comparability(self.comparability)

        # Enforce NR consistency
        if self.measurement_type == MeasurementType.NR:
            self.value = None
            self.mean = None
            self.std = None
            self.confidence_interval = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["measurement_type"] = self.measurement_type.value
        d["comparability"] = self.comparability.value
        return d


class ResultsCollection:
    # In-memory collection of BenchmarkResultRecord entries with serialization.

    def __init__(self):
        self.records: List[BenchmarkResultRecord] = []

    def add_record(self, record: BenchmarkResultRecord) -> None:
        self.records.append(record)

    def extend(self, records: List[BenchmarkResultRecord]) -> None:
        self.records.extend(records)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=[
                "algorithm", "paper_id", "metric", "submetric", "value", "unit",
                "mean", "std", "confidence_interval", "dataset", "dataset_split",
                "run_id", "seed", "measurement_type", "comparability", "timestamp",
                "configuration", "notes"
            ])
        rows = [r.to_dict() for r in self.records]
        # Format configuration as compact JSON string for CSV export
        for r in rows:
            if isinstance(r.get("configuration"), dict):
                r["configuration"] = json.dumps(r["configuration"])
            if isinstance(r.get("confidence_interval"), list):
                r["confidence_interval"] = json.dumps(r["confidence_interval"])
        return pd.DataFrame(rows)

    def save_csv(self, filepath: str) -> None:
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
