"""
Standardized experiment configuration for common benchmark evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os
import platform
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


@dataclass
class HardwareProfile:
    """Hardware and OS baseline profile of the execution environment."""
    system: str = field(default_factory=platform.system)
    release: str = field(default_factory=platform.release)
    processor: str = field(default_factory=platform.processor)
    cpu_cores_physical: int = field(default_factory=lambda: (psutil.cpu_count(logical=False) if _HAS_PSUTIL else os.cpu_count()) or 1)
    cpu_cores_logical: int = field(default_factory=lambda: (psutil.cpu_count(logical=True) if _HAS_PSUTIL else os.cpu_count()) or 1)
    cpu_freq_max_mhz: float = field(default_factory=lambda: (psutil.cpu_freq().max if (_HAS_PSUTIL and psutil.cpu_freq()) else 0.0))
    ram_total_gb: float = field(default_factory=lambda: (round(psutil.virtual_memory().total / (1024 ** 3), 2) if _HAS_PSUTIL else 8.0))
    python_version: str = field(default_factory=platform.python_version)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system": self.system,
            "release": self.release,
            "processor": self.processor,
            "cpu_cores_physical": self.cpu_cores_physical,
            "cpu_cores_logical": self.cpu_cores_logical,
            "cpu_freq_max_mhz": self.cpu_freq_max_mhz,
            "ram_total_gb": self.ram_total_gb,
            "python_version": self.python_version,
        }


@dataclass
class ExperimentConfig:
    """Standardized configuration ensuring identical conditions across algorithms."""
    # Dataset configuration
    dataset_name: str = "edge_iiotset"
    data_dir: str = "data/processed"
    dataset_split: str = "test"
    max_records: Optional[int] = None
    warmup_records: int = 1000

    # Repeatability & Seeds
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    current_seed: int = 42
    run_id: str = "run_001"

    # Streaming & Buffer dimensions
    stream_chunk_size: int = 5000
    sliding_window_size: int = 60
    batch_size: int = 64

    # Scalability parameters
    dimension_scales: List[int] = field(default_factory=lambda: [10, 25, 50, 62])
    node_scales: List[int] = field(default_factory=lambda: [5, 10, 20, 50])

    # SLO / SLA threshold for latency breaches
    slo_latency_threshold_ms: float = 100.0

    # Power model coefficients (SPECpower / Edge reference model for Estimated Energy)
    power_model_idle_watts: float = 2.7   # Raspberry Pi 4B idle baseline
    power_model_peak_watts: float = 6.4   # Raspberry Pi 4B peak baseline

    # Output directory layout
    output_dir: str = "results"

    # System profile
    hardware: HardwareProfile = field(default_factory=HardwareProfile)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "data_dir": self.data_dir,
            "dataset_split": self.dataset_split,
            "max_records": self.max_records,
            "warmup_records": self.warmup_records,
            "seeds": self.seeds,
            "current_seed": self.current_seed,
            "run_id": self.run_id,
            "stream_chunk_size": self.stream_chunk_size,
            "sliding_window_size": self.sliding_window_size,
            "batch_size": self.batch_size,
            "dimension_scales": self.dimension_scales,
            "node_scales": self.node_scales,
            "slo_latency_threshold_ms": self.slo_latency_threshold_ms,
            "power_model_idle_watts": self.power_model_idle_watts,
            "power_model_peak_watts": self.power_model_peak_watts,
            "output_dir": self.output_dir,
            "hardware": self.hardware.to_dict(),
        }
