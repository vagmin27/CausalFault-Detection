"""
Experiment Reproducibility and Environment Profiling Module.

Captures immutable hardware, software, Git, and package metadata to ensure
full empirical reproducibility of all benchmark results.
"""

import os
import sys
import platform
import subprocess
import time
import json
from typing import Dict, Any, Optional

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def get_git_commit_hash() -> str:
    """Retrieves current Git commit SHA-1 if available."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2.0
        ).decode("utf-8").strip()
        return out
    except Exception:
        return "GIT_NOT_AVAILABLE"


def get_git_status_clean() -> bool:
    """Checks if git repository working tree is clean."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            timeout=2.0
        ).decode("utf-8").strip()
        return len(out) == 0
    except Exception:
        return False


def get_installed_package_versions() -> Dict[str, str]:
    """Records exact installed versions of key scientific and system packages."""
    packages = [
        "numpy", "pandas", "scipy", "scikit-learn", "torch",
        "river", "networkx", "dowhy", "psutil", "joblib", "matplotlib"
    ]
    versions = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "UNKNOWN")
        except ImportError:
            versions[pkg] = "NOT_INSTALLED"
    return versions


def capture_reproducibility_environment(
    experiment_name: str = "CommonBenchmark",
    config: Optional[Any] = None,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Constructs a comprehensive reproducibility record.
    Saved alongside raw benchmark results.
    """
    env = {
        "experiment_name": experiment_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "architecture": platform.architecture()[0],
        },
        "hardware": {
            "processor": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False) if _HAS_PSUTIL else (os.cpu_count() or 1),
            "logical_cores": psutil.cpu_count(logical=True) if _HAS_PSUTIL else (os.cpu_count() or 1),
            "cpu_freq_max_mhz": (psutil.cpu_freq().max if (_HAS_PSUTIL and psutil.cpu_freq()) else 0.0),
            "total_ram_bytes": psutil.virtual_memory().total if _HAS_PSUTIL else 0,
            "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2) if _HAS_PSUTIL else 8.0,
        },
        "git": {
            "commit_hash": get_git_commit_hash(),
            "working_tree_clean": get_git_status_clean(),
        },
        "installed_packages": get_installed_package_versions(),
        "random_seed": seed,
        "configuration": config.to_dict() if hasattr(config, "to_dict") else (config or {}),
    }
    return env


def save_reproducibility_record(record: Dict[str, Any], output_path: str = "results/raw/environment.json") -> None:
    """Persists environment record to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2)
