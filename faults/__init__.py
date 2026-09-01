"""
Fault injection and modeling module for Edge-IoT systems.
"""
from .fault_types import FaultType, FaultRecord
from .fault_generator import FaultGenerator

__all__ = ["FaultType", "FaultRecord", "FaultGenerator"]
