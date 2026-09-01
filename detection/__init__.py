"""
Detection package providing real-time River streaming and PyTorch Autoencoder detectors.
"""
from .river_detector import RiverFaultDetector
from .pytorch_detector import PyTorchFaultDetector

__all__ = ["RiverFaultDetector", "PyTorchFaultDetector"]
