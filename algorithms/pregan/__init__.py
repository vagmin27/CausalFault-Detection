"""
PreGAN Package (Tuli et al., 2022).
"""

from .algorithm import PreGANAlgorithm, PreGANConfig
from .model import PreGANGenerator, PreGANDiscriminator

__all__ = ["PreGANAlgorithm", "PreGANConfig", "PreGANGenerator", "PreGANDiscriminator"]
