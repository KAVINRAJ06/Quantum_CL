"""SAM + PennyLane continual semantic segmentation."""

from .model import QuantumSAMSegmenter
from .continual import EWC, ReplayBuffer

__all__ = ["QuantumSAMSegmenter", "EWC", "ReplayBuffer"]
