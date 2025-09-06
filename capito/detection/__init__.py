"""Detection module initialization."""

from .detector import ObjectDetector, Detection
from .segmentation import SAM2Segmentator, Mask

__all__ = ["ObjectDetector", "Detection", "SAM2Segmentator", "Mask"]
