"""Object detection and segmentation modules."""

from .detector import ObjectDetector
from .segmentation import MaskGenerator
from .pipeline import DetectionPipeline

__all__ = ['ObjectDetector', 'MaskGenerator', 'DetectionPipeline']
