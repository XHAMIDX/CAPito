"""
CAPito: Content-Aware Photo Image Text Optimizer
===============================================

A high-level, modular system for object detection, segmentation, and 
intelligent captioning using Vision-Language Models.

Key Components:
- VLM: AlphaCLIP-based vision-language understanding
- Detection: YOLO-based object detection  
- Segmentation: SAM2-based mask generation
- Captioning: Controllable text generation with BERT/RoBERTa

Usage:
    from capito import CAPito, get_default_config
    
    config = get_default_config()
    capito = CAPito(config)
    
    results = capito.process_image("path/to/image.jpg")
"""

from .core.config import (
    CapitoConfig,
    get_default_config,
    get_vlm_config, 
    get_fast_config,
    get_quality_config
)
from .core.pipeline import CAPito
from .vlm.alpha_clip import AlphaCLIPWrapper
from .detection.detector import ObjectDetector
from .detection.segmentation import SAM2Segmentator
from .captioning.generator import CaptionGenerator

__version__ = "2.0.0"
__author__ = "CAPito Development Team"

__all__ = [
    # Main classes
    "CAPito",
    "AlphaCLIPWrapper", 
    "ObjectDetector",
    "SAM2Segmentator",
    "CaptionGenerator",
    
    # Configuration
    "CapitoConfig",
    "get_default_config",
    "get_vlm_config",
    "get_fast_config", 
    "get_quality_config",
]
