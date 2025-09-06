"""
CAPito: Content-Aware Photo Image Text Optimizer
===============================================

Enhanced modular system for comprehensive scene understanding including:
- Object detection and segmentation
- Depth estimation and pose analysis
- Individual object captioning
- Scene graph generation
- Relationship analysis

Key Components:
- VLM: AlphaCLIP-based vision-language understanding
- Detection: YOLO-based object detection  
- Segmentation: SAM2-based mask generation
- Captioning: Controllable text generation with BERT/RoBERTa
- Analysis: Depth estimation, pose detection, object tracking
- Graph: Scene graph generation and relationship analysis

Usage:
    from capito import CAPito, get_default_config
    
    config = get_default_config()
    config.graph.enable_graph = True  # Enable scene graph generation
    capito = CAPito(config)
    
    results = capito.process_image("path/to/image.jpg")
    
    # Access enhanced results
    objects = results["enhanced_objects"]
    scene_graph = results["scene_graph"] 
    graph_analysis = results["graph_analysis"]
    scene_summary = results["scene_summary"]
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
