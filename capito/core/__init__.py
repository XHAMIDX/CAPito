"""Core module initialization."""

from .config import (
    CapitoConfig, 
    ModelPaths,
    VLMConfig,
    DetectionConfig, 
    SegmentationConfig,
    CaptioningConfig,
    SystemConfig,
    get_default_config,
    get_vlm_config,
    get_fast_config,
    get_quality_config
)
from .pipeline import CAPito

__all__ = [
    "CapitoConfig",
    "ModelPaths", 
    "VLMConfig",
    "DetectionConfig",
    "SegmentationConfig", 
    "CaptioningConfig",
    "SystemConfig",
    "get_default_config",
    "get_vlm_config",
    "get_fast_config", 
    "get_quality_config",
    "CAPito"
]
