"""
CAPito Configuration Management
=============================

Centralized configuration for the CAPito project.
Manages all model paths, generation parameters, and system settings.
"""

import os
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class ModelPaths:
    """Centralized model path management."""
    
    # Root directory for all models
    models_root: str = "models"
    
    # VLM models (AlphaCLIP)
    vlm_models: Dict[str, str] = field(default_factory=lambda: {
        "ViT-B/16": "clip_b16_grit1m_fultune_8xe.pth",
        "ViT-B/32": "clip_b32_grit1m_fultune_8xe.pth",
        "ViT-L/14": "clip_l14_grit1m_fultune_8xe.pth",
        "ViT-L/14@336px": "clip_l14_336_grit1m_fultune_8xe.pth",
        "RN50": "clip_rn50_grit1m_fultune_8xe.pth"
    })
    
    # Detection models
    detection_models: Dict[str, str] = field(default_factory=lambda: {
        "yolo8n": "yolov8n.pt",
        "yolo8s": "yolov8s.pt",
        "yolo8m": "yolov8m.pt",
        "yolo8l": "yolov8l.pt"
    })
    
    # Segmentation models (SAM2)
    segmentation_models: Dict[str, str] = field(default_factory=lambda: {
        "sam2_tiny": "sam2_t.pt",
        "sam2_small": "sam2_s.pt",
        "sam2_base": "sam2_b.pt",
        "sam2_large": "sam2_l.pt"
    })
    
    # Language models (for controllable generation)
    language_models: Dict[str, str] = field(default_factory=lambda: {
        "bert_base": "bert-base-uncased",
        "bert_large": "bert-large-uncased", 
        "roberta_base": "roberta-base",
        "roberta_large": "roberta-large"
    })
    
    def __post_init__(self):
        """Ensure models directory exists."""
        os.makedirs(self.models_root, exist_ok=True)
    
    def get_model_path(self, model_type: str, model_name: str) -> str:
        """Get full path for any model."""
        model_maps = {
            "vlm": self.vlm_models,
            "detection": self.detection_models,
            "segmentation": self.segmentation_models,
            "language": self.language_models
        }
        
        if model_type not in model_maps:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_map = model_maps[model_type]
        if model_name not in model_map:
            raise ValueError(f"Unknown {model_type} model: {model_name}")
        
        # For Hugging Face models, return the model name directly
        if model_type == "language":
            return model_map[model_name]
        
        # For local models, return full path
        return os.path.join(self.models_root, model_map[model_name])


@dataclass
class VLMConfig:
    """AlphaCLIP Vision-Language Model Configuration."""
    
    model_name: str = "ViT-B/16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        valid_models = ["ViT-B/16", "ViT-B/32", "ViT-L/14", "ViT-L/14@336px", "RN50"]
        if self.model_name not in valid_models:
            raise ValueError(f"Invalid VLM model: {self.model_name}")


@dataclass
class DetectionConfig:
    """Object Detection Configuration."""
    
    model_name: str = "yolo8n"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Validate configuration."""
        valid_models = ["yolo8n", "yolo8s", "yolo8m", "yolo8l"]
        if self.model_name not in valid_models:
            raise ValueError(f"Invalid detection model: {self.model_name}")


@dataclass
class SegmentationConfig:
    """Segmentation (SAM2) Configuration."""
    
    model_name: str = "sam2_tiny"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Validate configuration."""
        valid_models = ["sam2_tiny", "sam2_small", "sam2_base", "sam2_large"]
        if self.model_name not in valid_models:
            raise ValueError(f"Invalid segmentation model: {self.model_name}")


@dataclass
class CaptioningConfig:
    """ConZIC Captioning Configuration based on the official paper."""
    
    # Generation parameters (ConZIC paper defaults)
    max_length: int = 20
    top_k: int = 300  # candidate_k in paper
    num_iterations: int = 25  # max_iters in paper
    
    # Scoring weights (ConZIC paper) - these are critical!
    alpha: float = 0.02  # Fluency weight (BERT) - very low as in paper
    beta: float = 2.0    # Image-text matching weight (CLIP) - high as in paper
    gamma: float = 5.0   # Controllable weight (sentiment/POS)
    
    # Generation settings (ConZIC paper)
    temperature: float = 0.1  # lm_temperature in paper - very low
    repetition_penalty: float = 1.2  # gentle repetition penalty
    generation_order: str = "sequential"  # order in paper: sequential, shuffle, random, span
    prompt_template: str = "A photo of"
    
    # Language model for controllable generation
    language_model: str = "bert_base"
    
    # Control settings
    enable_control: bool = False
    control_type: str = "sentiment"  # sentiment, pos
    sentiment_type: str = "positive"  # positive, negative
    
    def __post_init__(self):
        """Validate configuration."""
        valid_orders = ["span", "shuffle", "sequential", "random"]
        if self.generation_order not in valid_orders:
            raise ValueError(f"Invalid generation order: {self.generation_order}")
        
        valid_controls = ["sentiment", "pos"]
        if self.control_type not in valid_controls:
            raise ValueError(f"Invalid control type: {self.control_type}")


@dataclass
class SystemConfig:
    """System-wide Configuration."""
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    seed: int = 42
    log_level: str = "INFO"
    
    # Output settings
    output_dir: str = "outputs"
    save_visualizations: bool = True
    save_results: bool = True
    
    def __post_init__(self):
        """Ensure output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class CapitoConfig:
    """Main CAPito Configuration."""
    
    model_paths: ModelPaths = field(default_factory=ModelPaths)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    captioning: CaptioningConfig = field(default_factory=CaptioningConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def get_model_path(self, model_type: str, model_name: str) -> str:
        """Convenience method to get model paths."""
        return self.model_paths.get_model_path(model_type, model_name)
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'CapitoConfig':
        """Load configuration from YAML/JSON file."""
        # Implementation would load from file
        # For now, return default config
        return cls()
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to file."""
        # Implementation would save to file
        pass


def get_default_config() -> CapitoConfig:
    """Get default CAPito configuration."""
    return CapitoConfig()


def get_vlm_config() -> CapitoConfig:
    """Get configuration optimized for VLM tasks."""
    config = CapitoConfig()
    config.vlm.model_name = "ViT-L/14"
    config.captioning.beta = 2.5  # Higher weight for image-text matching
    return config


def get_fast_config() -> CapitoConfig:
    """Get configuration optimized for speed."""
    config = CapitoConfig()
    config.vlm.model_name = "ViT-B/32"
    config.detection.model_name = "yolo8n"
    config.segmentation.model_name = "sam2_tiny"
    config.captioning.num_iterations = 5
    config.captioning.top_k = 50  # Use fewer candidates for speed
    return config


def get_quality_config() -> CapitoConfig:
    """Get configuration optimized for quality."""
    config = CapitoConfig()
    config.vlm.model_name = "ViT-L/14@336px"
    config.detection.model_name = "yolo8l"
    config.segmentation.model_name = "sam2_large"
    config.captioning.num_iterations = 50  # More iterations for quality
    config.captioning.top_k = 500  # More candidates for better selection
    config.captioning.language_model = "bert_large"
    return config
