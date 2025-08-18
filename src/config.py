"""Configuration management for GET_CAPTION project."""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ModelPathsConfig:
    """Centralized model paths configuration."""
    # Base paths
    models_root: str = "Model/"
    
    # AlphaCLIP model paths
    alpha_clip_root: str = "Model/"
    alpha_clip_checkpoints: str = "Model/"
    
    # Detection model paths
    detection_root: str = "Model/"
    yolo_models: str = "Model/"
    sam_models: str = "Model/"
    
    # Language model paths
    language_models_root: str = "Model/"
    
    # Legacy model paths
    legacy_root: str = "Model/"
    
    def __post_init__(self):
        """Create all necessary directories."""
        # Ensure the unified Model directory exists
        os.makedirs(self.models_root, exist_ok=True)
    
    def get_alpha_clip_path(self, model_name: str) -> str:
        """Get full path for AlphaCLIP model."""
        # Map model names to checkpoint files
        model_mapping = {
            "ViT-B/32": "clip_b32_grit1m_fultune_8xe.pth",
            "ViT-B/16": "clip_b16_grit1m_fultune_8xe.pth", 
            "ViT-L/14": "clip_l14_grit1m_fultune_8xe.pth",
            "ViT-L/14@336px": "clip_l14_336_grit1m_fultune_8xe.pth",
            "RN50": "clip_rn50_grit1m_fultune_8xe.pth"
        }
        
        checkpoint_file = model_mapping.get(model_name, "clip_b16_grit1m_fultune_8xe.pth")
        return os.path.join(self.models_root, checkpoint_file)
    
    def get_detection_model_path(self, model_name: str) -> str:
        """Get full path for detection model."""
        # All models are centralized under the unified Model directory
        return os.path.join(self.models_root, model_name)


@dataclass
class ModelConfig:
    """Model configuration settings."""
    # AlphaCLIP settings
    alpha_clip_model: str = "ViT-B/16"  # ViT-B/32, ViT-B/16, ViT-L/14, RN50
    
    # Language model settings
    lm_model: str = "bert-base-uncased"  # or roberta-base
    
    # Object detection settings
    detection_model: str = "yolov8n.pt"  # yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt
    detection_conf: float = 0.25
    detection_iou: float = 0.45
    
    # SAM2 settings
    sam_model: str = "sam2_t.pt"  # sam2_t.pt (tiny), sam2_s.pt (small), sam2_b.pt (base), sam2_l.pt (large)
    TORCH_USE_CUDA_DSA = True
    # Device settings
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


@dataclass
class GenerationConfig:
    """Text generation configuration."""
    # Generation parameters
    sentence_len: int = 6  # Reduced from 8 for more concise captions
    candidate_k: int = 30  # Reduced from 50 for better quality
    num_iterations: int = 10  # Reduced from 15 for faster generation
    
    # Scoring weights - better balance for image relevance
    alpha: float = 0.4  # Reduced weight for fluency (BERT quality) from 0.8
    beta: float = 2.0   # Increased weight for image-matching degree (CLIP) from 1.5
    gamma: float = 0.3  # weight for controllable degree (sentiment/POS)
    
    # Temperature and sampling
    lm_temperature: float = 0.2  # Reduced from 0.3 for more focused generation
    
    # Generation order
    order: str = "span"  # Changed from "shuffle" to "span" for better coherence
    
    # Prompt template
    prompt: str = "A photo of"  # Simple, clean prompt
    
    # Control settings
    run_type: str = "caption"  # caption, controllable
    control_type: str = "sentiment"  # sentiment, pos
    sentiment_type: str = "positive"  # positive, negative
    pos_type: List[List[str]] = field(default_factory=lambda: [
        ['DET'], ['ADJ', 'NOUN'], ['NOUN'], ['VERB'], ['ADV'], ['ADP'], 
        ['DET', 'NOUN'], ['NOUN'], ['VERB'], ['ADP'], ['DET', 'NOUN']
    ])


@dataclass
class ProcessingConfig:
    """Image processing configuration."""
    # Input/Output paths
    input_path: str = "examples/"
    output_path: str = "results/"
    
    # Processing settings
    batch_size: int = 1
    samples_num: int = 3
    
    # Object detection filtering
    min_object_area: int = 1000  # minimum pixel area for objects
    max_objects_per_image: int = 10
    
    # Mask processing
    mask_blur_radius: int = 2
    mask_threshold: float = 0.5


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    model_paths: ModelPathsConfig = field(default_factory=ModelPathsConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    # Logging
    log_level: str = "INFO"
    save_intermediate: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure output directory exists
        os.makedirs(self.processing.output_path, exist_ok=True)
        os.makedirs("logs", exist_ok=True)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or return default."""
    if config_path and os.path.exists(config_path):
        # TODO: Implement YAML/JSON config loading
        pass
    return get_default_config()
