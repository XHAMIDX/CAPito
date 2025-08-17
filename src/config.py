"""Configuration management for GET_CAPTION project."""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ModelPathsConfig:
    """Centralized model paths configuration."""
    # Base paths
    models_root: str = "models/"
    
    # AlphaCLIP model paths
    alpha_clip_root: str = "models/alpha_clip/"
    alpha_clip_checkpoints: str = "models/alpha_clip/checkpoints/"
    
    # Detection model paths
    detection_root: str = "models/detection/"
    yolo_models: str = "models/detection/yolo/"
    sam_models: str = "models/detection/sam/"
    
    # Language model paths
    language_models_root: str = "models/language/"
    
    # Legacy model paths
    legacy_root: str = "models/legacy/"
    
    def __post_init__(self):
        """Create all necessary directories."""
        for path in [self.models_root, self.alpha_clip_root, self.alpha_clip_checkpoints,
                    self.detection_root, self.yolo_models, self.sam_models,
                    self.language_models_root, self.legacy_root]:
            os.makedirs(path, exist_ok=True)
    
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
        return os.path.join(self.alpha_clip_checkpoints, checkpoint_file)
    
    def get_detection_model_path(self, model_name: str) -> str:
        """Get full path for detection model."""
        if model_name.startswith("yolo"):
            return os.path.join(self.yolo_models, model_name)
        elif model_name.startswith("sam"):
            return os.path.join(self.sam_models, model_name)
        else:
            return model_name  # Return as-is for ultralytics auto-download


@dataclass
class ModelConfig:
    """Model configuration settings."""
    # AlphaCLIP settings
    alpha_clip_model: str = "ViT-L/14"  # ViT-B/32, ViT-B/16, ViT-L/14, RN50
    
    # Language model settings
    lm_model: str = "bert-base-uncased"  # or roberta-base
    
    # Object detection settings
    detection_model: str = "yolov8n.pt"  # yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt
    detection_conf: float = 0.25
    detection_iou: float = 0.45
    
    # SAM2 settings
    sam_model: str = "sam2_t.pt"  # sam2_t.pt (tiny), sam2_s.pt (small), sam2_b.pt (base), sam2_l.pt (large)
    
    # Device settings
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


@dataclass
class GenerationConfig:
    """Text generation configuration."""
    # Generation parameters
    sentence_len: int = 8
    candidate_k: int = 50
    num_iterations: int = 15
    
    # Scoring weights
    alpha: float = 0.8  # weight for fluency (BERT quality)
    beta: float = 1.5   # weight for image-matching degree (CLIP)
    gamma: float = 0.3  # weight for controllable degree (sentiment/POS)
    
    # Temperature and sampling
    lm_temperature: float = 0.3
    
    # Generation order
    order: str = "shuffle"  # sequential, shuffle, span, random
    
    # Prompt settings
    prompt: str = "A detailed image showing "
    
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
