"""Model management utilities for GET_CAPTION project."""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
from tqdm import tqdm

try:
    from ..config import ModelPathsConfig
except ImportError:
    from config import ModelPathsConfig


class ModelManager:
    """Centralized model management for GET_CAPTION project."""
    
    def __init__(self, model_paths: Optional[ModelPathsConfig] = None):
        """Initialize model manager.
        
        Args:
            model_paths: Model paths configuration. If None, creates default.
        """
        self.model_paths = model_paths or ModelPathsConfig()
        self.logger = logging.getLogger(__name__)
        
        # Model download URLs
        self.model_urls = {
            # YOLOv8 models
            "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
            "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
            "yolov8l.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
            "yolov8x.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",
            
            # SAM2 models
            "sam2_t.pt": "https://github.com/ultralytics/assets/releases/download/v8.1.0/sam2_t.pt",
            "sam2_s.pt": "https://github.com/ultralytics/assets/releases/download/v8.1.0/sam2_s.pt",
            "sam2_b.pt": "https://github.com/ultralytics/assets/releases/download/v8.1.0/sam2_b.pt",
            "sam2_l.pt": "https://github.com/ultralytics/assets/releases/download/v8.1.0/sam2_l.pt",
            
            # AlphaCLIP models
            "clip_b16_grit1m_fultune_8xe.pth": "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_b16_grit1m_fultune_8xe.pth",
            "clip_l14_grit1m_fultune_8xe.pth": "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_grit1m_fultune_8xe.pth",
            "clip_l14_336_grit1m_fultune_8xe.pth": "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_336_grit1m_fultune_8xe.pth",
            "clip_b16_grit20m_fultune_2xe.pth": "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_b16_grit20m_fultune_2xe.pth",
            "clip_l14_grit20m_fultune_2xe.pth": "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_grit20m_fultune_2xe.pth",
            "clip_l14_336_grit20m_fultune_2xe.pth": "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_336_grit20m_fultune_2xe.pth"
        }
    
    def organize_existing_models(self) -> Dict[str, List[str]]:
        """Organize existing models into the new structure.
        
        Returns:
            Dictionary mapping source locations to moved files
        """
        moved_files = {}
        
        # Move AlphaCLIP checkpoints
        alpha_clip_source = Path("AlphaCLIP/checkpoints")
        if alpha_clip_source.exists():
            moved_files["AlphaCLIP"] = []
            for checkpoint_file in alpha_clip_source.glob("*.pth"):
                dest_path = self.model_paths.alpha_clip_checkpoints / checkpoint_file.name
                if not dest_path.exists():
                    shutil.copy2(checkpoint_file, dest_path)
                    moved_files["AlphaCLIP"].append(str(dest_path))
                    self.logger.info(f"Copied AlphaCLIP checkpoint: {checkpoint_file.name}")
        
        # Move legacy models if they exist
        legacy_source = Path("legacy/ConZIC")
        if legacy_source.exists():
            moved_files["Legacy"] = []
            for model_file in legacy_source.rglob("*.pth"):
                if model_file.is_file():
                    dest_path = self.model_paths.legacy_root / model_file.name
                    if not dest_path.exists():
                        shutil.copy2(model_file, dest_path)
                        moved_files["Legacy"].append(str(dest_path))
                        self.logger.info(f"Copied legacy model: {model_file.name}")
        
        return moved_files
    
    def download_model(self, model_name: str, force: bool = False) -> str:
        """Download a model if it doesn't exist.
        
        Args:
            model_name: Name of the model to download
            force: Force re-download even if file exists
            
        Returns:
            Path to the downloaded model
        """
        if model_name not in self.model_urls:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Determine destination path
        if model_name.startswith("yolo"):
            dest_path = Path(self.model_paths.yolo_models) / model_name
        elif model_name.startswith("sam"):
            dest_path = Path(self.model_paths.sam_models) / model_name
        elif model_name.startswith("clip"):
            dest_path = Path(self.model_paths.alpha_clip_checkpoints) / model_name
        else:
            dest_path = Path(self.model_paths.models_root) / model_name
        
        # Check if already exists
        if dest_path.exists() and not force:
            self.logger.info(f"Model already exists: {dest_path}")
            return str(dest_path)
        
        # Download model
        url = self.model_urls[model_name]
        self.logger.info(f"Downloading {model_name} from {url}")
        
        try:
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc=model_name) as pbar:
                def progress_hook(block_num, block_size, total_size):
                    pbar.total = total_size
                    pbar.update(block_size)
                
                urllib.request.urlretrieve(url, dest_path, progress_hook)
            
            self.logger.info(f"Successfully downloaded {model_name} to {dest_path}")
            return str(dest_path)
            
        except Exception as e:
            self.logger.error(f"Failed to download {model_name}: {e}")
            if dest_path.exists():
                dest_path.unlink()  # Remove partial download
            raise
    
    def download_all_models(self, force: bool = False) -> Dict[str, str]:
        """Download all available models.
        
        Args:
            force: Force re-download of existing models
            
        Returns:
            Dictionary mapping model names to their paths
        """
        downloaded_models = {}
        
        for model_name in self.model_urls.keys():
            try:
                path = self.download_model(model_name, force=force)
                downloaded_models[model_name] = path
            except Exception as e:
                self.logger.warning(f"Failed to download {model_name}: {e}")
        
        return downloaded_models
    
    def get_model_path(self, model_name: str) -> str:
        """Get the full path to a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Full path to the model file
        """
        # Check if model exists in organized structure
        if model_name.startswith("yolo"):
            path = Path(self.model_paths.yolo_models) / model_name
        elif model_name.startswith("sam"):
            path = Path(self.model_paths.sam_models) / model_name
        elif model_name.startswith("clip"):
            path = Path(self.model_paths.alpha_clip_checkpoints) / model_name
        else:
            # For other models, check in root models directory
            path = Path(self.model_paths.models_root) / model_name
        
        if path.exists():
            return str(path)
        
        # If not found, return the model name for ultralytics/transformers auto-download
        return model_name
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models in the organized structure.
        
        Returns:
            Dictionary mapping model categories to available models
        """
        available = {
            "AlphaCLIP": [],
            "YOLO": [],
            "SAM2": [],
            "Legacy": [],
            "Language": []
        }
        
        # Check AlphaCLIP models
        alpha_clip_path = Path(self.model_paths.alpha_clip_checkpoints)
        if alpha_clip_path.exists():
            available["AlphaCLIP"] = [f.name for f in alpha_clip_path.glob("*.pth")]
        
        # Check YOLO models
        yolo_path = Path(self.model_paths.yolo_models)
        if yolo_path.exists():
            available["YOLO"] = [f.name for f in yolo_path.glob("*.pt")]
        
        # Check SAM2 models
        sam_path = Path(self.model_paths.sam_models)
        if sam_path.exists():
            available["SAM2"] = [f.name for f in sam_path.glob("*.pt")]
        
        # Check legacy models
        legacy_path = Path(self.model_paths.legacy_root)
        if legacy_path.exists():
            available["Legacy"] = [f.name for f in legacy_path.glob("*.pth")]
        
        return available
    
    def cleanup_old_structure(self) -> List[str]:
        """Clean up old model structure after migration.
        
        Returns:
            List of cleaned up paths
        """
        cleaned_paths = []
        
        # Remove old AlphaCLIP checkpoints if they exist in new location
        old_alpha_clip = Path("AlphaCLIP/checkpoints")
        if old_alpha_clip.exists():
            new_alpha_clip = Path(self.model_paths.alpha_clip_checkpoints)
            if new_alpha_clip.exists() and any(new_alpha_clip.glob("*.pth")):
                # Only remove if new location has models
                try:
                    shutil.rmtree(old_alpha_clip)
                    cleaned_paths.append(str(old_alpha_clip))
                    self.logger.info(f"Cleaned up old AlphaCLIP checkpoints: {old_alpha_clip}")
                except Exception as e:
                    self.logger.warning(f"Could not clean up {old_alpha_clip}: {e}")
        
        return cleaned_paths


def setup_model_environment() -> ModelManager:
    """Set up the model environment and return manager instance.
    
    Returns:
        Configured ModelManager instance
    """
    model_paths = ModelPathsConfig()
    manager = ModelManager(model_paths)
    
    # Organize existing models
    moved_files = manager.organize_existing_models()
    
    # Log what was moved
    for source, files in moved_files.items():
        if files:
            logging.info(f"Moved {len(files)} models from {source}")
    
    return manager
