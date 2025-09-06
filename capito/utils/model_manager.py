"""
Model Management Utilities
==========================

Utilities for managing model downloads, loading, and caching.
"""

import os
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, Union
from urllib.parse import urlparse
import requests
from tqdm import tqdm


logger = logging.getLogger(__name__)


class ModelManager:
    """
    Centralized model management for CAPito.
    
    Handles:
    - Model downloading
    - Model caching
    - Model verification
    - Model organization
    """
    
    def __init__(self, models_root: str = "models"):
        """
        Initialize model manager.
        
        Args:
            models_root: Root directory for all models
        """
        self.models_root = Path(models_root)
        self.models_root.mkdir(parents=True, exist_ok=True)
        
        # Model registry with download URLs and checksums
        self.model_registry = {
            # AlphaCLIP models
            "clip_b16_grit1m_fultune_8xe.pth": {
                "url": "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_b16_grit1m_fultune_8xe.pth",
                "checksum": None,  # Add actual checksums
                "type": "vlm"
            },
            "clip_b32_grit1m_fultune_8xe.pth": {
                "url": "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_b32_grit1m_fultune_8xe.pth",
                "checksum": None,
                "type": "vlm"
            },
            "clip_l14_grit1m_fultune_8xe.pth": {
                "url": "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_grit1m_fultune_8xe.pth",
                "checksum": None,
                "type": "vlm"
            },
            
            # YOLO models
            "yolov8n.pt": {
                "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
                "checksum": None,
                "type": "detection"
            },
            "yolov8s.pt": {
                "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
                "checksum": None,
                "type": "detection"
            },
            "yolov8m.pt": {
                "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
                "checksum": None,
                "type": "detection"
            },
            "yolov8l.pt": {
                "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
                "checksum": None,
                "type": "detection"
            },
            
            # SAM2 models
            "sam2_t.pt": {
                "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
                "checksum": None,
                "type": "segmentation"
            },
            "sam2_s.pt": {
                "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
                "checksum": None,
                "type": "segmentation"
            },
            "sam2_b.pt": {
                "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
                "checksum": None,
                "type": "segmentation"
            },
            "sam2_l.pt": {
                "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
                "checksum": None,
                "type": "segmentation"
            }
        }
    
    def get_model_path(self, model_name: str) -> Path:
        """Get full path to model file."""
        return self.models_root / model_name
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if model is available locally."""
        model_path = self.get_model_path(model_name)
        return model_path.exists() and model_path.is_file()
    
    def download_model(
        self, 
        model_name: str, 
        force_download: bool = False,
        verify_checksum: bool = True
    ) -> Path:
        """
        Download model if not available locally.
        
        Args:
            model_name: Name of the model to download
            force_download: Force re-download even if model exists
            verify_checksum: Verify file integrity with checksum
            
        Returns:
            Path to downloaded model
            
        Raises:
            ValueError: If model not found in registry
            RuntimeError: If download fails
        """
        if model_name not in self.model_registry:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        model_path = self.get_model_path(model_name)
        model_info = self.model_registry[model_name]
        
        # Check if already exists and not forcing download
        if self.is_model_available(model_name) and not force_download:
            if verify_checksum and model_info["checksum"]:
                if self._verify_checksum(model_path, model_info["checksum"]):
                    logger.info(f"Model '{model_name}' already available")
                    return model_path
                else:
                    logger.warning(f"Checksum mismatch for '{model_name}', re-downloading")
            else:
                logger.info(f"Model '{model_name}' already available")
                return model_path
        
        # Download model
        logger.info(f"Downloading model: {model_name}")
        url = model_info["url"]
        
        try:
            self._download_file(url, model_path)
            
            # Verify checksum if provided
            if verify_checksum and model_info["checksum"]:
                if not self._verify_checksum(model_path, model_info["checksum"]):
                    model_path.unlink()  # Remove corrupted file
                    raise RuntimeError(f"Checksum verification failed for '{model_name}'")
            
            logger.info(f"Successfully downloaded: {model_name}")
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to download '{model_name}': {e}")
            if model_path.exists():
                model_path.unlink()  # Clean up partial download
            raise RuntimeError(f"Download failed for '{model_name}': {e}")
    
    def _download_file(self, url: str, output_path: Path) -> None:
        """Download file with progress bar."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as file, tqdm(
            desc=output_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = file.write(chunk)
                pbar.update(size)
    
    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file checksum."""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            actual_checksum = sha256_hash.hexdigest()
            return actual_checksum == expected_checksum
            
        except Exception as e:
            logger.error(f"Checksum verification failed: {e}")
            return False
    
    def ensure_model(self, model_name: str) -> Path:
        """
        Ensure model is available, download if necessary.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to model file
        """
        if not self.is_model_available(model_name):
            return self.download_model(model_name)
        return self.get_model_path(model_name)
    
    def list_available_models(self) -> Dict[str, Dict]:
        """List all available models with their status."""
        models_status = {}
        
        for model_name, model_info in self.model_registry.items():
            is_available = self.is_model_available(model_name)
            model_path = self.get_model_path(model_name)
            
            status = {
                "available": is_available,
                "type": model_info["type"],
                "path": str(model_path) if is_available else None,
                "size_mb": model_path.stat().st_size / (1024 * 1024) if is_available else None,
                "url": model_info["url"]
            }
            
            models_status[model_name] = status
        
        return models_status
    
    def download_all_models(self, model_type: Optional[str] = None) -> None:
        """
        Download all models or models of specific type.
        
        Args:
            model_type: Type of models to download (vlm, detection, segmentation)
        """
        models_to_download = []
        
        for model_name, model_info in self.model_registry.items():
            if model_type is None or model_info["type"] == model_type:
                if not self.is_model_available(model_name):
                    models_to_download.append(model_name)
        
        if not models_to_download:
            logger.info("All models already available")
            return
        
        logger.info(f"Downloading {len(models_to_download)} models...")
        
        for model_name in models_to_download:
            try:
                self.download_model(model_name)
            except Exception as e:
                logger.error(f"Failed to download {model_name}: {e}")
                continue
    
    def remove_model(self, model_name: str) -> bool:
        """
        Remove model from local storage.
        
        Args:
            model_name: Name of model to remove
            
        Returns:
            True if successfully removed
        """
        model_path = self.get_model_path(model_name)
        
        if model_path.exists():
            try:
                model_path.unlink()
                logger.info(f"Removed model: {model_name}")
                return True
            except Exception as e:
                logger.error(f"Failed to remove {model_name}: {e}")
                return False
        else:
            logger.warning(f"Model not found: {model_name}")
            return False
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get detailed information about a model."""
        if model_name not in self.model_registry:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        model_info = self.model_registry[model_name].copy()
        model_path = self.get_model_path(model_name)
        
        model_info.update({
            "name": model_name,
            "available": self.is_model_available(model_name),
            "path": str(model_path),
            "size_mb": model_path.stat().st_size / (1024 * 1024) if model_path.exists() else None
        })
        
        return model_info
    
    def migrate_models(self, old_location: Union[str, Path]) -> None:
        """
        Migrate models from old location to centralized models directory.
        
        Args:
            old_location: Path to old models directory
        """
        old_location = Path(old_location)
        
        if not old_location.exists():
            logger.warning(f"Old models directory not found: {old_location}")
            return
        
        logger.info(f"Migrating models from {old_location} to {self.models_root}")
        
        migrated_count = 0
        
        for model_name in self.model_registry.keys():
            old_path = old_location / model_name
            new_path = self.get_model_path(model_name)
            
            if old_path.exists() and not new_path.exists():
                try:
                    shutil.copy2(old_path, new_path)
                    logger.info(f"Migrated: {model_name}")
                    migrated_count += 1
                except Exception as e:
                    logger.error(f"Failed to migrate {model_name}: {e}")
        
        logger.info(f"Migration completed. Migrated {migrated_count} models.")
    
    def cleanup_cache(self) -> None:
        """Clean up old or corrupted model files."""
        logger.info("Cleaning up model cache...")
        
        cleaned_count = 0
        
        for model_file in self.models_root.iterdir():
            if model_file.is_file():
                model_name = model_file.name
                
                # Check if it's a known model
                if model_name in self.model_registry:
                    model_info = self.model_registry[model_name]
                    
                    # Verify checksum if available
                    if model_info["checksum"]:
                        if not self._verify_checksum(model_file, model_info["checksum"]):
                            logger.warning(f"Removing corrupted model: {model_name}")
                            model_file.unlink()
                            cleaned_count += 1
                else:
                    # Unknown model file
                    logger.info(f"Found unknown model file: {model_name}")
        
        logger.info(f"Cleanup completed. Removed {cleaned_count} files.")


def migrate_legacy_models():
    """Migrate models from legacy locations."""
    manager = ModelManager()
    
    # Common legacy locations
    legacy_locations = [
        "Model",
        "models", 
        "checkpoints",
        "AlphaCLIP/checkpoints",
        "."  # Root directory
    ]
    
    for location in legacy_locations:
        if os.path.exists(location):
            manager.migrate_models(location)
