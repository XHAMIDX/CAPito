"""
Depth Estimation Module
======================

Provides depth estimation capabilities for scene understanding.
Integrates with the CAPito pipeline to add spatial depth information to detected objects.
"""

import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Union
import logging


class DepthEstimator:
    """
    Depth estimation for scene understanding.
    
    Provides depth values for detected objects to enable 3D scene analysis.
    """
    
    def __init__(self, device: str = None):
        """
        Initialize depth estimator.
        
        Args:
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        # Initialize depth estimation model
        try:
            from transformers import pipeline
            self.model = pipeline("depth-estimation", model="Intel/dpt-large", device=self.device)
            self.logger.info(f"Depth estimator initialized on {self.device} using DPT-Large")
        except Exception as e:
            self.logger.warning(f"Failed to load DPT model: {e}")
            self.logger.info("Using fallback depth estimation...")
            self.model = None
    
    def estimate_depth(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Estimate depth for entire image.
        
        Args:
            image: Input image as numpy array (H, W, C) or PIL Image
            
        Returns:
            Depth map as numpy array (H, W) with normalized depth values
        """
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = image
        
        if self.model is None:
            return self._fallback_depth_estimation(pil_image)
        
        try:
            # Predict depth using model
            result = self.model(pil_image)
            depth_map = np.array(result["depth"])
            
            # Normalize depth values to [0, 1] range
            depth_map = depth_map.astype(np.float32)
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            
            return depth_map
            
        except Exception as e:
            self.logger.warning(f"Depth estimation failed: {e}")
            return self._fallback_depth_estimation(pil_image)
    
    def get_object_depth(
        self, 
        depth_map: np.ndarray, 
        bbox: List[float],
        image_size: tuple
    ) -> float:
        """
        Get average depth for a specific object bounding box.
        
        Args:
            depth_map: Full image depth map
            bbox: Bounding box [x1, y1, x2, y2] in normalized coordinates [0, 1]
            image_size: (height, width) of original image
            
        Returns:
            Average depth value for the object region
        """
        h, w = image_size
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)
        
        # Ensure coordinates are within bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        
        # Extract object region and compute average depth
        object_region = depth_map[y1:y2, x1:x2]
        
        if object_region.size == 0:
            return 0.5  # Default middle depth
        
        # Use median depth to avoid outliers
        return float(np.median(object_region))
    
    def _fallback_depth_estimation(self, image: Image.Image) -> np.ndarray:
        """
        Simple fallback depth estimation based on image intensity.
        
        Args:
            image: PIL Image
            
        Returns:
            Simple depth map based on image brightness
        """
        # Convert to grayscale
        gray = np.array(image.convert('L'))
        
        # Invert so brighter areas appear closer (common assumption)
        depth = 255 - gray
        
        # Normalize to [0, 1] range
        depth = depth.astype(np.float32) / 255.0
        
        return depth
    
    def cleanup(self):
        """Clean up model resources."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
