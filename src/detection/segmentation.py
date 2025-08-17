"""Mask generation using SAM2 from ultralytics."""

import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Any, Optional
import logging
import os

try:
    from ultralytics import SAM
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logging.warning("SAM2 not available in ultralytics. Using fallback bbox masks.")


class MaskGenerator:
    """Mask generation using SAM2 or fallback methods."""
    
    def __init__(
        self, 
        model_name: str = "sam2_b.pt",
        device: str = "cpu"
    ):
        """Initialize mask generator.
        
        Args:
            model_name: SAM model name (sam2_t.pt, sam2_s.pt, sam2_b.pt, sam2_l.pt)
            device: Device to run inference on
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.use_sam = SAM_AVAILABLE
        self.logger = logging.getLogger(__name__)
        
        if self.use_sam:
            self._load_sam_model()
        else:
            self.logger.warning("Using fallback bbox-based masks instead of SAM2")
    
    def _load_sam_model(self) -> None:
        """Load SAM2 model with improved error handling."""
        try:
            # Check if model file exists and is valid
            if os.path.exists(self.model_name):
                file_size = os.path.getsize(self.model_name)
                if file_size < 1024:  # Less than 1KB indicates corrupted file
                    raise ValueError(f"Model file {self.model_name} appears to be corrupted (size: {file_size} bytes)")
                
                self.logger.info(f"Loading SAM2 model from local path: {self.model_name}")
                self.model = SAM(self.model_name)
            else:
                # Try to download the model
                self.logger.info(f"Downloading SAM2 model: {self.model_name}")
                self.model = SAM(self.model_name)
            
            self.model.to(self.device)
            self.logger.info(f"Successfully loaded SAM2 model: {self.model_name} on {self.device}")
            
        except Exception as e:
            error_msg = str(e)
            if "PytorchStreamReader failed reading zip archive" in error_msg:
                self.logger.error(f"SAM2 model file {self.model_name} is corrupted or incomplete. Please re-download the model.")
            elif "failed finding central directory" in error_msg:
                self.logger.error(f"SAM2 model file {self.model_name} is corrupted. Please re-download the model.")
            else:
                self.logger.error(f"Failed to load SAM2 model: {error_msg}")
            
            self.logger.info("Falling back to bbox-based masks")
            self.use_sam = False
            self.model = None
    
    def generate_masks(
        self, 
        image: Image.Image,
        detections: List[Dict[str, Any]],
        mask_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Generate masks for detected objects.
        
        Args:
            image: PIL Image
            detections: List of detection dictionaries from ObjectDetector
            mask_threshold: Threshold for mask binarization
            
        Returns:
            List of detection dictionaries with added 'mask' key containing
            binary mask as numpy array (H, W) with values 0.0 or 1.0
        """
        if not detections:
            return []
        
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        
        results = []
        
        for detection in detections:
            detection_copy = detection.copy()
            
            if self.use_sam and self.model is not None:
                # Use SAM2 for precise segmentation
                mask = self._generate_sam_mask(
                    image, detection['bbox'], mask_threshold
                )
            else:
                # Fallback: use bounding box as mask
                mask = self._generate_bbox_mask(
                    (h, w), detection['bbox']
                )
            
            detection_copy['mask'] = mask
            results.append(detection_copy)
        
        self.logger.info(f"Generated masks for {len(results)} objects")
        return results
    
    def _generate_sam_mask(
        self, 
        image: Image.Image, 
        bbox: Tuple[int, int, int, int],
        threshold: float = 0.5
    ) -> np.ndarray:
        """Generate mask using SAM2.
        
        Args:
            image: PIL Image
            bbox: Bounding box (x1, y1, x2, y2)
            threshold: Mask threshold
            
        Returns:
            Binary mask array (H, W)
        """
        try:
            # Use bbox as prompt for SAM2
            x1, y1, x2, y2 = bbox
            
            # SAM2 expects prompts in specific format
            # Use the center point of bbox as point prompt
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Run SAM2 prediction
            results = self.model(image, points=[[center_x, center_y]], labels=[1])
            
            if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
                # Get the first (usually best) mask
                mask = results[0].masks.data[0].cpu().numpy()
                
                # Ensure mask is binary
                mask = (mask > threshold).astype(np.float32)
                
                return mask
            else:
                # Fallback if SAM2 fails
                self.logger.warning("SAM2 failed to generate mask, using bbox fallback")
                return self._generate_bbox_mask(image.size[::-1], bbox)
                
        except Exception as e:
            self.logger.warning(f"SAM2 mask generation failed: {e}. Using bbox fallback.")
            return self._generate_bbox_mask(image.size[::-1], bbox)
    
    def _generate_bbox_mask(
        self, 
        image_shape: Tuple[int, int], 
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Generate mask from bounding box.
        
        Args:
            image_shape: (height, width) of the image
            bbox: (x1, y1, x2, y2) bounding box coordinates
            
        Returns:
            Binary mask array (H, W)
        """
        try:
            from ..utils.image_utils import create_bbox_mask
        except ImportError:
            from utils.image_utils import create_bbox_mask
        return create_bbox_mask(image_shape, bbox)
    
    def visualize_masks(
        self, 
        image: Image.Image, 
        detections_with_masks: List[Dict[str, Any]],
        alpha: float = 0.5
    ) -> Image.Image:
        """Visualize masks overlaid on image.
        
        Args:
            image: Original PIL Image
            detections_with_masks: List of detection dictionaries with masks
            alpha: Transparency of mask overlay
            
        Returns:
            PIL Image with visualized masks
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        # Convert PIL to numpy
        image_np = np.array(image)
        
        # Create overlay
        overlay = image_np.copy()
        
        # Define colors for different objects
        colors = plt.cm.Set3(np.linspace(0, 1, len(detections_with_masks)))
        
        for i, det in enumerate(detections_with_masks):
            mask = det['mask']
            color = colors[i][:3]  # RGB only
            
            # Apply colored mask
            for c in range(3):
                overlay[:, :, c] = np.where(
                    mask > 0.5,
                    overlay[:, :, c] * (1 - alpha) + color[c] * 255 * alpha,
                    overlay[:, :, c]
                )
        
        return Image.fromarray(overlay.astype(np.uint8))
    
    def save_individual_masks(
        self, 
        detections_with_masks: List[Dict[str, Any]], 
        output_dir: str,
        image_name: str
    ) -> List[str]:
        """Save individual masks as separate images.
        
        Args:
            detections_with_masks: List of detection dictionaries with masks
            output_dir: Directory to save masks
            image_name: Base image name
            
        Returns:
            List of saved mask file paths
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        base_name = os.path.splitext(image_name)[0]
        
        for i, det in enumerate(detections_with_masks):
            mask = det['mask']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Create filename
            mask_filename = f"{base_name}_{class_name}_{confidence:.2f}_mask_{i}.png"
            mask_path = os.path.join(output_dir, mask_filename)
            
            # Convert mask to PIL Image and save
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
            mask_pil.save(mask_path)
            
            saved_paths.append(mask_path)
        
        return saved_paths
