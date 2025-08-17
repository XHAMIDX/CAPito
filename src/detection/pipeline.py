"""Detection pipeline combining object detection and segmentation."""

import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple
import logging

try:
    from .detector import ObjectDetector
    from .segmentation import MaskGenerator
    from ..utils.image_utils import mask_to_alpha, numpy_to_torch_alpha
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from detector import ObjectDetector
    from segmentation import MaskGenerator
    from utils.image_utils import mask_to_alpha, numpy_to_torch_alpha


class DetectionPipeline:
    """Combined object detection and segmentation pipeline."""
    
    def __init__(
        self,
        detection_model: str = "yolov8n.pt",
        sam_model: str = "sam2_t.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cuda:8"
    ):
        """Initialize detection pipeline.
        
        Args:
            detection_model: YOLO model name
            sam_model: SAM2 model name
            conf_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on
        """
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Try to get model paths from config if available
        try:
            from ..config import ModelPathsConfig
            model_paths = ModelPathsConfig()
            detection_model_path = model_paths.get_detection_model_path(detection_model)
            sam_model_path = model_paths.get_detection_model_path(sam_model)
        except ImportError:
            detection_model_path = detection_model
            sam_model_path = sam_model
        
        # Initialize detector
        self.detector = ObjectDetector(
            model_name=detection_model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device
        )
        
        # Initialize mask generator
        self.mask_generator = MaskGenerator(
            model_name=sam_model_path,
            device=device
        )
    
    def process_image(
        self,
        image: Image.Image,
        min_area: int = 1000,
        max_objects: int = 10,
        mask_threshold: float = 0.5,
        blur_radius: int = 2
    ) -> Dict[str, Any]:
        """Process image through full detection and segmentation pipeline.
        
        Args:
            image: PIL Image
            min_area: Minimum object area in pixels
            max_objects: Maximum number of objects to process
            mask_threshold: Threshold for mask binarization
            blur_radius: Blur radius for alpha mask smoothing
            
        Returns:
            Dictionary containing:
                - detections: List of detection dictionaries with masks
                - alpha_masks: List of alpha masks ready for AlphaCLIP
                - visualization: PIL Image with visualized detections
        """
        self.logger.info("Starting detection pipeline")
        
        # Step 1: Object detection
        detections = self.detector.detect(
            image=image,
            min_area=min_area,
            max_objects=max_objects
        )
        
        if not detections:
            self.logger.warning("No objects detected")
            return {
                'detections': [],
                'alpha_masks': [],
                'visualization': image.copy()
            }
        
        # Step 2: Mask generation
        detections_with_masks = self.mask_generator.generate_masks(
            image=image,
            detections=detections,
            mask_threshold=mask_threshold
        )
        
        # Step 3: Convert masks to AlphaCLIP-compatible alpha masks
        alpha_masks = []
        for det in detections_with_masks:
            binary_mask = det['mask']
            
            # Convert to alpha mask with smoothing
            alpha_mask = mask_to_alpha(binary_mask, blur_radius=blur_radius)
            
            alpha_masks.append({
                'alpha_mask': alpha_mask,
                'detection': det
            })
        
        # Step 4: Create visualization
        visualization = self._create_visualization(image, detections_with_masks)
        
        self.logger.info(f"Pipeline completed: {len(alpha_masks)} objects processed")
        
        return {
            'detections': detections_with_masks,
            'alpha_masks': alpha_masks,
            'visualization': visualization
        }
    
    def prepare_alpha_masks_for_clip(
        self,
        alpha_masks: List[Dict[str, Any]],
        target_size: Tuple[int, int] = (224, 224)
    ) -> List[Dict[str, Any]]:
        """Prepare alpha masks for AlphaCLIP inference.
        
        Args:
            alpha_masks: List of alpha mask dictionaries
            target_size: Target size for AlphaCLIP (height, width)
            
        Returns:
            List of dictionaries with torch tensors ready for AlphaCLIP
        """
        prepared_masks = []
        
        for mask_info in alpha_masks:
            alpha_mask = mask_info['alpha_mask']
            detection = mask_info['detection']
            
            # Convert to torch tensor
            alpha_tensor = numpy_to_torch_alpha(
                alpha_mask=alpha_mask,
                target_size=target_size,
                device=self.device
            )
            
            prepared_masks.append({
                'alpha_tensor': alpha_tensor,
                'detection': detection,
                'class_name': detection['class_name'],
                'confidence': detection['confidence'],
                'bbox': detection['bbox']
            })
        
        return prepared_masks
    
    def _create_visualization(
        self,
        image: Image.Image,
        detections_with_masks: List[Dict[str, Any]]
    ) -> Image.Image:
        """Create visualization combining bounding boxes and masks.
        
        Args:
            image: Original PIL Image
            detections_with_masks: List of detection dictionaries with masks
            
        Returns:
            PIL Image with visualized detections and masks
        """
        # First, visualize masks
        vis_image = self.mask_generator.visualize_masks(
            image=image,
            detections_with_masks=detections_with_masks,
            alpha=0.3
        )
        
        # Then, add bounding boxes and labels
        vis_image = self.detector.visualize_detections(
            image=vis_image,
            detections=detections_with_masks
        )
        
        return vis_image
    
    def process_batch(
        self,
        images: List[Image.Image],
        image_names: List[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process a batch of images.
        
        Args:
            images: List of PIL Images
            image_names: Optional list of image names
            **kwargs: Additional arguments for process_image
            
        Returns:
            List of processing results for each image
        """
        if image_names is None:
            image_names = [f"image_{i}" for i in range(len(images))]
        
        results = []
        
        for i, (image, name) in enumerate(zip(images, image_names)):
            self.logger.info(f"Processing image {i+1}/{len(images)}: {name}")
            
            try:
                result = self.process_image(image, **kwargs)
                result['image_name'] = name
                result['success'] = True
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process {name}: {e}")
                results.append({
                    'image_name': name,
                    'success': False,
                    'error': str(e),
                    'detections': [],
                    'alpha_masks': [],
                    'visualization': image.copy()
                })
        
        return results
