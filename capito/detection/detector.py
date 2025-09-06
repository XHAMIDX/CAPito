"""
Object Detection Module
======================

YOLO-based object detection for the CAPito pipeline.
Provides clean interface for object detection and bounding box extraction.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import cv2
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError as e:
    logging.error(f"Failed to import ultralytics: {e}")
    logging.error("Please install ultralytics: pip install ultralytics")
    raise


class Detection:
    """Container for detection results."""
    
    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        confidence: float,
        class_id: int,
        class_name: str
    ):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        
        # Additional properties
        self.area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        self.center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "area": self.area,
            "center": self.center
        }
    
    def __repr__(self) -> str:
        return f"Detection(class='{self.class_name}', conf={self.confidence:.3f}, bbox={self.bbox})"


class ObjectDetector:
    """
    YOLO-based object detector.
    
    Provides methods for:
    - Object detection
    - Bounding box extraction
    - Confidence filtering
    - NMS post-processing
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cuda"
    ):
        """
        Initialize object detector.
        
        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = torch.device(device)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing ObjectDetector: {model_path}")
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load YOLO model."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = YOLO(self.model_path)
            
            # Move to device
            if self.device.type == 'cuda' and torch.cuda.is_available():
                self.model.to(self.device)
            
            self.logger.info("YOLO model loaded successfully")
            
            # Get class names
            self.class_names = self.model.names
            self.logger.info(f"Model supports {len(self.class_names)} classes")
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect(
        self, 
        image: Union[Image.Image, np.ndarray, str],
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        classes: Optional[List[int]] = None,
        max_detections: int = 100
    ) -> List[Detection]:
        """
        Detect objects in image.
        
        Args:
            image: Input image (PIL Image, numpy array, or file path)
            confidence_threshold: Override default confidence threshold
            iou_threshold: Override default IoU threshold
            classes: List of class IDs to detect (None for all)
            max_detections: Maximum number of detections to return
            
        Returns:
            List of Detection objects
        """
        # Use provided thresholds or defaults
        conf_thresh = confidence_threshold or self.confidence_threshold
        iou_thresh = iou_threshold or self.iou_threshold
        
        try:
            # Run inference
            results = self.model(
                image,
                conf=conf_thresh,
                iou=iou_thresh,
                classes=classes,
                max_det=max_detections,
                verbose=False
            )
            
            # Parse results
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Extract box coordinates (x1, y1, x2, y2)
                        bbox = boxes.xyxy[i].cpu().numpy()
                        confidence = boxes.conf[i].cpu().numpy().item()
                        class_id = int(boxes.cls[i].cpu().numpy().item())
                        class_name = self.class_names[class_id]
                        
                        detection = Detection(
                            bbox=tuple(bbox),
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name
                        )
                        
                        detections.append(detection)
            
            self.logger.info(f"Detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return []
    
    def detect_and_crop(
        self, 
        image: Union[Image.Image, np.ndarray],
        **kwargs
    ) -> List[Tuple[Detection, Union[Image.Image, np.ndarray]]]:
        """
        Detect objects and return cropped regions.
        
        Args:
            image: Input image
            **kwargs: Arguments passed to detect()
            
        Returns:
            List of (Detection, cropped_image) tuples
        """
        detections = self.detect(image, **kwargs)
        
        results = []
        for detection in detections:
            try:
                # Crop region
                if isinstance(image, Image.Image):
                    x1, y1, x2, y2 = detection.bbox
                    cropped = image.crop((int(x1), int(y1), int(x2), int(y2)))
                elif isinstance(image, np.ndarray):
                    x1, y1, x2, y2 = detection.bbox
                    cropped = image[int(y1):int(y2), int(x1):int(x2)]
                else:
                    self.logger.warning(f"Unsupported image type for cropping: {type(image)}")
                    continue
                
                results.append((detection, cropped))
                
            except Exception as e:
                self.logger.warning(f"Failed to crop detection: {e}")
                continue
        
        return results
    
    def filter_detections(
        self,
        detections: List[Detection],
        min_confidence: Optional[float] = None,
        min_area: Optional[float] = None,
        max_area: Optional[float] = None,
        classes: Optional[List[str]] = None
    ) -> List[Detection]:
        """
        Filter detections based on criteria.
        
        Args:
            detections: List of detections to filter
            min_confidence: Minimum confidence threshold
            min_area: Minimum bounding box area
            max_area: Maximum bounding box area
            classes: List of class names to keep
            
        Returns:
            Filtered list of detections
        """
        filtered = detections.copy()
        
        # Filter by confidence
        if min_confidence is not None:
            filtered = [d for d in filtered if d.confidence >= min_confidence]
        
        # Filter by area
        if min_area is not None:
            filtered = [d for d in filtered if d.area >= min_area]
        if max_area is not None:
            filtered = [d for d in filtered if d.area <= max_area]
        
        # Filter by class
        if classes is not None:
            filtered = [d for d in filtered if d.class_name in classes]
        
        return filtered
    
    def get_dominant_objects(
        self,
        detections: List[Detection],
        top_k: int = 5,
        sort_by: str = "confidence"
    ) -> List[Detection]:
        """
        Get the most dominant objects in the image.
        
        Args:
            detections: List of detections
            top_k: Number of top objects to return
            sort_by: Sorting criteria ("confidence", "area", "combined")
            
        Returns:
            List of top detections
        """
        if not detections:
            return []
        
        if sort_by == "confidence":
            sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        elif sort_by == "area":
            sorted_detections = sorted(detections, key=lambda x: x.area, reverse=True)
        elif sort_by == "combined":
            # Combine confidence and normalized area
            max_area = max(d.area for d in detections)
            min_area = min(d.area for d in detections)
            area_range = max_area - min_area if max_area != min_area else 1
            
            def combined_score(detection):
                norm_area = (detection.area - min_area) / area_range
                return detection.confidence * 0.7 + norm_area * 0.3
            
            sorted_detections = sorted(detections, key=combined_score, reverse=True)
        else:
            raise ValueError(f"Invalid sort_by value: {sort_by}")
        
        return sorted_detections[:top_k]
    
    def visualize_detections(
        self,
        image: Union[Image.Image, np.ndarray],
        detections: List[Detection],
        show_confidence: bool = True,
        show_class: bool = True,
        thickness: int = 2
    ) -> Union[Image.Image, np.ndarray]:
        """
        Visualize detections on image.
        
        Args:
            image: Input image
            detections: List of detections to visualize
            show_confidence: Whether to show confidence scores
            show_class: Whether to show class names
            thickness: Line thickness for bounding boxes
            
        Returns:
            Image with visualized detections
        """
        if isinstance(image, Image.Image):
            # Convert to numpy for OpenCV operations
            img_array = np.array(image)
            is_pil = True
        else:
            img_array = image.copy()
            is_pil = False
        
        # Ensure RGB format
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        for detection in detections:
            x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(img_array, (x1, y1), (x2, y2), color, thickness)
            
            # Create label
            label_parts = []
            if show_class:
                label_parts.append(detection.class_name)
            if show_confidence:
                label_parts.append(f"{detection.confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # Get text size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                text_thickness = 1
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
                
                # Draw label background
                cv2.rectangle(
                    img_array,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    img_array,
                    label,
                    (x1, y1 - 5),
                    font,
                    font_scale,
                    (255, 255, 255),
                    text_thickness
                )
        
        # Convert back to RGB
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Convert back to PIL if input was PIL
        if is_pil:
            return Image.fromarray(img_array)
        else:
            return img_array
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_path": self.model_path,
            "model_type": "YOLO",
            "device": str(self.device),
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "num_classes": len(self.class_names),
            "class_names": list(self.class_names.values())
        }
    
    def update_thresholds(
        self, 
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> None:
        """Update detection thresholds."""
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
        
        self.logger.info(f"Updated thresholds: conf={self.confidence_threshold}, iou={self.iou_threshold}")
    
    def cleanup(self) -> None:
        """Clean up model resources."""
        # YOLO models don't have explicit cleanup, but clear CUDA cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
