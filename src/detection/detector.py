"""Object detection using YOLO models."""

import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Any
import logging

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logging.warning("Ultralytics not available. Please install: pip install ultralytics")


class ObjectDetector:
    """Object detection using YOLOv8."""
    
    def __init__(
        self, 
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ):
        """Initialize object detector.
        
        Args:
            model_name: YOLO model name (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics is required for object detection. "
                "Install with: pip install ultralytics"
            )
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load YOLO model."""
        try:
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            self.logger.info(f"Loaded YOLO model: {self.model_name} on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect(
        self, 
        image: Image.Image,
        min_area: int = 1000,
        max_objects: int = 10
    ) -> List[Dict[str, Any]]:
        """Detect objects in image.
        
        Args:
            image: PIL Image
            min_area: Minimum object area in pixels
            max_objects: Maximum number of objects to return
            
        Returns:
            List of detection dictionaries with keys:
                - bbox: (x1, y1, x2, y2) coordinates
                - confidence: Detection confidence
                - class_id: Class ID
                - class_name: Class name
                - area: Bounding box area
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Run detection
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Get bounding box coordinates
                bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = bbox
                
                # Calculate area
                area = (x2 - x1) * (y2 - y1)
                
                # Filter by minimum area
                if area < min_area:
                    continue
                
                # Get confidence and class
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.model.names[class_id]
                
                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name,
                    'area': area
                }
                
                detections.append(detection)
        
        # Sort by confidence and limit number of objects
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        detections = detections[:max_objects]
        
        self.logger.info(f"Detected {len(detections)} objects")
        
        return detections
    
    def visualize_detections(
        self, 
        image: Image.Image, 
        detections: List[Dict[str, Any]]
    ) -> Image.Image:
        """Visualize detections on image.
        
        Args:
            image: Original PIL Image
            detections: List of detection dictionaries
            
        Returns:
            PIL Image with visualized detections
        """
        from PIL import ImageDraw, ImageFont
        
        # Create a copy of the image
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', 
            '#00FFFF', '#FFA500', '#800080', '#FFC0CB', '#A52A2A'
        ]
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            
            # Choose color
            color = colors[i % len(colors)]
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw label
            label = f"{det['class_name']} ({det['confidence']:.2f})"
            text_bbox = draw.textbbox((x1, y1-20), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1-20), label, fill='white', font=font)
        
        return vis_image
