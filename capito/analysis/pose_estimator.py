"""
Pose Estimation Module
=====================

Human pose estimation for enhanced scene understanding.
Provides keypoint detection and pose analysis for human subjects.
"""

import torch
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any, Optional, Union
import logging


class PoseEstimator:
    """
    Human pose estimation using YOLO pose models.
    
    Provides keypoint detection and pose analysis for human subjects
    to enhance scene understanding and object relationships.
    """
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize pose estimator.
        
        Args:
            model_path: Path to YOLO pose model (defaults to yolov8n-pose.pt)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        try:
            from ultralytics import YOLO
            model_path = model_path or 'yolov8n-pose.pt'
            self.model = YOLO(model_path)
            self.logger.info(f"Pose estimator initialized with {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load pose model: {e}")
            self.model = None
        
        # COCO keypoint names (17 keypoints)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def estimate_poses(
        self, 
        image: Union[np.ndarray, Image.Image], 
        conf_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Detect human poses in image.
        
        Args:
            image: Input image
            conf_threshold: Confidence threshold for pose detection
            
        Returns:
            List of pose dictionaries with keypoints and metadata
        """
        if self.model is None:
            return []
        
        try:
            # Run pose detection
            results = self.model.predict(image, conf=conf_threshold, verbose=False)
            
            poses = []
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints = result.keypoints.data.cpu().numpy()
                    
                    # Get bounding boxes if available
                    boxes = None
                    confidences = None
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                    
                    for i, kpts in enumerate(keypoints):
                        pose_data = {
                            'keypoints': self._process_keypoints(kpts),
                            'keypoint_names': self.keypoint_names,
                            'bbox': boxes[i].tolist() if boxes is not None else None,
                            'confidence': float(confidences[i]) if confidences is not None else 1.0,
                            'pose_features': self._extract_pose_features(kpts)
                        }
                        poses.append(pose_data)
            
            return poses
            
        except Exception as e:
            self.logger.warning(f"Pose estimation failed: {e}")
            return []
    
    def _process_keypoints(self, keypoints: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Process raw keypoints into structured format.
        
        Args:
            keypoints: Raw keypoint array (17, 3) - [x, y, confidence]
            
        Returns:
            Dictionary mapping keypoint names to coordinates and confidence
        """
        processed = {}
        
        for i, name in enumerate(self.keypoint_names):
            if i < len(keypoints):
                x, y, conf = keypoints[i]
                processed[name] = {
                    'x': float(x),
                    'y': float(y),
                    'confidence': float(conf),
                    'visible': conf > 0.5
                }
            else:
                processed[name] = {
                    'x': 0.0,
                    'y': 0.0,
                    'confidence': 0.0,
                    'visible': False
                }
        
        return processed
    
    def _extract_pose_features(self, keypoints: np.ndarray) -> Dict[str, Any]:
        """
        Extract high-level pose features for graph analysis.
        
        Args:
            keypoints: Raw keypoint array (17, 3)
            
        Returns:
            Dictionary of pose features
        """
        features = {
            'pose_type': 'unknown',
            'body_orientation': 'unknown',
            'activity': 'standing',
            'limb_positions': {}
        }
        
        try:
            # Basic pose analysis
            visible_kpts = keypoints[keypoints[:, 2] > 0.5]
            
            if len(visible_kpts) > 0:
                # Estimate body orientation based on shoulder positions
                left_shoulder = keypoints[5] if len(keypoints) > 5 else None
                right_shoulder = keypoints[6] if len(keypoints) > 6 else None
                
                if (left_shoulder is not None and right_shoulder is not None and 
                    left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5):
                    
                    shoulder_angle = np.arctan2(
                        right_shoulder[1] - left_shoulder[1],
                        right_shoulder[0] - left_shoulder[0]
                    )
                    features['body_orientation'] = f"{np.degrees(shoulder_angle):.1f}°"
                
                # Simple activity detection based on limb positions
                features['activity'] = self._detect_activity(keypoints)
        
        except Exception as e:
            self.logger.warning(f"Feature extraction failed: {e}")
        
        return features
    
    def _detect_activity(self, keypoints: np.ndarray) -> str:
        """
        Simple activity detection based on keypoint positions.
        
        Args:
            keypoints: Keypoint array
            
        Returns:
            Estimated activity string
        """
        try:
            # Get key points for analysis
            nose = keypoints[0] if len(keypoints) > 0 else None
            left_ankle = keypoints[15] if len(keypoints) > 15 else None
            right_ankle = keypoints[16] if len(keypoints) > 16 else None
            
            # Simple heuristics
            if (nose is not None and left_ankle is not None and 
                nose[2] > 0.5 and left_ankle[2] > 0.5):
                
                # If nose is much higher than ankles, likely standing
                height_ratio = (nose[1] - left_ankle[1]) / (left_ankle[1] + 1e-8)
                
                if height_ratio > 0.3:
                    return "standing"
                elif height_ratio < -0.1:
                    return "sitting"
                else:
                    return "walking"
            
            return "unknown"
            
        except Exception:
            return "unknown"
    
    def get_pose_for_human(
        self, 
        image: Union[np.ndarray, Image.Image],
        human_bbox: List[float]
    ) -> Optional[Dict[str, Any]]:
        """
        Get pose information for a specific human detection.
        
        Args:
            image: Input image
            human_bbox: Human bounding box [x1, y1, x2, y2]
            
        Returns:
            Pose information for the human or None if not found
        """
        poses = self.estimate_poses(image)
        
        if not poses:
            return None
        
        # Find the pose that best matches the given bounding box
        best_pose = None
        best_overlap = 0.0
        
        for pose in poses:
            if pose['bbox'] is not None:
                overlap = self._calculate_bbox_overlap(human_bbox, pose['bbox'])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_pose = pose
        
        return best_pose if best_overlap > 0.3 else None
    
    def _calculate_bbox_overlap(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU overlap between two bounding boxes."""
        try:
            x1 = max(bbox1[0], bbox2[0])
            y1 = max(bbox1[1], bbox2[1])
            x2 = min(bbox1[2], bbox2[2])
            y2 = min(bbox1[3], bbox2[3])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            union = area1 + area2 - intersection
            
            return intersection / (union + 1e-8)
            
        except Exception:
            return 0.0
    
    def cleanup(self):
        """Clean up model resources."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
