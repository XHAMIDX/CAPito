# pip install torch torchvision
# pip install git+https://github.com/lpiccinelli-eth/UniDepth.git

import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional
from transformers import pipeline

class DepthEstimator:
    def __init__(self, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize MiDaS depth estimation model
        try:
            self.model = pipeline("depth-estimation", model="Intel/dpt-large", device=self.device)
            print(f"Depth estimator initialized on {self.device} using MiDaS")
        except Exception as e:
            print(f"Failed to load MiDaS model: {e}")
            print("Falling back to simple depth estimation...")
            self.model = None
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth for entire frame
        
        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format
            
        Returns:
            Depth map as numpy array (H, W) in meters
        """
        if self.model is None:
            # Fallback: simple depth estimation based on image intensity
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Invert so brighter areas appear closer
            depth = 255 - gray
            # Normalize to reasonable depth range (0-10 meters)
            depth = depth.astype(np.float32) * 10.0 / 255.0
            return depth
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(img_rgb)
        
        # Predict depth using MiDaS
        result = self.model(pil_image)
        depth_map = result['depth']
        
        # Convert to numpy array and normalize
        depth_array = np.array(depth_map)
        
        # Normalize to reasonable depth range (0-10 meters)
        depth_array = depth_array.astype(np.float32) * 10.0 / depth_array.max()
        
        return depth_array
    
    def get_object_depth(self, depth_map: np.ndarray, bbox: List[float]) -> Dict[str, Any]:
        """
        Get depth statistics for a specific object bounding box
        
        Args:
            depth_map: Depth map as numpy array (H, W) in meters
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Dictionary containing depth statistics
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within bounds
        h, w = depth_map.shape
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        # Extract depth region for the object
        object_depth = depth_map[y1:y2, x1:x2]
        
        if object_depth.size == 0:
            return {
                'min_depth': 0.0,
                'max_depth': 0.0,
                'mean_depth': 0.0,
                'median_depth': 0.0,
                'depth_std': 0.0,
                'depth_range': 0.0,
                'valid_pixels': 0
            }
        
        # Calculate depth statistics
        valid_depths = object_depth[object_depth > 0]  # Filter out invalid depths
        
        if len(valid_depths) == 0:
            return {
                'min_depth': 0.0,
                'max_depth': 0.0,
                'mean_depth': 0.0,
                'median_depth': 0.0,
                'depth_std': 0.0,
                'depth_range': 0.0,
                'valid_pixels': 0
            }
        
        return {
            'min_depth': float(np.min(valid_depths)),
            'max_depth': float(np.max(valid_depths)),
            'mean_depth': float(np.mean(valid_depths)),
            'median_depth': float(np.median(valid_depths)),
            'depth_std': float(np.std(valid_depths)),
            'depth_range': float(np.max(valid_depths) - np.min(valid_depths)),
            'valid_pixels': int(len(valid_depths))
        }
    
    def process_frame_with_objects(self, frame: np.ndarray, object_detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process frame and add depth information to object detections
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            object_detections: List of object detection dictionaries
            
        Returns:
            Dictionary containing:
            - 'frame': Original frame
            - 'depth_map': Depth map
            - 'objects_with_depth': List of objects with depth information
        """
        # Estimate depth for entire frame
        depth_map = self.estimate_depth(frame)
        
        # Add depth information to each object
        objects_with_depth = []
        
        for obj in object_detections:
            # Get depth statistics for this object
            depth_stats = self.get_object_depth(depth_map, obj['bbox'])
            
            # Create new object dict with depth information
            obj_with_depth = obj.copy()
            obj_with_depth['depth'] = depth_stats
            objects_with_depth.append(obj_with_depth)
        
        return {
            'frame': frame,
            'depth_map': depth_map,
            'objects_with_depth': objects_with_depth
        }
    
    def get_depth_summary(self, objects_with_depth: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics for depth information
        
        Args:
            objects_with_depth: List of objects with depth information
            
        Returns:
            Summary dictionary with depth statistics
        """
        if not objects_with_depth:
            return {
                'total_objects': 0,
                'objects_with_valid_depth': 0,
                'avg_depth': 0.0,
                'depth_range': {'min': 0.0, 'max': 0.0},
                'depth_by_class': {}
            }
        
        valid_depths = []
        depth_by_class = {}
        
        for obj in objects_with_depth:
            depth_info = obj['depth']
            class_name = obj['class_name']
            
            if depth_info['valid_pixels'] > 0:
                valid_depths.append(depth_info['mean_depth'])
                
                if class_name not in depth_by_class:
                    depth_by_class[class_name] = []
                depth_by_class[class_name].append(depth_info['mean_depth'])
        
        return {
            'total_objects': len(objects_with_depth),
            'objects_with_valid_depth': len(valid_depths),
            'avg_depth': float(np.mean(valid_depths)) if valid_depths else 0.0,
            'depth_range': {
                'min': float(np.min(valid_depths)) if valid_depths else 0.0,
                'max': float(np.max(valid_depths)) if valid_depths else 0.0
            },
            'depth_by_class': {
                class_name: {
                    'count': len(depths),
                    'avg_depth': float(np.mean(depths)),
                    'min_depth': float(np.min(depths)),
                    'max_depth': float(np.max(depths))
                } for class_name, depths in depth_by_class.items()
            }
        }
