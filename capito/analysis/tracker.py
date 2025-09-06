"""
Object Tracking Module
=====================

Simple object tracking for maintaining object identity across frames.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging


class ObjectTracker:
    """
    Simple object tracker for maintaining object identity.
    
    Uses IoU-based tracking to maintain object identities across frames.
    """
    
    def __init__(self, max_disappeared: int = 10, max_distance: float = 0.3):
        """
        Initialize object tracker.
        
        Args:
            max_disappeared: Max frames an object can be missing before removal
            max_distance: Maximum distance threshold for object matching
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.logger = logging.getLogger(__name__)
    
    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries with 'bbox' key
            
        Returns:
            List of tracked objects with 'track_id' added
        """
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            return []
        
        # If no existing objects, register all detections as new
        if len(self.objects) == 0:
            for detection in detections:
                self._register(detection)
        else:
            # Compute cost matrix between existing objects and new detections
            object_ids = list(self.objects.keys())
            cost_matrix = np.zeros((len(object_ids), len(detections)))
            
            for i, object_id in enumerate(object_ids):
                for j, detection in enumerate(detections):
                    cost_matrix[i, j] = self._compute_distance(
                        self.objects[object_id]['bbox'], 
                        detection['bbox']
                    )
            
            # Assign detections to existing objects
            self._assign_detections(object_ids, detections, cost_matrix)
        
        # Prepare output with track IDs
        tracked_objects = []
        for detection in detections:
            if 'track_id' in detection:
                tracked_objects.append(detection)
        
        return tracked_objects
    
    def _register(self, detection: Dict[str, Any]) -> None:
        """Register a new object."""
        self.objects[self.next_id] = detection.copy()
        self.disappeared[self.next_id] = 0
        detection['track_id'] = self.next_id
        self.next_id += 1
    
    def _deregister(self, object_id: int) -> None:
        """Deregister an object."""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def _compute_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute distance between two bounding boxes using IoU."""
        try:
            # Calculate IoU
            x1 = max(bbox1[0], bbox2[0])
            y1 = max(bbox1[1], bbox2[1])
            x2 = min(bbox1[2], bbox2[2])
            y2 = min(bbox1[3], bbox2[3])
            
            if x2 <= x1 or y2 <= y1:
                return 1.0  # No overlap
            
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            union = area1 + area2 - intersection
            
            iou = intersection / (union + 1e-8)
            return 1.0 - iou  # Convert IoU to distance
            
        except Exception:
            return 1.0
    
    def _assign_detections(
        self, 
        object_ids: List[int], 
        detections: List[Dict[str, Any]], 
        cost_matrix: np.ndarray
    ) -> None:
        """Assign detections to existing objects."""
        # Simple greedy assignment
        used_detection_indices = set()
        
        for i, object_id in enumerate(object_ids):
            # Find best matching detection
            min_cost = float('inf')
            best_detection_idx = -1
            
            for j in range(len(detections)):
                if j in used_detection_indices:
                    continue
                
                if cost_matrix[i, j] < min_cost:
                    min_cost = cost_matrix[i, j]
                    best_detection_idx = j
            
            # Assign if distance is acceptable
            if min_cost < self.max_distance and best_detection_idx != -1:
                # Update existing object
                self.objects[object_id] = detections[best_detection_idx].copy()
                self.disappeared[object_id] = 0
                detections[best_detection_idx]['track_id'] = object_id
                used_detection_indices.add(best_detection_idx)
            else:
                # Mark object as disappeared
                self.disappeared[object_id] += 1
        
        # Register new detections that weren't assigned
        for j, detection in enumerate(detections):
            if j not in used_detection_indices:
                self._register(detection)
    
    def get_object_count(self) -> int:
        """Get current number of tracked objects."""
        return len(self.objects)
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.objects = {}
        self.disappeared = {}
        self.next_id = 0
