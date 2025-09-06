import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from ultralytics import YOLO
import torch

class DeepSORTTracker:
    def __init__(self, model_path='yolo11n.pt', max_age=30, min_hits=1, iou_threshold=0.3):
        """
        Initialize DeepSORT tracker
        
        Args:
            model_path: YOLO model path
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections before confirming track
            iou_threshold: IOU threshold for association
        """
        self.model = YOLO(model_path)
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        # Track storage
        self.tracks = []
        self.frame_count = 0
        self.next_id = 1
        
        print("DeepSORT Tracker initialized")
    
    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of detection dictionaries with bbox, confidence, class_name
            
        Returns:
            List of tracked objects with track_id
        """
        self.frame_count += 1
        
        # Convert detections to format expected by tracker
        detection_boxes = []
        detection_scores = []
        detection_classes = []
        
        for det in detections:
            bbox = det['bbox']
            detection_boxes.append(bbox)
            detection_scores.append(det['confidence'])
            detection_classes.append(det['class_name'])
        
        if not detection_boxes:
            # No detections, update existing tracks
            self._update_tracks_without_detections()
            return self._get_tracked_objects()
        
        # Convert to numpy arrays
        detection_boxes = np.array(detection_boxes)
        detection_scores = np.array(detection_scores)
        
        # Associate detections with existing tracks
        matched_tracks, unmatched_tracks, unmatched_detections = self._associate_detections_to_tracks(
            detection_boxes, detection_scores
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            self.tracks[track_idx]['bbox'] = detection_boxes[det_idx]
            self.tracks[track_idx]['confidence'] = detection_scores[det_idx]
            self.tracks[track_idx]['class_name'] = detection_classes[det_idx]
            self.tracks[track_idx]['age'] = 0
            self.tracks[track_idx]['hits'] += 1
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self._create_new_track(
                detection_boxes[det_idx],
                detection_scores[det_idx],
                detection_classes[det_idx]
            )
        
        # Update unmatched tracks (increment age)
        for track_idx in unmatched_tracks:
            self.tracks[track_idx]['age'] += 1
        
        # Remove old tracks
        self._remove_old_tracks()
        
        return self._get_tracked_objects()
    
    def _associate_detections_to_tracks(self, detection_boxes, detection_scores):
        """
        Associate detections to existing tracks using IOU
        """
        if not self.tracks:
            return [], [], list(range(len(detection_boxes)))
        
        # Calculate IOU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detection_boxes)))
        for i, track in enumerate(self.tracks):
            for j, det_box in enumerate(detection_boxes):
                iou_matrix[i, j] = self._calculate_iou(track['bbox'], det_box)
        
        # Simple greedy association
        matched_tracks = []
        matched_detections = []
        
        # Find best matches
        while True:
            max_iou = np.max(iou_matrix)
            if max_iou < self.iou_threshold:
                break
            
            track_idx, det_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            matched_tracks.append((track_idx, det_idx))
            matched_detections.append(det_idx)
            
            # Remove matched row and column
            iou_matrix[track_idx, :] = 0
            iou_matrix[:, det_idx] = 0
        
        # Find unmatched tracks and detections
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in [t[0] for t in matched_tracks]]
        unmatched_detections = [i for i in range(len(detection_boxes)) if i not in matched_detections]
        
        return matched_tracks, unmatched_tracks, unmatched_detections
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union between two bounding boxes
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _create_new_track(self, bbox, confidence, class_name):
        """
        Create a new track
        """
        track = {
            'track_id': self.next_id,
            'bbox': bbox,
            'confidence': confidence,
            'class_name': class_name,
            'age': 0,
            'hits': 1,
            'total_hits': 1
        }
        self.tracks.append(track)
        self.next_id += 1
    
    def _update_tracks_without_detections(self):
        """
        Update tracks when no detections are available
        """
        for track in self.tracks:
            track['age'] += 1
    
    def _remove_old_tracks(self):
        """
        Remove tracks that are too old or have too few hits
        """
        self.tracks = [track for track in self.tracks 
                      if track['age'] < self.max_age]
    
    def _get_tracked_objects(self):
        """
        Get current tracked objects
        """
        tracked_objects = []
        for track in self.tracks:
            # Return all tracks for debugging, not just confirmed ones
            tracked_objects.append({
                'track_id': track['track_id'],
                'bbox': track['bbox'],
                'confidence': track['confidence'],
                'class_name': track['class_name'],
                'age': track['age'],
                'hits': track['hits'],
                'total_hits': track['total_hits']
            })
        return tracked_objects
    
    def reset(self):
        """
        Reset tracker state
        """
        self.tracks = []
        self.frame_count = 0
        self.next_id = 1 