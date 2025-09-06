import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any

class ObjectDetector:
    def __init__(self, model_path='yolo11n.pt'):
        self.model = YOLO(model_path)
        self.class_names = self.model.names
    
    def detect_frame(self, frame: np.ndarray, conf_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Detect objects in a single frame and return structured data
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Dictionary containing:
            - 'frame': Original frame
            - 'detections': List of detection dictionaries
            - 'bboxes': Numpy array of bounding boxes (N, 4) - [x1, y1, x2, y2]
            - 'scores': Numpy array of confidence scores (N,)
            - 'class_ids': Numpy array of class IDs (N,)
            - 'class_names': List of class names for detected objects
        """
        results = self.model.predict(frame, conf=conf_threshold, verbose=False)
        result = results[0]
        
        detections = []
        bboxes = []
        scores = []
        class_ids = []
        class_names = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_indices = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_indices):
                detection = {
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'class_id': int(cls_id),
                    'class_name': self.class_names[cls_id]
                }
                detections.append(detection)
                bboxes.append(box)
                scores.append(conf)
                class_ids.append(cls_id)
                class_names.append(self.class_names[cls_id])
        
        return {
            'frame': frame,
            'detections': detections,
            'bboxes': np.array(bboxes) if bboxes else np.empty((0, 4)),
            'scores': np.array(scores) if scores else np.empty((0,)),
            'class_ids': np.array(class_ids) if class_ids else np.empty((0,), dtype=int),
            'class_names': class_names
        }
    
    def detect_video_frames(self, video_path: str, conf_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Process video and return list of frame detections
        
        Args:
            video_path: Path to video file
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of detection dictionaries for each frame
        """
        cap = cv2.VideoCapture(video_path)
        frame_detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            detection_data = self.detect_frame(frame, conf_threshold)
            frame_detections.append(detection_data)
        
        cap.release()
        return frame_detections
    
    
    def get_detection_summary(self, frame_detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics from frame detections
        Args:
            frame_detections: List of frame detection dictionaries
            
        Returns:
            Summary dictionary with statistics
        """
        total_detections = sum(len(frame['detections']) for frame in frame_detections)
        class_counts = {}
        
        for frame in frame_detections:
            for detection in frame['detections']:
                class_name = detection['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'total_frames': len(frame_detections),
            'total_detections': total_detections,
            'class_counts': class_counts,
            'avg_detections_per_frame': total_detections / len(frame_detections) if frame_detections else 0
        }
    