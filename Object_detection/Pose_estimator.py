import torch
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any, Optional
from ultralytics import YOLO

class PoseEstimator:
    def __init__(self, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use YOLO pose model instead of transformers
        self.pose_model = YOLO('yolov8x-pose.pt')
        
        # COCO keypoint names
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def detect_humans(self, frame: np.ndarray, conf_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Detect humans in frame using YOLO pose model
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            conf_threshold: Confidence threshold for human detection
            
        Returns:
            List of human detection dictionaries
        """
        results = self.pose_model(frame, conf=conf_threshold, verbose=False)
        result = results[0]
        
        humans = []
        if result.keypoints is not None:
            keypoints = result.keypoints.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
            confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
            
            for i, (keypoint, box, conf) in enumerate(zip(keypoints, boxes, confidences)):
                humans.append({
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'class_name': 'person',
                    'keypoints': keypoint.tolist()
                })
        
        return humans
    
    def estimate_pose(self, frame: np.ndarray, human_bbox: List[float]) -> Dict[str, Any]:
        """
        Estimate pose for a single human bounding box
        
        Args:
            frame: Input frame as numpy array
            human_bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Pose estimation dictionary
        """
        # Crop human region
        x1, y1, x2, y2 = map(int, human_bbox)
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return {
                'keypoints': [],
                'keypoint_names': self.keypoint_names,
                'bbox': human_bbox
            }
        
        # Run pose estimation on crop
        results = self.pose_model(crop, verbose=False)
        result = results[0]
        
        if result.keypoints is not None and len(result.keypoints) > 0:
            keypoints = result.keypoints.data[0].cpu().numpy()  # (17, 3) - x, y, confidence
            
            # Convert keypoints back to original frame coordinates
            keypoints[:, 0] += x1
            keypoints[:, 1] += y1
            
            return {
                'keypoints': keypoints.tolist(),
                'keypoint_names': self.keypoint_names,
                'bbox': human_bbox
            }
        else:
            return {
                'keypoints': [],
                'keypoint_names': self.keypoint_names,
                'bbox': human_bbox
            }
    
    def process_frame(self, frame: np.ndarray, conf_threshold: float = 0.9) -> Dict[str, Any]:
        """
        Process frame to detect humans and estimate poses
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            conf_threshold: Confidence threshold for human detection
            
        Returns:
            Dictionary containing:
            - 'frame': Original frame
            - 'humans': List of human detections with poses
            - 'poses': List of pose estimations
            - 'keypoints': Numpy array of all keypoints (N, 17, 3)
        """
        # Detect humans with pose
        humans = self.detect_humans(frame, conf_threshold)
        
        poses = []
        all_keypoints = []
        
        # Extract pose data from humans
        for human in humans:
            pose_data = {
                'keypoints': human['keypoints'],
                'keypoint_names': self.keypoint_names,
                'bbox': human['bbox']
            }
            poses.append(pose_data)
            
            if human['keypoints']:
                all_keypoints.append(human['keypoints'])
            
            # Add pose data to human detection
            human['pose'] = pose_data
        
        return {
            'frame': frame,
            'humans': humans,
            'poses': poses,
            'keypoints': np.array(all_keypoints) if all_keypoints else np.empty((0, 17, 3))
        }
    
    def process_video_frames(self, video_path: str, conf_threshold: float = 0.9) -> List[Dict[str, Any]]:
        """
        Process video and return pose estimations for each frame
        
        Args:
            video_path: Path to video file
            conf_threshold: Confidence threshold for human detection
            
        Returns:
            List of pose estimation dictionaries for each frame
        """
        cap = cv2.VideoCapture(video_path)
        frame_poses = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            pose_data = self.process_frame(frame, conf_threshold)
            frame_poses.append(pose_data)
        
        cap.release()
        return frame_poses
    
    def get_pose_summary(self, frame_poses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics from pose estimations
        
        Args:
            frame_poses: List of frame pose dictionaries
            
        Returns:
            Summary dictionary with statistics
        """
        total_humans = sum(len(frame['humans']) for frame in frame_poses)
        total_frames_with_humans = sum(1 for frame in frame_poses if len(frame['humans']) > 0)
        
        return {
            'total_frames': len(frame_poses),
            'total_humans_detected': total_humans,
            'frames_with_humans': total_frames_with_humans,
            'avg_humans_per_frame': total_humans / len(frame_poses) if frame_poses else 0
        }
