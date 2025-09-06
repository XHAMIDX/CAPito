import os
import json
import numpy as np
import cv2
import sys
import argparse
import torch

# Force use of GPU 8 and 9
os.environ['CUDA_VISIBLE_DEVICES'] = '8,9'
torch.cuda.set_device(0)  # This will be GPU 8 since we set visible devices
print(f"Using GPU 8 (visible as GPU 0)")

def convert_numpy_to_json(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-serializable format"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_json(item) for item in obj]
    else:
        return obj

from Object_detector import ObjectDetector
from Pose_estimator import PoseEstimator
from Depth_estimator import DepthEstimator
from Tracker import DeepSORTTracker

class DetectionPipeline:
    def __init__(self, object_model='yolo11n.pt'):
        # Force GPU 8 (visible as GPU 0)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device} (GPU 8)")
        
        self.object_detector = ObjectDetector(object_model)
        self.pose_estimator = PoseEstimator(device=device)
        self.depth_estimator = DepthEstimator(device=device)
        self.tracker = DeepSORTTracker(object_model)
        
        # Create RESULTS folder
        os.makedirs('RESULTS', exist_ok=True)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            device_id = torch.cuda.current_device()
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(device_id)/1024**3:.2f} GB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved(device_id)/1024**3:.2f} GB")
    
    def process_video(self, video_path: str, conf_threshold: float = 0.3, sample_rate: int = 1, start_time: float = None, end_time: float = None):
        """
        Process video with object detection, tracking, pose estimation, and depth estimation
        
        Args:
            video_path: Path to input video
            conf_threshold: Confidence threshold for detections
            sample_rate: Process every Nth frame (1 = all frames)
            start_time: Start time in seconds (None = from beginning)
            end_time: End time in seconds (None = to end)
        """
        print(f"Processing video: {video_path}")
        print(f"Sample rate: every {sample_rate} frame(s)")
        if start_time is not None and end_time is not None:
            print(f"Time segment: {start_time}s to {end_time}s")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video")
            return None
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {total_frames} frames, {fps} FPS, {width}x{height}")
        
        # Calculate frame range for time segment
        start_frame = 0
        end_frame = total_frames
        
        if start_time is not None:
            start_frame = int(start_time * fps)
        if end_time is not None:
            end_frame = int(end_time * fps)
        
        # Ensure valid frame range
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))
        
        print(f"Processing frames {start_frame} to {end_frame} (out of {total_frames})")
        
        # Initialize output video writer
        output_path = os.path.join('RESULTS', f'{os.path.splitext(os.path.basename(video_path))[0]}_output.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        processed_frames = []
        
        # Skip to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        while cap.isOpened() and frame_count < (end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame
            if frame_count % sample_rate == 0:
                actual_frame_number = start_frame + frame_count
                print(f"Processing frame {actual_frame_number + 1}/{total_frames} (time: {actual_frame_number/fps:.1f}s)")
                
                # Step 1: Object Detection
                object_data = self.object_detector.detect_frame(frame, conf_threshold)
                
                # Step 2: Object Tracking (for all sequence)
                tracked_objects = self.tracker.update(object_data['detections'])
                
                # Step 3: Process all tracked objects for depth and pose estimation
                objects_with_depth = []
                poses = []
                all_keypoints = []
                
                # Extract humans for pose estimation
                humans = [obj for obj in tracked_objects if obj['class_name'] == 'person']
                
                # Process each tracked object
                for obj in tracked_objects:
                    # Add depth information to all objects
                    depth_stats = self.depth_estimator.get_object_depth(
                        self.depth_estimator.estimate_depth(frame), 
                        obj['bbox']
                    )
                    
                    obj_with_depth = obj.copy()
                    obj_with_depth['depth'] = depth_stats
                    objects_with_depth.append(obj_with_depth)
                    
                    # Add pose estimation for humans only
                    if obj['class_name'] == 'person':
                        pose_data = self.pose_estimator.estimate_pose(frame, obj['bbox'])
                        poses.append(pose_data)
                        if pose_data['keypoints']:
                            all_keypoints.append(pose_data['keypoints'])
                
                # Step 4: Combine results
                frame_data = {
                    'frame_number': actual_frame_number,
                    'time_stamp': actual_frame_number / fps,
                    'original_frame': frame,
                    'tracked_objects': tracked_objects,
                    'objects_with_depth': objects_with_depth,
                    'human_detections': humans,
                    'pose_estimations': poses,
                    'all_keypoints': all_keypoints
                }
                
                processed_frames.append(frame_data)
                
                # Step 5: Visualize and write to output video
                vis_frame = self.visualize_frame(frame_data)
                out.write(vis_frame)
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"Video processing completed. Output saved to: {output_path}")
        return processed_frames
    
    def visualize_frame(self, frame_data: dict):
        """
        Visualize detection results on a single frame
        """
        vis_image = frame_data['original_frame'].copy()
        
        # Colors for different object types
        colors = {
            'person': (0, 255, 0),      # Green for humans
            'car': (255, 0, 0),         # Blue for cars
            'bicycle': (0, 0, 255),       # Red for chairs
            'default': (255, 255, 0)    # Yellow for others
        }
        
        # Draw tracked objects with depth information
        for obj in frame_data['objects_with_depth']:
            bbox = obj['bbox']
            class_name = obj['class_name']
            confidence = obj['confidence']
            track_id = obj.get('track_id', 'N/A')
            depth_info = obj['depth']
            
            # Get color for this class
            color = colors.get(class_name, colors['default'])
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with tracking and depth information
            if depth_info['valid_pixels'] > 0:
                label = f"ID:{track_id} {class_name}: {confidence:.2f} | {depth_info['mean_depth']:.1f}m"
            else:
                label = f"ID:{track_id} {class_name}: {confidence:.2f} | No depth"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw pose keypoints for humans
        for i, pose in enumerate(frame_data['pose_estimations']):
            if pose['keypoints']:
                keypoints = np.array(pose['keypoints'])
                
                # Draw keypoints
                for j, (x, y, conf) in enumerate(keypoints):
                    if conf > 0.5:
                        cv2.circle(vis_image, (int(x), int(y)), 3, (0, 255, 255), -1)
                
                # Draw skeleton connections
                skeleton_connections = [
                    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
                    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)
                ]
                
                for start_idx, end_idx in skeleton_connections:
                    if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                        keypoints[start_idx][2] > 0.5 and keypoints[end_idx][2] > 0.5):
                        start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                        end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                        cv2.line(vis_image, start_point, end_point, (255, 0, 255), 2)
        
        return vis_image
    
    def save_video_metadata(self, processed_frames: list, video_name: str):
        """
        Save comprehensive metadata for video processing including all component data and temporal pose tracks
        """
        # Prepare comprehensive metadata
        metadata = {
            'video_name': video_name,
            'total_frames_processed': len(processed_frames),
            'time_range': {
                'start_time': processed_frames[0]['time_stamp'] if processed_frames else 0.0,
                'end_time': processed_frames[-1]['time_stamp'] if processed_frames else 0.0,
                'duration': (processed_frames[-1]['time_stamp'] - processed_frames[0]['time_stamp']) if len(processed_frames) > 1 else 0.0
            },
            'processing_components': {
                'object_detection': 'YOLO-based detection with bounding boxes and confidence scores',
                'pose_estimation': 'YOLO pose estimation for humans with 17 keypoints',
                'depth_estimation': 'MiDaS depth estimation for all objects',
                'tracking': 'DeepSORT tracking with unique IDs across frames'
            },
            'frames_data': [],
            'temporal_pose_tracks': {}
        }
        
        # Build temporal pose tracks
        pose_tracks = {}
        for frame_data in processed_frames:
            frame_number = frame_data['frame_number']
            time_stamp = frame_data['time_stamp']
            for human, pose in zip(frame_data['human_detections'], frame_data['pose_estimations']):
                track_id = human.get('track_id', None)
                if track_id is None:
                    continue
                pose_entry = {
                    'frame_number': frame_number,
                    'time_stamp': time_stamp,
                    'bbox': human['bbox'],
                    'confidence': human['confidence'],
                    'keypoints': pose['keypoints'],
                    'keypoint_names': pose['keypoint_names']
                }
                if str(track_id) not in pose_tracks:
                    pose_tracks[str(track_id)] = []
                pose_tracks[str(track_id)].append(pose_entry)
        metadata['temporal_pose_tracks'] = pose_tracks
        
        for frame_data in processed_frames:
            # Comprehensive frame metadata
            frame_metadata = {
                'frame_info': {
                    'frame_number': frame_data['frame_number'],
                    'time_stamp': frame_data['time_stamp'],
                    'frame_time_seconds': frame_data['time_stamp']
                },
                # Object Detection Data
                'object_detection': {
                    'total_objects': len(frame_data['tracked_objects']),
                    'objects': [
                        {
                            'track_id': obj.get('track_id', 'N/A'),
                            'class_name': obj['class_name'],
                            'confidence': obj['confidence'],
                            'bbox': obj['bbox'],
                            'bbox_coordinates': {
                                'x1': obj['bbox'][0],
                                'y1': obj['bbox'][1],
                                'x2': obj['bbox'][2],
                                'y2': obj['bbox'][3],
                                'width': obj['bbox'][2] - obj['bbox'][0],
                                'height': obj['bbox'][3] - obj['bbox'][1]
                            }
                        } for obj in frame_data['tracked_objects']
                    ]
                },
                # Tracking Data
                'tracking': {
                    'active_tracks': len(frame_data['tracked_objects']),
                    'tracked_objects': [
                        {
                            'track_id': obj.get('track_id', 'N/A'),
                            'class_name': obj['class_name'],
                            'confidence': obj['confidence'],
                            'bbox': obj['bbox'],
                            'track_age': obj.get('age', 0),
                            'track_hits': obj.get('hits', 0),
                            'total_hits': obj.get('total_hits', 0)
                        } for obj in frame_data['tracked_objects']
                    ]
                },
                # Pose Estimation Data (for humans only)
                'pose_estimation': {
                    'total_humans': len(frame_data['human_detections']),
                    'poses_estimated': len(frame_data['pose_estimations']),
                    'human_poses': [
                        {
                            'track_id': human.get('track_id', 'N/A'),
                            'bbox': human['bbox'],
                            'confidence': human['confidence'],
                            'keypoints': pose['keypoints'],
                            'keypoint_names': pose['keypoint_names'],
                            'keypoint_count': len(pose['keypoints']) if pose['keypoints'] else 0,
                            'keypoint_details': [
                                {
                                    'name': pose['keypoint_names'][i] if i < len(pose['keypoint_names']) else f'keypoint_{i}',
                                    'x': kp[0],
                                    'y': kp[1],
                                    'confidence': kp[2]
                                } for i, kp in enumerate(pose['keypoints']) if pose['keypoints']
                            ]
                        } for human, pose in zip(frame_data['human_detections'], frame_data['pose_estimations'])
                    ]
                },
                # Depth Estimation Data (for all objects)
                'depth_estimation': {
                    'objects_with_depth': len([obj for obj in frame_data['objects_with_depth'] if obj['depth']['valid_pixels'] > 0]),
                    'objects_depth_data': [
                        {
                            'track_id': obj.get('track_id', 'N/A'),
                            'class_name': obj['class_name'],
                            'bbox': obj['bbox'],
                            'depth_statistics': {
                                'min_depth_meters': obj['depth']['min_depth'],
                                'max_depth_meters': obj['depth']['max_depth'],
                                'mean_depth_meters': obj['depth']['mean_depth'],
                                'median_depth_meters': obj['depth']['median_depth'],
                                'depth_std_meters': obj['depth']['depth_std'],
                                'depth_range_meters': obj['depth']['depth_range'],
                                'valid_pixels': obj['depth']['valid_pixels'],
                                'depth_quality': 'good' if obj['depth']['valid_pixels'] > 100 else 'poor'
                            }
                        } for obj in frame_data['objects_with_depth']
                    ]
                },
                # Summary statistics for this frame
                'frame_summary': {
                    'total_objects': len(frame_data['tracked_objects']),
                    'humans_detected': len(frame_data['human_detections']),
                    'poses_estimated': len(frame_data['pose_estimations']),
                    'objects_with_valid_depth': len([obj for obj in frame_data['objects_with_depth'] if obj['depth']['valid_pixels'] > 0]),
                    'unique_track_ids': len(set(obj.get('track_id', 'N/A') for obj in frame_data['tracked_objects']))
                }
            }
            metadata['frames_data'].append(frame_metadata)
        
        # Save comprehensive metadata to JSON file
        output_path = os.path.join('RESULTS', f'{video_name}_detailed_metadata.json')
        with open(output_path, 'w') as f:
            json.dump(convert_numpy_to_json(metadata), f, indent=2)
        
        print(f"Detailed metadata saved to: {output_path}")
        
        # Save summary statistics
        summary = self.generate_video_summary(processed_frames)
        summary_path = os.path.join('RESULTS', f'{video_name}_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(convert_numpy_to_json(summary), f, indent=2)
        
        print(f"Video summary saved to: {summary_path}")
    
    def generate_video_summary(self, processed_frames: list) -> dict:
        """
        Generate comprehensive summary statistics for video
        """
        total_objects = sum(len(frame['tracked_objects']) for frame in processed_frames)
        total_humans = sum(len(frame['human_detections']) for frame in processed_frames)
        total_poses = sum(len(frame['pose_estimations']) for frame in processed_frames)
        
        # Track statistics
        all_track_ids = set()
        track_lifespans = {}
        for frame in processed_frames:
            for obj in frame['tracked_objects']:
                if 'track_id' in obj:
                    track_id = obj['track_id']
                    all_track_ids.add(track_id)
                    if track_id not in track_lifespans:
                        track_lifespans[track_id] = {'first_frame': frame['frame_number'], 'last_frame': frame['frame_number']}
                    else:
                        track_lifespans[track_id]['last_frame'] = frame['frame_number']
        
        # Object class distribution
        class_counts = {}
        class_confidence_avg = {}
        for frame in processed_frames:
            for obj in frame['tracked_objects']:
                class_name = obj['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                if class_name not in class_confidence_avg:
                    class_confidence_avg[class_name] = []
                class_confidence_avg[class_name].append(obj['confidence'])
        
        # Calculate average confidence per class
        for class_name in class_confidence_avg:
            class_confidence_avg[class_name] = float(np.mean(class_confidence_avg[class_name]))
        
        # Depth summary
        all_depths = []
        depth_by_class = {}
        for frame in processed_frames:
            for obj in frame['objects_with_depth']:
                if obj['depth']['valid_pixels'] > 0:
                    depth = obj['depth']['mean_depth']
                    all_depths.append(depth)
                    class_name = obj['class_name']
                    if class_name not in depth_by_class:
                        depth_by_class[class_name] = []
                    depth_by_class[class_name].append(depth)
        
        # Pose statistics
        pose_keypoint_stats = []
        for frame in processed_frames:
            for pose in frame['pose_estimations']:
                if pose['keypoints']:
                    pose_keypoint_stats.append(len(pose['keypoints']))
        
        # Calculate track lifespans
        track_lifespan_stats = []
        for track_id, lifespan in track_lifespans.items():
            lifespan_frames = lifespan['last_frame'] - lifespan['first_frame'] + 1
            track_lifespan_stats.append(lifespan_frames)
        
        return {
            'video_processing_summary': {
                'total_frames_processed': len(processed_frames),
                'time_range_seconds': {
                    'start': processed_frames[0]['time_stamp'] if processed_frames else 0.0,
                    'end': processed_frames[-1]['time_stamp'] if processed_frames else 0.0,
                    'duration': (processed_frames[-1]['time_stamp'] - processed_frames[0]['time_stamp']) if len(processed_frames) > 1 else 0.0
                }
            },
            
            'object_detection_summary': {
                'total_objects_detected': total_objects,
                'avg_objects_per_frame': total_objects / len(processed_frames) if processed_frames else 0,
                'object_class_distribution': class_counts,
                'class_confidence_averages': class_confidence_avg
            },
            
            'tracking_summary': {
                'unique_tracks': len(all_track_ids),
                'track_ids': list(all_track_ids),
                'track_lifespan_statistics': {
                    'avg_lifespan_frames': float(np.mean(track_lifespan_stats)) if track_lifespan_stats else 0.0,
                    'min_lifespan_frames': float(np.min(track_lifespan_stats)) if track_lifespan_stats else 0.0,
                    'max_lifespan_frames': float(np.max(track_lifespan_stats)) if track_lifespan_stats else 0.0,
                    'total_track_frames': sum(track_lifespan_stats)
                }
            },
            
            'pose_estimation_summary': {
                'total_humans_detected': total_humans,
                'total_poses_estimated': total_poses,
                'avg_humans_per_frame': total_humans / len(processed_frames) if processed_frames else 0,
                'pose_success_rate': (total_poses / total_humans * 100) if total_humans > 0 else 0,
                'keypoint_statistics': {
                    'avg_keypoints_per_pose': float(np.mean(pose_keypoint_stats)) if pose_keypoint_stats else 0.0,
                    'min_keypoints_per_pose': float(np.min(pose_keypoint_stats)) if pose_keypoint_stats else 0.0,
                    'max_keypoints_per_pose': float(np.max(pose_keypoint_stats)) if pose_keypoint_stats else 0.0
                }
            },
            
            'depth_estimation_summary': {
                'objects_with_valid_depth': len(all_depths),
                'depth_success_rate': (len(all_depths) / total_objects * 100) if total_objects > 0 else 0,
                'overall_depth_statistics': {
                    'avg_depth_meters': float(np.mean(all_depths)) if all_depths else 0.0,
                    'min_depth_meters': float(np.min(all_depths)) if all_depths else 0.0,
                    'max_depth_meters': float(np.max(all_depths)) if all_depths else 0.0,
                    'depth_std_meters': float(np.std(all_depths)) if all_depths else 0.0
                },
                'depth_by_class': {
                    class_name: {
                        'count': len(depths),
                        'avg_depth_meters': float(np.mean(depths)),
                        'min_depth_meters': float(np.min(depths)),
                        'max_depth_meters': float(np.max(depths))
                    } for class_name, depths in depth_by_class.items()
                }
            },
            
            'performance_metrics': {
                'detection_density': total_objects / len(processed_frames) if processed_frames else 0,
                'human_detection_rate': (total_humans / total_objects * 100) if total_objects > 0 else 0,
                'pose_estimation_rate': (total_poses / total_humans * 100) if total_humans > 0 else 0,
                'depth_estimation_rate': (len(all_depths) / total_objects * 100) if total_objects > 0 else 0
            }
        }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Video Analysis Pipeline with Object Detection, Tracking, Pose Estimation, and Depth Estimation')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('--sample_rate', type=int, default=1, help='Sample rate (1=all frames, 2=every 2nd frame, etc.)')
    parser.add_argument('--start_time', type=float, default=None, help='Start time in seconds (e.g., 10.0 for 10 seconds)')
    parser.add_argument('--end_time', type=float, default=None, help='End time in seconds (e.g., 11.0 for 11 seconds)')
    parser.add_argument('--conf_threshold', type=float, default=0.3, help='Confidence threshold for detections (default: 0.3)')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
        return
    
    # Initialize pipeline
    pipeline = DetectionPipeline()
    
    # Get video name for output files
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    
    print(f"Processing video: {args.video_path}")
    print(f"Sample rate: {args.sample_rate}")
    print(f"Confidence threshold: {args.conf_threshold}")
    if args.start_time is not None and args.end_time is not None:
        print(f"Time segment: {args.start_time}s to {args.end_time}s")
    
    # Process video
    processed_frames = pipeline.process_video(
        args.video_path, 
        conf_threshold=args.conf_threshold, 
        sample_rate=args.sample_rate, 
        start_time=args.start_time, 
        end_time=args.end_time
    )
    
    if processed_frames is not None:
        # Save metadata
        pipeline.save_video_metadata(processed_frames, video_name)
        
        print("\nVideo processing completed!")
        print(f"Processed {len(processed_frames)} frames")
        print(f"Output video saved to RESULTS folder")
    else:
        print("Video processing failed!")

if __name__ == "__main__":
    main() 