"""
CAPito Main Pipeline
===================

The main orchestration class for the CAPito system.
Coordinates object detection, segmentation, and captioning.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import torch
from PIL import Image
import numpy as np
import networkx as nx

from .config import CapitoConfig
from ..vlm.alpha_clip import AlphaCLIPWrapper
from ..detection.detector import ObjectDetector
from ..detection.segmentation import SAM2Segmentator
from ..captioning.generator import CaptionGenerator
from ..analysis.depth_estimator import DepthEstimator
from ..analysis.pose_estimator import PoseEstimator
from ..analysis.tracker import ObjectTracker
from ..graph.graph_builder import SceneGraphBuilder
from ..graph.graph_analyzer import GraphAnalyzer
from ..utils.logger import setup_logging
from ..utils.image_utils import load_image, save_image
from ..utils.model_manager import ModelManager


class CAPito:
    """
    Main CAPito pipeline for content-aware image captioning.
    
    Combines object detection, segmentation, and vision-language modeling
    to generate intelligent, contextual captions for images.
    """
    
    def __init__(self, config: CapitoConfig):
        """
        Initialize the CAPito pipeline.
        
        Args:
            config: Configuration object containing all settings
        """
        self.config = config
        self.logger = setup_logging(config.system.log_level)
        self.model_manager = ModelManager(config.model_paths.models_root)
        
        self.logger.info("Initializing CAPito pipeline...")
        
        # Initialize components
        self._init_vlm()
        self._init_detection()
        self._init_segmentation()
        self._init_captioning()
        self._init_analysis()
        self._init_graph()
        
        self.logger.info("CAPito pipeline initialized successfully")
    
    def _init_vlm(self) -> None:
        """Initialize AlphaCLIP vision-language model."""
        self.logger.info(f"Loading VLM: {self.config.vlm.model_name}")
        model_path = self.config.get_model_path("vlm", self.config.vlm.model_name)
        
        self.vlm = AlphaCLIPWrapper(
            model_name=self.config.vlm.model_name,
            model_path=model_path,
            device=self.config.vlm.device
        )
    
    def _init_detection(self) -> None:
        """Initialize object detection model."""
        self.logger.info(f"Loading detection model: {self.config.detection.model_name}")
        model_path = self.config.get_model_path("detection", self.config.detection.model_name)
        
        self.detector = ObjectDetector(
            model_path=model_path,
            confidence_threshold=self.config.detection.confidence_threshold,
            iou_threshold=self.config.detection.iou_threshold,
            device=self.config.detection.device
        )
    
    def _init_segmentation(self) -> None:
        """Initialize SAM2 segmentation model."""
        self.logger.info(f"Loading segmentation model: {self.config.segmentation.model_name}")
        model_path = self.config.get_model_path("segmentation", self.config.segmentation.model_name)
        
        self.segmentator = SAM2Segmentator(
            model_path=model_path,
            device=self.config.segmentation.device
        )
    
    def _init_captioning(self) -> None:
        """Initialize caption generation model."""
        self.logger.info("Loading caption generation models...")
        
        # Get language model path for controllable generation
        lm_path = self.config.get_model_path("language", self.config.captioning.language_model)
        
        self.caption_generator = CaptionGenerator(
            vlm=self.vlm,
            language_model_path=lm_path,
            config=self.config.captioning,
            device=self.config.system.device
        )
    
    def _init_analysis(self) -> None:
        """Initialize analysis components (depth, pose, tracking)."""
        self.logger.info("Loading analysis models...")
        
        # Initialize depth estimator
        if self.config.analysis.enable_depth:
            self.depth_estimator = DepthEstimator(device=self.config.analysis.device)
        else:
            self.depth_estimator = None
        
        # Initialize pose estimator
        if self.config.analysis.enable_pose:
            pose_model_path = self.config.get_model_path("analysis", self.config.analysis.pose_model)
            self.pose_estimator = PoseEstimator(
                model_path=pose_model_path,
                device=self.config.analysis.device
            )
        else:
            self.pose_estimator = None
        
        # Initialize object tracker
        if self.config.analysis.enable_tracking:
            self.tracker = ObjectTracker(
                max_disappeared=self.config.analysis.tracking_max_disappeared,
                max_distance=self.config.analysis.tracking_max_distance
            )
        else:
            self.tracker = None
    
    def _init_graph(self) -> None:
        """Initialize graph generation components."""
        if not self.config.graph.enable_graph:
            self.graph_builder = None
            self.graph_analyzer = None
            return
        
        self.logger.info("Loading graph generation models...")
        
        # Initialize graph builder
        self.graph_builder = SceneGraphBuilder(
            similarity_model=self.config.graph.similarity_model,
            device=self.config.graph.device
        )
        
        # Initialize graph analyzer
        self.graph_analyzer = GraphAnalyzer()
    
    def process_image(
        self, 
        image_path: Union[str, Path], 
        output_dir: Optional[str] = None,
        save_visualizations: Optional[bool] = None,
        save_results: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save outputs (optional)
            save_visualizations: Whether to save visualization images
            save_results: Whether to save JSON results
            
        Returns:
            Dictionary containing all processing results
        """
        # Set default values from config if not provided
        if save_visualizations is None:
            save_visualizations = self.config.system.save_visualizations
        if save_results is None:
            save_results = self.config.system.save_results
        if output_dir is None:
            output_dir = self.config.system.output_dir
        
        start_time = time.time()
        image_path = Path(image_path)
        
        self.logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = load_image(str(image_path))
        
        # Step 1: Object Detection
        self.logger.info("Running object detection...")
        detections = self.detector.detect(image)
        
        # Step 2: Segmentation
        self.logger.info("Generating segmentation masks...")
        masks = self.segmentator.segment(image, detections)
        
        # Step 3: Enhanced Analysis
        self.logger.info("Running enhanced analysis...")
        
        # Depth estimation
        depth_map = None
        if self.depth_estimator:
            depth_map = self.depth_estimator.estimate_depth(image)
        
        # Pose estimation for humans
        human_poses = []
        if self.pose_estimator:
            human_poses = self.pose_estimator.estimate_poses(image)
        
        # Object tracking (convert detections to dict format for tracker)
        tracked_objects = detections  # Will be enhanced for video
        if self.tracker:
            detection_dicts = [det.to_dict() for det in detections]
            tracked_detection_dicts = self.tracker.update(detection_dicts)
            # Convert back to Detection objects if needed
            tracked_objects = detections  # Keep original for now
        
        # Step 4: Caption Generation with Enhanced Features
        self.logger.info("Generating captions...")
        captions = []
        enhanced_objects = []
        
        # Generate one caption per detected object with enhanced features
        for i, detection in enumerate(detections):
            try:
                # Get corresponding mask if available
                mask_obj = masks[i] if i < len(masks) else None
                mask_data = mask_obj.mask if mask_obj else None
                
                # Normalize bounding box coordinates
                bbox_normalized = self._normalize_bbox(detection.bbox, image.size)
                area_normalized = self._normalize_area(detection.area, image.size)
                
                # Add depth information
                depth_value = 0.5  # Default
                if depth_map is not None and self.depth_estimator:
                    depth_value = self.depth_estimator.get_object_depth(
                        depth_map, 
                        bbox_normalized,
                        image.size
                    )
                
                # Add pose information for humans
                pose_data = None
                if detection.class_name == 'person' and self.pose_estimator:
                    pose_data = self.pose_estimator.get_pose_for_human(
                        image, 
                        list(bbox_normalized)
                    )
                
                # Create enhanced object data
                enhanced_object = {
                    'bbox': bbox_normalized,
                    'class_name': detection.class_name,
                    'confidence': detection.confidence,
                    'area': area_normalized,
                    'depth': depth_value,
                    'pose': pose_data,
                    'track_id': getattr(detection, 'track_id', -1)
                }
                
                # Generate caption for this specific object
                object_caption = self.caption_generator.generate_for_object(
                    image=image,
                    detection=detection,
                    mask=mask_data
                )
                
                enhanced_object['caption'] = object_caption
                enhanced_objects.append(enhanced_object)
                captions.append(object_caption)
                
            except Exception as e:
                self.logger.warning(f"Failed to process object {i}: {e}")
                # Add fallback object
                bbox_normalized = self._normalize_bbox(detection.bbox, image.size)
                area_normalized = self._normalize_area(detection.area, image.size)
                
                enhanced_objects.append({
                    'bbox': bbox_normalized,
                    'class_name': detection.class_name,
                    'confidence': detection.confidence,
                    'area': area_normalized,
                    'depth': 0.5,
                    'pose': None,
                    'track_id': -1,
                    'caption': f"A {detection.class_name}"
                })
                captions.append(f"A {detection.class_name}")
        
        # Step 5: Scene Graph Generation
        scene_graph = None
        graph_analysis = None
        scene_summary = ""
        
        if self.graph_builder and enhanced_objects:
            self.logger.info("Building scene graph...")
            
            scene_graph = self.graph_builder.build_graph(
                enhanced_objects,
                image.size,
                self.config.graph.similarity_threshold,
                self.config.graph.distance_threshold
            )
            
            if self.graph_analyzer:
                graph_analysis = self.graph_analyzer.analyze_graph(scene_graph)
                scene_summary = self.graph_analyzer.generate_scene_summary(scene_graph)
        
        # If no objects detected, generate general scene caption
        if not captions:
            scene_captions = self.caption_generator.generate(
                image=image,
                detections=detections,
                masks=masks,
                num_captions=1
            )
            captions = scene_captions
        
        # Compile results
        processing_time = time.time() - start_time
        results = {
            "image_path": str(image_path),
            "processing_time": processing_time,
            "detections": detections,
            "masks": masks,
            "captions": captions,
            "enhanced_objects": enhanced_objects,
            "scene_graph": nx.node_link_data(scene_graph) if scene_graph is not None else None,
            "graph_analysis": graph_analysis,
            "scene_summary": scene_summary,
            "depth_map": depth_map.tolist() if depth_map is not None else None,
            "human_poses": human_poses,
            "metadata": {
                "config": self._get_config_summary(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "image_size": image.size,
                "num_objects": len(enhanced_objects),
                "has_graph": scene_graph is not None,
                "has_depth": depth_map is not None,
                "num_humans": len([obj for obj in enhanced_objects if obj['class_name'] == 'person'])
            }
        }
        
        # Save outputs if requested
        if save_results or save_visualizations:
            self._save_outputs(
                results, 
                image, 
                image_path.stem, 
                output_dir,
                save_visualizations,
                save_results
            )
        
        self.logger.info(f"Processing completed in {processing_time:.2f}s")
        return results
    
    def process_batch(
        self, 
        image_paths: List[Union[str, Path]], 
        output_dir: Optional[str] = None,
        save_visualizations: Optional[bool] = None,
        save_results: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of paths to input images
            output_dir: Directory to save outputs
            save_visualizations: Whether to save visualization images
            save_results: Whether to save JSON results
            
        Returns:
            List of dictionaries containing results for each image
        """
        self.logger.info(f"Processing batch of {len(image_paths)} images")
        
        results = []
        for i, image_path in enumerate(image_paths):
            self.logger.info(f"Processing image {i+1}/{len(image_paths)}")
            try:
                result = self.process_image(
                    image_path=image_path,
                    output_dir=output_dir,
                    save_visualizations=save_visualizations,
                    save_results=save_results
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    "image_path": str(image_path),
                    "error": str(e),
                    "processing_time": 0
                })
        
        # Save batch summary
        if save_results and output_dir:
            self._save_batch_summary(results, output_dir)
        
        return results
    
    def _save_outputs(
        self,
        results: Dict[str, Any],
        image: Image.Image,
        image_name: str,
        output_dir: str,
        save_visualizations: bool,
        save_results: bool
    ) -> None:
        """Save processing outputs to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON results
        if save_results:
            results_path = os.path.join(output_dir, f"{image_name}_results.json")
            with open(results_path, 'w') as f:
                # Make results JSON serializable
                json_results = self._make_json_serializable(results)
                json.dump(json_results, f, indent=2)
            
            # Save captions as text file
            captions_path = os.path.join(output_dir, f"{image_name}_captions.txt")
            with open(captions_path, 'w') as f:
                for caption in results["captions"]:
                    f.write(f"{caption}\n")
        
        # Save visualizations
        if save_visualizations:
            vis_image = self._create_visualization(image, results)
            vis_path = os.path.join(output_dir, f"{image_name}_visualization.jpg")
            save_image(vis_image, vis_path)
    
    def _save_batch_summary(self, results: List[Dict[str, Any]], output_dir: str) -> None:
        """Save batch processing summary."""
        summary = {
            "total_images": len(results),
            "successful": len([r for r in results if "error" not in r]),
            "failed": len([r for r in results if "error" in r]),
            "total_processing_time": sum(r.get("processing_time", 0) for r in results),
            "average_processing_time": sum(r.get("processing_time", 0) for r in results) / len(results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": self._make_json_serializable(results)
        }
        
        summary_path = os.path.join(output_dir, "batch_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _create_visualization(self, image: Image.Image, results: Dict[str, Any]) -> Image.Image:
        """Create visualization with detections and captions."""
        from PIL import ImageDraw, ImageFont
        import random
        
        # Create a copy of the image to draw on
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Define colors for different classes
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (255, 192, 203), # Pink
            (165, 42, 42),  # Brown
        ]
        
        # Draw bounding boxes and labels for each detection
        for i, detection in enumerate(results.get("detections", [])):
            # Get bounding box coordinates
            x1, y1, x2, y2 = detection.bbox
            
            # Choose color based on class_id
            color = colors[detection.class_id % len(colors)]
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Create label text
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            
            # Draw label background
            if font:
                bbox = draw.textbbox((x1, y1-25), label, font=font)
                draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
                draw.text((x1, y1-25), label, fill=(255, 255, 255), font=font)
            else:
                # Fallback without font
                draw.rectangle([x1, y1-20, x1+len(label)*8, y1], fill=color)
                draw.text((x1+2, y1-18), label, fill=(255, 255, 255))
        
        return vis_image
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.generic):
            # Catch any other numpy types
            return obj.item()
        elif type(obj).__module__ == 'numpy':
            # Catch numpy types that may not inherit from np.generic
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                return float(obj) if hasattr(obj, '__float__') else str(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj
    
    def _get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        return {
            "vlm_model": self.config.vlm.model_name,
            "detection_model": self.config.detection.model_name,
            "segmentation_model": self.config.segmentation.model_name,
            "captioning": {
                "max_length": self.config.captioning.max_length,
                "generation_order": self.config.captioning.generation_order,
                "enable_control": self.config.captioning.enable_control
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "vlm": self.vlm.get_model_info(),
            "detector": self.detector.get_model_info(),
            "segmentator": self.segmentator.get_model_info(),
            "caption_generator": self.caption_generator.get_model_info()
        }
    
    def update_config(self, new_config: CapitoConfig) -> None:
        """Update configuration and reload models if necessary."""
        # This would implement hot-swapping of configuration
        # For now, require reinitialization
        raise NotImplementedError("Configuration updates require reinitialization")
    
    def _normalize_bbox(self, bbox: Tuple[float, float, float, float], image_size: Tuple[int, int]) -> List[float]:
        """Normalize bounding box coordinates to [0, 1] range."""
        width, height = image_size
        x1, y1, x2, y2 = bbox
        return [x1 / width, y1 / height, x2 / width, y2 / height]
    
    def _normalize_area(self, area: float, image_size: Tuple[int, int]) -> float:
        """Normalize area to [0, 1] range based on image size."""
        width, height = image_size
        total_area = width * height
        return area / total_area
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("Cleaning up CAPito pipeline...")
        
        # Clean up models if they have cleanup methods
        components = [
            self.vlm, self.detector, self.segmentator, self.caption_generator,
            getattr(self, 'depth_estimator', None),
            getattr(self, 'pose_estimator', None),
            getattr(self, 'graph_builder', None)
        ]
        
        for component in components:
            if component and hasattr(component, 'cleanup'):
                component.cleanup()
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
