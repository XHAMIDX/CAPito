"""Main pipeline for GET_CAPTION: Object Detection + Mask Generation + Captioning."""

import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging

from PIL import Image
import torch

try:
    # Try relative imports first (when used as module)
    from .config import Config, load_config
    from .utils import setup_logging, load_image, save_image, create_summary_report
    from .detection import DetectionPipeline
    from .captioning import CaptionPipeline
except ImportError:
    # Fall back to absolute imports (when run directly)
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from config import Config, load_config
    from utils import setup_logging, load_image, save_image, create_summary_report
    from detection import DetectionPipeline
    from captioning import CaptionPipeline


class GetCaptionPipeline:
    """Main pipeline for object detection and captioning."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the complete pipeline.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or Config()
        self.logger = setup_logging(self.config)
        
        self.logger.info("Initializing GET_CAPTION Pipeline")
        
        # Initialize detection pipeline
        self.detection_pipeline = DetectionPipeline(
            detection_model=self.config.model.detection_model,
            sam_model=self.config.model.sam_model,
            conf_threshold=self.config.model.detection_conf,
            iou_threshold=self.config.model.detection_iou,
            device=self.config.model.device
        )
        
        # Initialize captioning pipeline
        self.caption_pipeline = CaptionPipeline(
            alpha_clip_model=self.config.model.alpha_clip_model,
            lm_model=self.config.model.lm_model,
            device=self.config.model.device
        )
        
        self.logger.info("Pipeline initialization completed")
    
    def process_single_image(
        self,
        image_path: Union[str, Path, Image.Image],
        output_dir: Optional[str] = None,
        save_intermediate: bool = None
    ) -> Dict[str, Any]:
        """Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to image file or PIL Image
            output_dir: Output directory for results
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Complete processing results
        """
        if save_intermediate is None:
            save_intermediate = self.config.save_intermediate
        
        if output_dir is None:
            output_dir = self.config.processing.output_path
        
        # Load image
        if isinstance(image_path, (str, Path)):
            image = load_image(image_path)
            image_name = Path(image_path).name
        else:
            image = image_path
            image_name = f"image_{int(time.time())}.jpg"
        
        self.logger.info(f"Processing image: {image_name}")
        start_time = time.time()
        
        # Step 1: Object Detection and Segmentation
        self.logger.info("Step 1: Object detection and segmentation")
        detection_results = self.detection_pipeline.process_image(
            image=image,
            min_area=self.config.processing.min_object_area,
            max_objects=self.config.processing.max_objects_per_image,
            mask_threshold=self.config.processing.mask_threshold,
            blur_radius=self.config.processing.mask_blur_radius
        )
        
        # Step 2: Prepare alpha masks for AlphaCLIP
        self.logger.info("Step 2: Preparing alpha masks for AlphaCLIP")
        alpha_masks = self.detection_pipeline.prepare_alpha_masks_for_clip(
            detection_results['alpha_masks']
        )
        
        # Step 3: Generate captions
        self.logger.info("Step 3: Generating captions")
        generation_config = {
            'prompt': self.config.generation.prompt,
            'max_length': self.config.generation.sentence_len,
            'num_iterations': self.config.generation.num_iterations,
            'top_k': self.config.generation.candidate_k,
            'temperature': self.config.generation.lm_temperature,
            'alpha': self.config.generation.alpha,
            'beta': self.config.generation.beta,
            'generation_order': self.config.generation.order
        }
        
        caption_results = self.caption_pipeline.process_image_complete(
            image=image,
            detection_results=alpha_masks,
            generation_config=generation_config,
            include_full_image=True,
            num_samples_per_object=self.config.processing.samples_num
        )
        
        # Combine results
        complete_results = {
            'image_name': image_name,
            'processing_time': time.time() - start_time,
            'detection_results': detection_results,
            'caption_results': caption_results,
            'config': {
                'model': self.config.model.__dict__,
                'generation': self.config.generation.__dict__,
                'processing': self.config.processing.__dict__
            }
        }
        
        # Save results if requested
        if save_intermediate:
            self._save_results(complete_results, output_dir, image_name)
            
            # Save visualization
            if 'visualization' in detection_results:
                vis_path = os.path.join(
                    output_dir, 
                    f"{Path(image_name).stem}_detection_visualization.jpg"
                )
                save_image(detection_results['visualization'], vis_path)
        
        self.logger.info(
            f"Image processing completed in {complete_results['processing_time']:.2f}s"
        )
        
        return complete_results
    
    def process_batch(
        self,
        input_path: Union[str, Path, List[str]],
        output_dir: Optional[str] = None,
        save_intermediate: bool = None
    ) -> List[Dict[str, Any]]:
        """Process multiple images.
        
        Args:
            input_path: Directory path, list of image paths, or glob pattern
            output_dir: Output directory for results
            save_intermediate: Whether to save intermediate results
            
        Returns:
            List of processing results for each image
        """
        if save_intermediate is None:
            save_intermediate = self.config.save_intermediate
        
        if output_dir is None:
            output_dir = self.config.processing.output_path
        
        # Get list of image paths
        if isinstance(input_path, (str, Path)):
            input_path = Path(input_path)
            if input_path.is_dir():
                # Directory: find all images
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
                image_paths = [
                    str(p) for p in input_path.iterdir() 
                    if p.suffix.lower() in image_extensions
                ]
            elif '*' in str(input_path):
                # Glob pattern
                import glob
                image_paths = glob.glob(str(input_path))
            else:
                # Single file
                image_paths = [str(input_path)]
        else:
            # List of paths
            image_paths = input_path
        
        self.logger.info(f"Processing {len(image_paths)} images")
        
        # Process each image
        results = []
        for i, image_path in enumerate(image_paths):
            self.logger.info(f"Processing image {i + 1}/{len(image_paths)}: {image_path}")
            
            try:
                result = self.process_single_image(
                    image_path=image_path,
                    output_dir=output_dir,
                    save_intermediate=save_intermediate
                )
                result['batch_index'] = i
                result['success'] = True
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    'image_name': Path(image_path).name,
                    'batch_index': i,
                    'success': False,
                    'error': str(e),
                    'processing_time': 0.0
                })
        
        # Save batch summary
        if save_intermediate:
            self._save_batch_summary(results, output_dir)
        
        successful = sum(1 for r in results if r.get('success', False))
        self.logger.info(f"Batch processing completed: {successful}/{len(results)} successful")
        
        return results
    
    def _save_results(
        self, 
        results: Dict[str, Any], 
        output_dir: str, 
        image_name: str
    ) -> None:
        """Save processing results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = Path(image_name).stem
        
        # Save complete results as JSON
        results_path = os.path.join(output_dir, f"{base_name}_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save captions as text file
        captions_path = os.path.join(output_dir, f"{base_name}_captions.txt")
        with open(captions_path, 'w', encoding='utf-8') as f:
            f.write(f"Captions for: {image_name}\n")
            f.write("=" * 50 + "\n\n")
            
            # Full image caption
            if results['caption_results'].get('full_image_caption'):
                full_cap = results['caption_results']['full_image_caption']
                if full_cap.get('success'):
                    f.write(f"Full Image: {full_cap['caption']}\n\n")
            
            # Object captions
            f.write("Detected Objects:\n")
            for i, obj_cap in enumerate(results['caption_results'].get('object_captions', [])):
                if obj_cap.get('success'):
                    obj_info = obj_cap['object_info']
                    f.write(f"{i+1}. {obj_info['class_name']} ({obj_info['confidence']:.2f}): ")
                    f.write(f"{obj_cap['caption']}\n")
    
    def _save_batch_summary(self, results: List[Dict[str, Any]], output_dir: str) -> None:
        """Save batch processing summary."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create summary report
        summary_text = create_summary_report(results)
        
        # Save summary
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(output_dir, f"batch_summary_{timestamp}.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        # Save detailed results
        results_path = os.path.join(output_dir, f"batch_results_{timestamp}.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)


def main():
    """Main entry point for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GET_CAPTION: Object Detection + Captioning")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("-o", "--output", default="results", help="Output directory")
    parser.add_argument("-c", "--config", help="Configuration file path")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to use")
    parser.add_argument("--batch", action="store_true", help="Process as batch")
    parser.add_argument("--no-save", action="store_true", help="Don't save intermediate results")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override device if specified
    if args.device:
        config.model.device = args.device
    
    # Override save setting
    if args.no_save:
        config.save_intermediate = False
    
    # Initialize pipeline
    pipeline = GetCaptionPipeline(config)
    
    # Process input
    if args.batch or Path(args.input).is_dir():
        results = pipeline.process_batch(
            input_path=args.input,
            output_dir=args.output
        )
        print(f"Processed {len(results)} images")
    else:
        result = pipeline.process_single_image(
            image_path=args.input,
            output_dir=args.output
        )
        print(f"Processing completed in {result['processing_time']:.2f}s")


if __name__ == "__main__":
    main()
