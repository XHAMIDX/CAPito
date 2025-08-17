"""Caption generation pipeline combining AlphaCLIP and text generation."""

from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
import logging
import torch

try:
    from .alpha_clip_wrapper import AlphaCLIPWrapper
    from .text_generator import TextGenerator
    from ..utils.text_utils import format_caption
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from alpha_clip_wrapper import AlphaCLIPWrapper
    from text_generator import TextGenerator
    from utils.text_utils import format_caption


class CaptionPipeline:
    """Complete caption generation pipeline for masked objects."""
    
    def __init__(
        self,
        alpha_clip_model: str = "ViT-L/14",
        lm_model: str = "bert-base-uncased",
        device: str = "cpu"
    ):
        """Initialize caption pipeline.
        
        Args:
            alpha_clip_model: AlphaCLIP model name
            lm_model: Language model name
            device: Device to run inference on
        """
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize AlphaCLIP wrapper
        self.clip_wrapper = AlphaCLIPWrapper(
            model_name=alpha_clip_model,
            device=device
        )
        
        # Initialize text generator
        self.text_generator = TextGenerator(
            lm_model_name=lm_model,
            clip_wrapper=self.clip_wrapper,
            device=device
        )
        
        self.logger.info("Caption pipeline initialized")
    
    def caption_object(
        self,
        image: Image.Image,
        alpha_mask: torch.Tensor,
        object_info: Dict[str, Any],
        generation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate caption for a single masked object.
        
        Args:
            image: PIL Image
            alpha_mask: Alpha mask tensor for the object
            object_info: Object detection information
            generation_config: Optional generation parameters
            
        Returns:
            Dictionary with caption results
        """
        if generation_config is None:
            generation_config = {}
        
        # Default generation parameters
        default_config = {
            'prompt': 'A photo of',
            'max_length': 8,
            'num_iterations': 15,
            'top_k': 50,
            'temperature': 0.3,
            'alpha': 0.8,
            'beta': 1.5,
            'generation_order': 'shuffle'
        }
        default_config.update(generation_config)
        
        # Generate caption
        try:
            caption, score = self.text_generator.generate_caption(
                image=image,
                alpha_mask=alpha_mask,
                **default_config
            )
            
            # Format caption with object information
            formatted_caption = format_caption(
                caption=caption,
                object_name=object_info.get('class_name'),
                confidence=object_info.get('confidence')
            )
            
            result = {
                'success': True,
                'caption': caption,
                'formatted_caption': formatted_caption,
                'clip_score': score,
                'object_info': object_info,
                'generation_config': default_config
            }
            
            self.logger.info(
                f"Generated caption for {object_info.get('class_name', 'object')}: "
                f"{caption} (score: {score:.3f})"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate caption: {e}")
            result = {
                'success': False,
                'error': str(e),
                'caption': '',
                'formatted_caption': '',
                'clip_score': 0.0,
                'object_info': object_info,
                'generation_config': default_config
            }
        
        return result
    
    def caption_multiple_objects(
        self,
        image: Image.Image,
        detection_results: List[Dict[str, Any]],
        generation_config: Optional[Dict[str, Any]] = None,
        num_samples_per_object: int = 1
    ) -> List[Dict[str, Any]]:
        """Generate captions for multiple detected objects.
        
        Args:
            image: PIL Image
            detection_results: List of detection results with alpha masks
            generation_config: Optional generation parameters
            num_samples_per_object: Number of caption samples per object
            
        Returns:
            List of caption results for each object
        """
        results = []
        
        for i, detection in enumerate(detection_results):
            self.logger.info(
                f"Captioning object {i + 1}/{len(detection_results)}: "
                f"{detection.get('class_name', 'unknown')}"
            )
            
            # Get alpha mask
            if 'alpha_tensor' in detection:
                # From prepared alpha masks (already torch tensor)
                alpha_mask = detection['alpha_tensor']
            elif 'alpha_mask' in detection:
                alpha_mask = detection['alpha_mask']
                if not isinstance(alpha_mask, torch.Tensor):
                    alpha_mask = torch.from_numpy(alpha_mask)
            elif 'mask' in detection:
                # Convert binary mask to alpha mask
                mask = detection['mask']
                if not isinstance(mask, torch.Tensor):
                    mask = torch.from_numpy(mask)
                alpha_mask = mask.float()
            else:
                self.logger.error(f"No mask found for object {i}")
                continue
            
            # Ensure alpha mask is on correct device
            alpha_mask = alpha_mask.to(self.device)
            
            if num_samples_per_object == 1:
                # Single caption
                result = self.caption_object(
                    image=image,
                    alpha_mask=alpha_mask,
                    object_info=detection,
                    generation_config=generation_config
                )
                results.append(result)
            else:
                # Multiple caption samples
                samples = []
                for sample_idx in range(num_samples_per_object):
                    self.logger.info(f"  Sample {sample_idx + 1}/{num_samples_per_object}")
                    
                    result = self.caption_object(
                        image=image,
                        alpha_mask=alpha_mask,
                        object_info=detection,
                        generation_config=generation_config
                    )
                    samples.append(result)
                
                # Keep the best sample
                best_sample = max(
                    samples, 
                    key=lambda x: x.get('clip_score', 0.0) if x.get('success', False) else -1
                )
                best_sample['all_samples'] = samples
                results.append(best_sample)
        
        return results
    
    def caption_full_image(
        self,
        image: Image.Image,
        generation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate caption for the full image (no masking).
        
        Args:
            image: PIL Image
            generation_config: Optional generation parameters
            
        Returns:
            Caption result dictionary
        """
        # Create full alpha mask (all ones)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get image size after preprocessing
        preprocessed = self.clip_wrapper.preprocess(image)
        _, h, w = preprocessed.shape
        
        full_mask = torch.ones(1, h, w, device=self.device)
        
        # Create dummy object info for full image
        object_info = {
            'class_name': 'full_image',
            'confidence': 1.0,
            'bbox': (0, 0, image.width, image.height),
            'area': image.width * image.height
        }
        
        return self.caption_object(
            image=image,
            alpha_mask=full_mask,
            object_info=object_info,
            generation_config=generation_config
        )
    
    def process_image_complete(
        self,
        image: Image.Image,
        detection_results: List[Dict[str, Any]],
        generation_config: Optional[Dict[str, Any]] = None,
        include_full_image: bool = True,
        num_samples_per_object: int = 1
    ) -> Dict[str, Any]:
        """Complete image processing: objects + full image captioning.
        
        Args:
            image: PIL Image
            detection_results: List of detection results with masks
            generation_config: Optional generation parameters
            include_full_image: Whether to include full image caption
            num_samples_per_object: Number of caption samples per object
            
        Returns:
            Complete results dictionary
        """
        self.logger.info("Starting complete image captioning")
        
        results = {
            'success': True,
            'image_info': {
                'size': image.size,
                'mode': image.mode
            },
            'object_captions': [],
            'full_image_caption': None,
            'summary': {}
        }
        
        try:
            # Caption detected objects
            if detection_results:
                object_captions = self.caption_multiple_objects(
                    image=image,
                    detection_results=detection_results,
                    generation_config=generation_config,
                    num_samples_per_object=num_samples_per_object
                )
                results['object_captions'] = object_captions
            
            # Caption full image
            if include_full_image:
                full_caption = self.caption_full_image(
                    image=image,
                    generation_config=generation_config
                )
                results['full_image_caption'] = full_caption
            
            # Create summary
            successful_objects = [
                obj for obj in results['object_captions'] 
                if obj.get('success', False)
            ]
            
            results['summary'] = {
                'total_objects_detected': len(detection_results),
                'successful_captions': len(successful_objects),
                'failed_captions': len(detection_results) - len(successful_objects),
                'has_full_image_caption': include_full_image and results['full_image_caption'].get('success', False)
            }
            
            self.logger.info(
                f"Captioning completed: {len(successful_objects)}/{len(detection_results)} "
                f"objects successfully captioned"
            )
            
        except Exception as e:
            self.logger.error(f"Complete image processing failed: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results
    
    def to(self, device: str):
        """Move pipeline to device."""
        self.device = device
        self.clip_wrapper.to(device)
        self.text_generator.to(device)
        return self
