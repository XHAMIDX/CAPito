"""
AlphaCLIP Vision-Language Model Wrapper
======================================

Wrapper for AlphaCLIP model providing clean interface for the CAPito pipeline.
Handles model loading, inference, and feature extraction.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Add AlphaCLIP to path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
alpha_clip_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "AlphaCLIP")
if alpha_clip_path not in sys.path:
    sys.path.append(alpha_clip_path)

try:
    from alpha_clip import alpha_clip
except ImportError as e:
    logging.error(f"Failed to import AlphaCLIP: {e}")
    logging.error("Make sure AlphaCLIP is properly installed")
    raise


class AlphaCLIPWrapper:
    """
    Wrapper for AlphaCLIP model with clean interface.
    
    Provides methods for:
    - Image encoding
    - Text encoding  
    - Image-text similarity computation
    - Feature extraction
    """
    
    def __init__(
        self, 
        model_name: str = "ViT-B/16",
        model_path: Optional[str] = None,
        device: str = "cuda",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize AlphaCLIP wrapper.
        
        Args:
            model_name: Name of the AlphaCLIP model variant
            model_path: Path to model checkpoint file  
            device: Device to load model on
            cache_dir: Cache directory for downloads
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = torch.device(device)
        self.cache_dir = cache_dir
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing AlphaCLIP: {model_name}")
        
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the AlphaCLIP model."""
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Load from local checkpoint
                self.logger.info(f"Loading AlphaCLIP from: {self.model_path}")
                self.model, self.preprocess = alpha_clip.load(
                    self.model_name,
                    alpha_vision_ckpt_pth=self.model_path,
                    device=self.device
                )
            else:
                # Load from default/download
                self.logger.info(f"Loading AlphaCLIP: {self.model_name}")
                self.model, self.preprocess = alpha_clip.load(
                    self.model_name,
                    device=self.device
                )
            
            self.model.eval()
            self.logger.info("AlphaCLIP loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load AlphaCLIP: {e}")
            raise
    
    def encode_image(
        self, 
        image: Union[Image.Image, torch.Tensor, np.ndarray],
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode image to feature vector with optional alpha mask.
        
        Args:
            image: Input image (PIL Image, tensor, or numpy array)
            mask: Alpha mask for the image (required for AlphaCLIP)
            normalize: Whether to normalize the features
            
        Returns:
            Image feature tensor
        """
        with torch.no_grad():
            # Preprocess image
            if isinstance(image, Image.Image):
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            elif isinstance(image, np.ndarray):
                image_pil = Image.fromarray(image)
                image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)
            elif isinstance(image, torch.Tensor):
                if image.dim() == 3:
                    image_tensor = image.unsqueeze(0).to(self.device)
                else:
                    image_tensor = image.to(self.device)
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Preprocess mask
            if mask is not None:
                if isinstance(mask, np.ndarray):
                    # Convert numpy mask to tensor and resize to match image
                    mask_tensor = torch.from_numpy(mask).float()
                    if mask_tensor.dim() == 2:
                        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                    elif mask_tensor.dim() == 3:
                        mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dim
                elif isinstance(mask, torch.Tensor):
                    mask_tensor = mask.float()
                    if mask_tensor.dim() == 2:
                        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
                    elif mask_tensor.dim() == 3:
                        mask_tensor = mask_tensor.unsqueeze(0)
                else:
                    raise ValueError(f"Unsupported mask type: {type(mask)}")
                
                # Resize mask to match image tensor size
                if mask_tensor.shape[-2:] != image_tensor.shape[-2:]:
                    mask_tensor = torch.nn.functional.interpolate(
                        mask_tensor,
                        size=image_tensor.shape[-2:],
                        mode='nearest'
                    )
                mask_tensor = mask_tensor.to(self.device)
            else:
                # Create a full mask if none provided (fallback)
                mask_tensor = torch.ones_like(image_tensor[:, :1, :, :])
            
            # Extract features with AlphaCLIP
            image_features = self.model.encode_image(image_tensor, mask_tensor)
            
            if normalize:
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features
    
    def encode_text(
        self, 
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode text to feature vector.
        
        Args:
            texts: Input text(s)
            normalize: Whether to normalize the features
            
        Returns:
            Text feature tensor
        """
        with torch.no_grad():
            # Handle single text input
            if isinstance(texts, str):
                texts = [texts]
            
            # Tokenize text
            text_tokens = alpha_clip.tokenize(texts).to(self.device)
            
            # Extract features
            text_features = self.model.encode_text(text_tokens)
            
            if normalize:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features
    
    def compute_similarity(
        self, 
        image: Union[Image.Image, torch.Tensor],
        texts: Union[str, List[str]],
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute image-text similarity scores.
        
        Args:
            image: Input image
            texts: Text candidates
            mask: Alpha mask for the image (for AlphaCLIP)
            temperature: Temperature scaling for logits
            
        Returns:
            Similarity scores tensor
        """
        # Encode image and text
        image_features = self.encode_image(image, mask=mask, normalize=True)
        text_features = self.encode_text(texts, normalize=True)
        
        # Compute similarity
        logits_per_image = (image_features @ text_features.T) / temperature
        
        return logits_per_image
    
    def rank_texts(
        self, 
        image: Union[Image.Image, torch.Tensor],
        texts: List[str],
        return_scores: bool = False
    ) -> Union[List[str], Tuple[List[str], List[float]]]:
        """
        Rank texts by similarity to image.
        
        Args:
            image: Input image
            texts: List of text candidates
            return_scores: Whether to return similarity scores
            
        Returns:
            Ranked texts (and optionally scores)
        """
        if not texts:
            return ([], []) if return_scores else []
        
        # Compute similarities
        similarities = self.compute_similarity(image, texts)
        similarities = similarities.squeeze(0)  # Remove batch dimension
        
        # Sort by similarity (descending)
        sorted_indices = torch.argsort(similarities, descending=True)
        
        ranked_texts = [texts[i] for i in sorted_indices]
        
        if return_scores:
            sorted_scores = [similarities[i].item() for i in sorted_indices]
            return ranked_texts, sorted_scores
        else:
            return ranked_texts
    
    def get_best_text(
        self, 
        image: Union[Image.Image, torch.Tensor],
        texts: List[str]
    ) -> Tuple[str, float]:
        """
        Get the best matching text for an image.
        
        Args:
            image: Input image
            texts: List of text candidates
            
        Returns:
            Best text and its similarity score
        """
        if not texts:
            return "", 0.0
        
        ranked_texts, scores = self.rank_texts(image, texts, return_scores=True)
        return ranked_texts[0], scores[0]
    
    def extract_region_features(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int]
    ) -> torch.Tensor:
        """
        Extract features for a specific region of the image.
        
        Args:
            image: Full image
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Region feature tensor
        """
        # Crop region
        x1, y1, x2, y2 = bbox
        region = image.crop((x1, y1, x2, y2))
        
        # Encode cropped region
        return self.encode_image(region)
    
    def batch_encode_images(
        self, 
        images: List[Union[Image.Image, torch.Tensor]],
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Encode multiple images in batches.
        
        Args:
            images: List of images
            batch_size: Batch size for processing
            
        Returns:
            Concatenated image features
        """
        all_features = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                if isinstance(img, Image.Image):
                    tensor = self.preprocess(img)
                elif isinstance(img, torch.Tensor):
                    tensor = img
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")
                batch_tensors.append(tensor)
            
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                all_features.append(features)
        
        return torch.cat(all_features, dim=0)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": str(self.device),
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
    
    def set_eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self.model.eval()
    
    def set_train_mode(self) -> None:
        """Set model to training mode."""
        self.model.train()
    
    def to_device(self, device: Union[str, torch.device]) -> None:
        """Move model to specified device."""
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
    
    def cleanup(self) -> None:
        """Clean up model resources."""
        if hasattr(self.model, 'cleanup'):
            self.model.cleanup()
        
        # Clear CUDA cache if on GPU
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def __call__(
        self, 
        image: Union[Image.Image, torch.Tensor],
        text: Union[str, List[str]]
    ) -> torch.Tensor:
        """Make the wrapper callable for similarity computation."""
        return self.compute_similarity(image, text)
