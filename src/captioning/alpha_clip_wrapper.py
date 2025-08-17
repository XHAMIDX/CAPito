"""Clean AlphaCLIP wrapper for masked image captioning."""

import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Tuple, Optional
import logging

# Add AlphaCLIP to path
alpha_clip_path = os.path.join(os.path.dirname(__file__), '..', '..', 'AlphaCLIP')
if alpha_clip_path not in sys.path:
    sys.path.append(alpha_clip_path)


class AlphaCLIPWrapper:
    """Clean wrapper for AlphaCLIP with mask support."""
    
    def __init__(
        self,
        model_name: str = "ViT-L/14",
        device: str = "cpu"
    ):
        """Initialize AlphaCLIP wrapper.
        
        Args:
            model_name: Model name (ViT-B/32, ViT-B/16, ViT-L/14, RN50)
            device: Device to run inference on
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.preprocess = None
        self.logger = logging.getLogger(__name__)
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load AlphaCLIP model."""
        try:
            from alpha_clip.alpha_clip import load
            
            # Validate model name
            valid_models = ["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50"]
            if self.model_name not in valid_models:
                self.logger.warning(
                    f"Model {self.model_name} not in {valid_models}. "
                    f"Using ViT-B/32 as fallback."
                )
                self.model_name = "ViT-B/32"
            
            # Try to get checkpoint path from config if available
            checkpoint_path = None
            try:
                from ..config import ModelPathsConfig
                model_paths = ModelPathsConfig()
                checkpoint_path = model_paths.get_alpha_clip_path(self.model_name)
                if not os.path.exists(checkpoint_path):
                    checkpoint_path = None
            except ImportError:
                pass
            
            # Load model
            self.model, self.preprocess = load(
                self.model_name, 
                alpha_vision_ckpt_pth=checkpoint_path,
                device=self.device
            )
            self.model.eval()
            
            self.logger.info(f"Loaded AlphaCLIP model: {self.model_name} on {self.device}")
            if checkpoint_path:
                self.logger.info(f"Using checkpoint: {checkpoint_path}")
            
        except ImportError as e:
            self.logger.error(f"Failed to import AlphaCLIP: {e}")
            raise ImportError(
                "AlphaCLIP not found. Please ensure it's properly installed "
                "and the AlphaCLIP directory is in the project root."
            )
        except Exception as e:
            self.logger.error(f"Failed to load AlphaCLIP model: {e}")
            raise
    
    def encode_image_with_mask(
        self,
        image: Image.Image,
        alpha_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode image with alpha mask using AlphaCLIP.
        
        Args:
            image: PIL Image
            alpha_mask: Alpha mask tensor (1, H, W) or (H, W)
            
        Returns:
            Image embedding tensor
        """
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image
        image_tensor = self.preprocess(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Ensure alpha mask has correct dimensions
        if alpha_mask.dim() == 2:
            alpha_mask = alpha_mask.unsqueeze(0)  # Add batch dimension
        
        # Resize alpha mask to match image tensor spatial dimensions
        _, _, h, w = image_tensor.shape
        if alpha_mask.shape[-2:] != (h, w):
            alpha_mask = torch.nn.functional.interpolate(
                alpha_mask.unsqueeze(0),  # Add channel dimension
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # Remove channel dimension
        
        # Ensure alpha mask is on correct device
        alpha_mask = alpha_mask.to(self.device)
        
        # Encode image with alpha mask
        with torch.no_grad():
            image_embeds = self.model.encode_image(image_tensor, alpha_mask)
        
        return image_embeds
    
    def encode_text(self, text_list: List[str]) -> torch.Tensor:
        """Encode text using AlphaCLIP.
        
        Args:
            text_list: List of text strings
            
        Returns:
            Text embedding tensor
        """
        from alpha_clip.alpha_clip import tokenize
        
        # Tokenize text
        text_tokens = tokenize(text_list)
        text_tokens = text_tokens.to(self.device)
        
        # Encode text
        with torch.no_grad():
            text_embeds = self.model.encode_text(text_tokens)
        
        return text_embeds
    
    def compute_similarity(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute image-text similarity.
        
        Args:
            image_embeds: Image embedding tensor
            text_embeds: Text embedding tensor
            normalize: Whether to normalize scores to [0, 1]
            
        Returns:
            Tuple of (normalized_scores, raw_scores)
        """
        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity
        if image_embeds.dim() == 2 and text_embeds.dim() == 2:
            # Simple case: batch x features
            similarity = torch.matmul(image_embeds, text_embeds.t())
        else:
            # More complex case: handle different shapes
            image_embeds = image_embeds.unsqueeze(-1)
            similarity = torch.matmul(text_embeds, image_embeds).squeeze(-1)
        
        raw_scores = similarity.clone()
        
        if normalize:
            # Normalize to [0, 1] range for fusion with other scores
            min_val = similarity.min()
            max_val = similarity.max()
            if max_val > min_val:
                normalized_scores = (similarity - min_val) / (max_val - min_val)
            else:
                normalized_scores = torch.zeros_like(similarity)
        else:
            normalized_scores = similarity
        
        return normalized_scores, raw_scores
    
    def score_text_candidates(
        self,
        image: Image.Image,
        alpha_mask: torch.Tensor,
        text_candidates: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score text candidates against masked image.
        
        Args:
            image: PIL Image
            alpha_mask: Alpha mask tensor
            text_candidates: List of text candidates to score
            
        Returns:
            Tuple of (normalized_scores, raw_scores)
        """
        # Encode image with mask
        image_embeds = self.encode_image_with_mask(image, alpha_mask)
        
        # Encode text candidates
        text_embeds = self.encode_text(text_candidates)
        
        # Compute similarity
        return self.compute_similarity(image_embeds, text_embeds)
    
    def get_image_features(
        self,
        image: Image.Image,
        alpha_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get image features for caching/reuse.
        
        Args:
            image: PIL Image
            alpha_mask: Optional alpha mask. If None, uses full image.
            
        Returns:
            Image feature tensor
        """
        if alpha_mask is None:
            # Create full mask (all ones)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_tensor = self.preprocess(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            _, _, h, w = image_tensor.shape
            alpha_mask = torch.ones(1, h, w, device=self.device)
        
        return self.encode_image_with_mask(image, alpha_mask)
    
    def to(self, device: str):
        """Move model to device."""
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self
