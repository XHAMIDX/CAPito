"""Clean AlphaCLIP wrapper for masked image captioning."""

import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Tuple, Optional
import logging
import time
import urllib.request
from urllib.error import URLError, HTTPError

# Add AlphaCLIP to path
alpha_clip_path = os.path.join(os.path.dirname(__file__), '..', '..', 'AlphaCLIP')
if alpha_clip_path not in sys.path:
    sys.path.append(alpha_clip_path)


class AlphaCLIPWrapper:
    """Clean wrapper for AlphaCLIP with mask support."""
    
    def __init__(
        self,
        model_name: str = "ViT-B/16",
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
            checkpoint_path: Optional[str] = None
            try:
                # Add project root to path for absolute import
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                from src.config import ModelPathsConfig
                model_paths = ModelPathsConfig()
                checkpoint_path = model_paths.get_alpha_clip_path(self.model_name)
                if not os.path.exists(checkpoint_path):
                    # Attempt auto-download for known models
                    self._ensure_checkpoint_available(self.model_name, checkpoint_path)
                    if not os.path.exists(checkpoint_path):
                        checkpoint_path = None
            except ImportError as e:
                self.logger.error(f"Failed to import ModelPathsConfig: {e}")
                pass
            
            # Load model
            # AlphaCLIP's load() expects string "None" when no ckpt provided
            alpha_ckpt_arg = checkpoint_path if checkpoint_path is not None else "None"
            self.model, self.preprocess = load(
                self.model_name, 
                alpha_vision_ckpt_pth=alpha_ckpt_arg,
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

    def _get_checkpoint_url(self, model_name: str) -> Optional[str]:
        """Return a download URL for a known AlphaCLIP checkpoint, if available."""
        # Source: AlphaCLIP model-zoo
        url_mapping = {
            # GRIT-1M finetuned weights
            "ViT-B/16": "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_b16_grit1m_fultune_8xe.pth",
            "ViT-L/14": "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_grit1m_fultune_8xe.pth",
            "ViT-L/14@336px": "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_336_grit1m_fultune_8xe.pth",
            # Some variants below may not be available; return None if unknown
            # "ViT-B/32": "",  # Not provided in official model-zoo
            # "RN50": "",
        }
        return url_mapping.get(model_name)

    def _ensure_checkpoint_available(self, model_name: str, destination_path: str) -> None:
        """Download the AlphaCLIP checkpoint to destination_path if the file does not exist."""
        try:
            if os.path.exists(destination_path):
                return
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            url = self._get_checkpoint_url(model_name)
            if not url:
                self.logger.warning(
                    f"No known download URL for AlphaCLIP checkpoint of {model_name}. "
                    f"Proceeding without a visual checkpoint."
                )
                return
            self.logger.info(f"Downloading AlphaCLIP checkpoint for {model_name}...")
            self.logger.info(f"From: {url}")
            self.logger.info(f"To:   {destination_path}")
            urllib.request.urlretrieve(url, destination_path)
            # Basic sanity check on file size (> 1 MB)
            file_size = os.path.getsize(destination_path) if os.path.exists(destination_path) else 0
            if file_size < 1 * 1024 * 1024:
                try:
                    os.remove(destination_path)
                except OSError:
                    pass
                raise RuntimeError("Downloaded checkpoint is too small; download may have failed.")
            self.logger.info("AlphaCLIP checkpoint downloaded successfully.")
        except (URLError, HTTPError, RuntimeError, Exception) as download_err:
            self.logger.warning(
                f"Could not download AlphaCLIP checkpoint automatically: {download_err}. "
                f"Continuing without a visual checkpoint."
            )
    
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
        try:
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            self.logger.info(f"Preprocessing image...")
            # Preprocess image
            image_tensor = self.preprocess(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            self.logger.info(f"Image tensor shape: {image_tensor.shape}")
            
            # Ensure alpha mask has correct dimensions
            if alpha_mask.dim() == 2:
                alpha_mask = alpha_mask.unsqueeze(0)  # Add batch dimension
            
            self.logger.info(f"Alpha mask shape after dimension fix: {alpha_mask.shape}")
            
            # Get the target size from the preprocessed image tensor
            _, _, target_h, target_w = image_tensor.shape
            
            # Resize alpha mask to match image tensor spatial dimensions
            current_h, current_w = alpha_mask.shape[-2:]
            
            self.logger.info(f"Target size: ({target_h}, {target_w}), Current size: ({current_h}, {current_w})")
            
            if alpha_mask.shape[-2:] != (target_h, target_w):
                self.logger.info("Resizing alpha mask to match image tensor")
                alpha_mask = torch.nn.functional.interpolate(
                    alpha_mask.unsqueeze(0),  # Add channel dimension
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)  # Remove channel dimension
            else:
                self.logger.info("Mask already matches target size")
            
            # Ensure alpha mask is on correct device
            alpha_mask = alpha_mask.to(self.device)
            self.logger.info(f"Final alpha mask shape: {alpha_mask.shape}, device: {alpha_mask.device}")
            
            # Encode image with alpha mask
            self.logger.info("Running AlphaCLIP model inference...")
            with torch.no_grad():
                image_embeds = self.model.encode_image(image_tensor, alpha_mask)
            
            self.logger.info(f"Image encoding successful, shape: {image_embeds.shape}")
            return image_embeds
            
        except Exception as e:
            self.logger.error(f"Error in encode_image_with_mask: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
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
        text_candidates: List[str],
        image_embeds: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score text candidates against masked image.
        
        Args:
            image: PIL Image
            alpha_mask: Alpha mask tensor
            text_candidates: List of text candidates to score
            image_embeds: Optional precomputed image embeddings for efficiency
            
        Returns:
            Tuple of (normalized_scores, raw_scores)
        """
        start_time = time.time()
        self.logger.info(f"Scoring {len(text_candidates)} text candidates with AlphaCLIP")
        
        try:
            # Check if model is loaded
            if self.model is None or self.preprocess is None:
                raise ValueError("AlphaCLIP model not properly loaded")
            
            self.logger.info(f"Model loaded: {self.model_name}, device: {self.device}")
            self.logger.info(f"Image size: {image.size}, mode: {image.mode}")
            self.logger.info(f"Alpha mask shape: {alpha_mask.shape}, device: {alpha_mask.device}")
            
            # Encode image with mask if not provided
            if image_embeds is None:
                self.logger.info("Encoding image with mask...")
                image_embeds = self.encode_image_with_mask(image, alpha_mask)
                self.logger.info(f"Image encoding shape: {image_embeds.shape}")
            
            # Encode text candidates
            self.logger.info("Encoding text candidates...")
            text_embeds = self.encode_text(text_candidates)
            self.logger.info(f"Text encoding shape: {text_embeds.shape}")
            
            # Compute similarity
            self.logger.info("Computing similarity...")
            result = self.compute_similarity(image_embeds, text_embeds)
            self.logger.info(f"Similarity result shapes: {[r.shape for r in result]}")
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"AlphaCLIP scoring completed in {elapsed_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during AlphaCLIP scoring: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Return zero scores as fallback
            num_candidates = len(text_candidates)
            zero_scores = torch.zeros(num_candidates, device=self.device)
            return zero_scores, zero_scores.clone()
    
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
