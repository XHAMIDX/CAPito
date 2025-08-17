"""Image processing utilities."""

import os
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional
import torch


def load_image(image_path: Union[str, Image.Image]) -> Image.Image:
    """Load image from path or return if already PIL Image."""
    if isinstance(image_path, str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path)
    else:
        image = image_path
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image


def save_image(image: Image.Image, save_path: str) -> None:
    """Save PIL Image to file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)


def resize_image(
    image: Image.Image, 
    size: Tuple[int, int], 
    maintain_aspect: bool = True
) -> Image.Image:
    """Resize image while optionally maintaining aspect ratio."""
    if maintain_aspect:
        image.thumbnail(size, Image.Resampling.LANCZOS)
        return image
    else:
        return image.resize(size, Image.Resampling.LANCZOS)


def mask_to_alpha(mask: np.ndarray, blur_radius: int = 2) -> np.ndarray:
    """Convert binary mask to alpha channel for AlphaCLIP.
    
    Args:
        mask: Binary mask array (H, W) with values 0 or 1
        blur_radius: Radius for edge smoothing
        
    Returns:
        Alpha mask array (H, W) with values 0.0 to 1.0
    """
    # Ensure mask is float
    alpha = mask.astype(np.float32)
    
    # Optional: Apply Gaussian blur for smoother edges
    if blur_radius > 0:
        from scipy.ndimage import gaussian_filter
        alpha = gaussian_filter(alpha, sigma=blur_radius)
    
    # Ensure values are in [0, 1] range
    alpha = np.clip(alpha, 0.0, 1.0)
    
    return alpha


def create_bbox_mask(
    image_shape: Tuple[int, int], 
    bbox: Tuple[int, int, int, int]
) -> np.ndarray:
    """Create a binary mask from bounding box coordinates.
    
    Args:
        image_shape: (height, width) of the image
        bbox: (x1, y1, x2, y2) bounding box coordinates
        
    Returns:
        Binary mask array (H, W)
    """
    mask = np.zeros(image_shape, dtype=np.float32)
    x1, y1, x2, y2 = bbox
    
    # Ensure coordinates are within image bounds
    x1 = max(0, min(x1, image_shape[1]))
    y1 = max(0, min(y1, image_shape[0]))
    x2 = max(0, min(x2, image_shape[1]))
    y2 = max(0, min(y2, image_shape[0]))
    
    mask[y1:y2, x1:x2] = 1.0
    return mask


def numpy_to_torch_alpha(
    alpha_mask: np.ndarray, 
    target_size: Tuple[int, int],
    device: str = 'cpu'
) -> torch.Tensor:
    """Convert numpy alpha mask to torch tensor for AlphaCLIP.
    
    Args:
        alpha_mask: Alpha mask array (H, W)
        target_size: Target size (height, width) for AlphaCLIP
        device: Target device
        
    Returns:
        Alpha tensor (1, H, W) ready for AlphaCLIP
    """
    # Resize if needed
    if alpha_mask.shape != target_size:
        from PIL import Image
        alpha_pil = Image.fromarray((alpha_mask * 255).astype(np.uint8), mode='L')
        alpha_pil = alpha_pil.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
        alpha_mask = np.array(alpha_pil).astype(np.float32) / 255.0
    
    # Convert to torch tensor
    alpha_tensor = torch.from_numpy(alpha_mask).unsqueeze(0)  # Add batch dimension
    alpha_tensor = alpha_tensor.to(device)
    
    return alpha_tensor
