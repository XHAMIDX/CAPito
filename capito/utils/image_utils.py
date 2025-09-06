"""
Image Utilities
==============

Image processing utilities for the CAPito system.
"""

import os
import logging
from typing import Union, Tuple, Optional
from pathlib import Path

import numpy as np
from PIL import Image
import cv2


logger = logging.getLogger(__name__)


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Load image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL Image object
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.debug(f"Loaded image: {image_path} ({image.size})")
        return image
        
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")


def save_image(image: Image.Image, output_path: Union[str, Path]) -> None:
    """
    Save PIL Image to file.
    
    Args:
        image: PIL Image to save
        output_path: Output file path
    """
    output_path = Path(output_path)
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        image.save(output_path, quality=95)
        logger.debug(f"Saved image: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save image to {output_path}: {e}")
        raise


def resize_image(
    image: Image.Image, 
    target_size: Tuple[int, int],
    maintain_aspect: bool = True,
    fill_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """
    Resize image to target size.
    
    Args:
        image: Input PIL Image
        target_size: Target size (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        fill_color: Fill color for padding if maintaining aspect ratio
        
    Returns:
        Resized PIL Image
    """
    if not maintain_aspect:
        return image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Calculate scaling factor
    original_width, original_height = image.size
    target_width, target_height = target_size
    
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    scale = min(scale_w, scale_h)
    
    # Calculate new size
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize image
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new image with target size and fill color
    result = Image.new('RGB', target_size, fill_color)
    
    # Paste resized image in center
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    result.paste(resized, (x_offset, y_offset))
    
    return result


def crop_image(
    image: Image.Image, 
    bbox: Tuple[int, int, int, int]
) -> Image.Image:
    """
    Crop image to bounding box.
    
    Args:
        image: Input PIL Image
        bbox: Bounding box (x1, y1, x2, y2)
        
    Returns:
        Cropped PIL Image
    """
    x1, y1, x2, y2 = bbox
    
    # Ensure coordinates are within image bounds
    width, height = image.size
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bounding box: {bbox}")
    
    return image.crop((x1, y1, x2, y2))


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.
    
    Args:
        array: Numpy array (H, W, C) or (H, W)
        
    Returns:
        PIL Image
    """
    if array.dtype != np.uint8:
        # Normalize to 0-255 range
        if array.max() <= 1.0:
            array = (array * 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)
    
    if len(array.shape) == 2:
        # Grayscale
        return Image.fromarray(array, mode='L')
    elif len(array.shape) == 3:
        # Color
        if array.shape[2] == 3:
            return Image.fromarray(array, mode='RGB')
        elif array.shape[2] == 4:
            return Image.fromarray(array, mode='RGBA')
    
    raise ValueError(f"Unsupported array shape: {array.shape}")


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to numpy array.
    
    Args:
        image: PIL Image
        
    Returns:
        Numpy array
    """
    return np.array(image)


def create_thumbnail(
    image: Image.Image, 
    size: Tuple[int, int] = (256, 256)
) -> Image.Image:
    """
    Create thumbnail of image.
    
    Args:
        image: Input PIL Image
        size: Thumbnail size
        
    Returns:
        Thumbnail PIL Image
    """
    thumbnail = image.copy()
    thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
    return thumbnail


def get_image_info(image: Union[Image.Image, str, Path]) -> dict:
    """
    Get information about an image.
    
    Args:
        image: PIL Image or path to image file
        
    Returns:
        Dictionary with image information
    """
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Get file info
        file_size = image_path.stat().st_size
        
        # Load image for dimensions
        img = load_image(image_path)
        
        return {
            "path": str(image_path),
            "filename": image_path.name,
            "file_size": file_size,
            "file_size_mb": file_size / (1024 * 1024),
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "format": img.format
        }
    
    elif isinstance(image, Image.Image):
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format
        }
    
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def validate_image_format(image_path: Union[str, Path]) -> bool:
    """
    Validate if file is a supported image format.
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if valid image format
    """
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    image_path = Path(image_path)
    extension = image_path.suffix.lower()
    
    if extension not in supported_formats:
        return False
    
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def batch_resize_images(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    target_size: Tuple[int, int],
    maintain_aspect: bool = True
) -> None:
    """
    Batch resize images in a directory.
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for resized images
        target_size: Target size for all images
        maintain_aspect: Whether to maintain aspect ratio
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    for image_path in input_dir.iterdir():
        if image_path.suffix.lower() in image_extensions:
            try:
                # Load and resize image
                image = load_image(image_path)
                resized = resize_image(image, target_size, maintain_aspect)
                
                # Save resized image
                output_path = output_dir / image_path.name
                save_image(resized, output_path)
                
                logger.info(f"Resized: {image_path.name}")
                
            except Exception as e:
                logger.error(f"Failed to resize {image_path.name}: {e}")


def create_image_grid(
    images: list,
    grid_size: Optional[Tuple[int, int]] = None,
    image_size: Tuple[int, int] = (256, 256),
    padding: int = 10,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """
    Create a grid of images.
    
    Args:
        images: List of PIL Images
        grid_size: Grid size (cols, rows). Auto-calculated if None
        image_size: Size to resize each image to
        padding: Padding between images
        background_color: Background color
        
    Returns:
        Grid image
    """
    if not images:
        raise ValueError("No images provided")
    
    num_images = len(images)
    
    if grid_size is None:
        # Calculate square-ish grid
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
        grid_size = (cols, rows)
    
    cols, rows = grid_size
    
    # Calculate grid dimensions
    grid_width = cols * image_size[0] + (cols + 1) * padding
    grid_height = rows * image_size[1] + (rows + 1) * padding
    
    # Create grid image
    grid = Image.new('RGB', (grid_width, grid_height), background_color)
    
    # Place images in grid
    for i, image in enumerate(images[:cols * rows]):
        row = i // cols
        col = i % cols
        
        # Resize image
        resized = resize_image(image, image_size, maintain_aspect=True)
        
        # Calculate position
        x = col * (image_size[0] + padding) + padding
        y = row * (image_size[1] + padding) + padding
        
        # Paste image
        grid.paste(resized, (x, y))
    
    return grid
