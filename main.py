#!/usr/bin/env python3
"""
CAPito Command Line Interface
============================

Main CLI for the CAPito content-aware image captioning system.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional

from capito import (
    CAPito, 
    get_default_config, 
    get_vlm_config, 
    get_fast_config, 
    get_quality_config
)
from capito.utils import setup_logging, ModelManager


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CAPito: Content-Aware Photo Image Text Optimizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output
    parser.add_argument(
        "input", 
        help="Input image path or directory"
    )
    parser.add_argument(
        "-o", "--output",
        default="outputs",
        help="Output directory for results"
    )
    
    # Configuration presets
    parser.add_argument(
        "--config",
        choices=["default", "vlm", "fast", "quality"],
        default="default",
        help="Configuration preset"
    )
    
    # Model selection
    parser.add_argument(
        "--vlm-model",
        choices=["ViT-B/16", "ViT-B/32", "ViT-L/14", "ViT-L/14@336px", "RN50"],
        help="VLM model to use"
    )
    parser.add_argument(
        "--detection-model", 
        choices=["yolo8n", "yolo8s", "yolo8m", "yolo8l"],
        help="Detection model to use"
    )
    parser.add_argument(
        "--segmentation-model",
        choices=["sam2_tiny", "sam2_small", "sam2_base", "sam2_large"],
        help="Segmentation model to use"
    )
    
    # Caption generation
    parser.add_argument(
        "--max-length",
        type=int,
        default=6,
        help="Maximum caption length"
    )
    parser.add_argument(
        "--num-captions",
        type=int,
        default=3,
        help="Number of captions to generate"
    )
    parser.add_argument(
        "--prompt",
        help="Custom prompt template"
    )
    
    # Control options
    parser.add_argument(
        "--enable-control",
        action="store_true",
        help="Enable controllable generation"
    )
    parser.add_argument(
        "--control-type",
        choices=["sentiment", "pos"],
        default="sentiment",
        help="Type of controllable generation"
    )
    parser.add_argument(
        "--sentiment",
        choices=["positive", "negative"],
        default="positive",
        help="Sentiment for controllable generation"
    )
    
    # Output options
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Don't save visualization images"
    )
    parser.add_argument(
        "--no-results",
        action="store_true", 
        help="Don't save JSON results"
    )
    
    # System options
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Device to use for inference"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    # Model management
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download required models before processing"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--migrate-models",
        action="store_true",
        help="Migrate models from legacy locations"
    )
    
    return parser.parse_args()


def get_config_from_args(args: argparse.Namespace):
    """Create configuration from command line arguments."""
    # Start with preset configuration
    if args.config == "vlm":
        config = get_vlm_config()
    elif args.config == "fast":
        config = get_fast_config()
    elif args.config == "quality":
        config = get_quality_config()
    else:
        config = get_default_config()
    
    # Override with command line arguments
    if args.vlm_model:
        config.vlm.model_name = args.vlm_model
    if args.detection_model:
        config.detection.model_name = args.detection_model
    if args.segmentation_model:
        config.segmentation.model_name = args.segmentation_model
    
    if args.device:
        config.system.device = args.device
        config.vlm.device = args.device
        config.detection.device = args.device
        config.segmentation.device = args.device
    
    config.system.batch_size = args.batch_size
    config.system.log_level = args.log_level
    config.system.output_dir = args.output
    config.system.save_visualizations = not args.no_visualizations
    config.system.save_results = not args.no_results
    
    config.captioning.max_length = args.max_length
    config.captioning.enable_control = args.enable_control
    config.captioning.control_type = args.control_type
    config.captioning.sentiment_type = args.sentiment
    
    if args.prompt:
        config.captioning.prompt_template = args.prompt
    
    return config


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Handle model management commands
    model_manager = ModelManager()
    
    if args.list_models:
        print("Available Models:")
        print("=" * 50)
        models_status = model_manager.list_available_models()
        
        for model_name, status in models_status.items():
            available = "✓" if status["available"] else "✗"
            size = f"{status['size_mb']:.1f}MB" if status["size_mb"] else "Unknown"
            print(f"{available} {model_name} ({status['type']}) - {size}")
        
        return 0
    
    if args.migrate_models:
        print("Migrating models from legacy locations...")
        from capito.utils import migrate_legacy_models
        migrate_legacy_models()
        print("Migration completed.")
        return 0
    
    # Get configuration
    config = get_config_from_args(args)
    
    # Download models if requested
    if args.download_models:
        print("Downloading required models...")
        try:
            model_manager.ensure_model(config.get_model_path("vlm", config.vlm.model_name))
            model_manager.ensure_model(config.get_model_path("detection", config.detection.model_name))
            model_manager.ensure_model(config.get_model_path("segmentation", config.segmentation.model_name))
            print("All models downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download models: {e}")
            return 1
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1
    
    # Initialize CAPito
    try:
        logger.info("Initializing CAPito pipeline...")
        capito = CAPito(config)
        logger.info("CAPito initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize CAPito: {e}")
        return 1
    
    # Process input
    try:
        if input_path.is_file():
            # Single image
            print(f"Processing image: {input_path}")
            results = capito.process_image(
                image_path=input_path,
                output_dir=args.output
            )
            
            # Display results
            print("\nGenerated Captions:")
            for i, caption in enumerate(results["captions"], 1):
                print(f"{i}. {caption}")
            
            print(f"\nProcessing completed in {results['processing_time']:.2f} seconds")
            print(f"Results saved to: {args.output}")
            
        elif input_path.is_dir():
            # Directory of images
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
            image_paths = [
                p for p in input_path.iterdir() 
                if p.suffix.lower() in image_extensions
            ]
            
            if not image_paths:
                logger.error(f"No images found in directory: {input_path}")
                return 1
            
            print(f"Processing {len(image_paths)} images from: {input_path}")
            results = capito.process_batch(
                image_paths=image_paths,
                output_dir=args.output
            )
            
            # Display summary
            successful = len([r for r in results if "error" not in r])
            failed = len([r for r in results if "error" in r])
            total_time = sum(r.get("processing_time", 0) for r in results)
            
            print(f"\nBatch processing completed:")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")
            print(f"  Total time: {total_time:.2f} seconds")
            print(f"  Average time: {total_time/len(results):.2f} seconds per image")
            print(f"  Results saved to: {args.output}")
        
        else:
            logger.error(f"Invalid input path: {input_path}")
            return 1
    
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1
    
    finally:
        # Cleanup
        capito.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
