#!/usr/bin/env python3
"""
Model Migration Script for GET_CAPTION Project

This script helps organize existing models into a centralized structure
to avoid conflicts when deploying to servers.

Usage:
    python migrate_models.py [--download-all] [--cleanup] [--list]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.model_manager import setup_model_environment, ModelManager
from config import ModelPathsConfig


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('model_migration.log')
        ]
    )


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description="Migrate models to centralized structure")
    parser.add_argument("--download-all", action="store_true", 
                       help="Download all available models")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up old model structure after migration")
    parser.add_argument("--list", action="store_true",
                       help="List all available models")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download of existing models")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model migration for GET_CAPTION project")
    
    try:
        # Initialize model manager
        model_manager = setup_model_environment()
        
        # List current models
        if args.list:
            logger.info("Current model inventory:")
            available_models = model_manager.list_available_models()
            for category, models in available_models.items():
                if models:
                    logger.info(f"  {category}: {', '.join(models)}")
                else:
                    logger.info(f"  {category}: None")
        
        # Download all models if requested
        if args.download_all:
            logger.info("Downloading all available models...")
            downloaded = model_manager.download_all_models(force=args.force)
            logger.info(f"Downloaded {len(downloaded)} models")
            
            # List downloaded models
            for model_name, path in downloaded.items():
                logger.info(f"  {model_name}: {path}")
        
        # Cleanup old structure if requested
        if args.cleanup:
            logger.info("Cleaning up old model structure...")
            cleaned_paths = model_manager.cleanup_old_structure()
            if cleaned_paths:
                logger.info(f"Cleaned up {len(cleaned_paths)} paths:")
                for path in cleaned_paths:
                    logger.info(f"  {path}")
            else:
                logger.info("No cleanup needed")
        
        # Show final structure
        logger.info("\nFinal model structure:")
        logger.info(f"  Models root: {model_manager.model_paths.models_root}")
        logger.info(f"  AlphaCLIP: {model_manager.model_paths.alpha_clip_checkpoints}")
        logger.info(f"  YOLO models: {model_manager.model_paths.yolo_models}")
        logger.info(f"  SAM2 models: {model_manager.model_paths.sam_models}")
        logger.info(f"  Language models: {model_manager.model_paths.language_models_root}")
        logger.info(f"  Legacy models: {model_manager.model_paths.legacy_root}")
        
        logger.info("\nMigration completed successfully!")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
