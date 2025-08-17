# Model Organization Guide for GET_CAPTION

This document explains the new centralized model organization structure designed to prevent conflicts when deploying to servers.

## Overview

The GET_CAPTION project uses multiple AI models that were previously scattered across different directories. This new structure centralizes all models in a single `models/` directory with clear categorization.

## Directory Structure

```
models/
├── alpha_clip/           # AlphaCLIP vision-language models
│   └── checkpoints/      # AlphaCLIP checkpoint files (.pth)
├── detection/            # Object detection and segmentation models
│   ├── yolo/            # YOLOv8 detection models (.pt)
│   └── sam/             # SAM2 segmentation models (.pt)
├── language/             # Language models (BERT, RoBERTa)
└── legacy/               # Legacy ConZIC models (.pth)
```

## Model Categories

### 1. AlphaCLIP Models
**Location**: `models/alpha_clip/checkpoints/`
**File Format**: `.pth` (PyTorch checkpoints)
**Models**:
- `clip_b16_grit1m_fultune_8xe.pth` - ViT-B/16 base model
- `clip_l14_grit1m_fultune_8xe.pth` - ViT-L/14 large model
- `clip_l14_336_grit1m_fultune_8xe.pth` - ViT-L/14 336px model
- `clip_b16_grit20m_fultune_2xe.pth` - ViT-B/16 20M dataset
- `clip_l14_grit20m_fultune_2xe.pth` - ViT-L/14 20M dataset
- `clip_l14_336_grit20m_fultune_2xe.pth` - ViT-L/14 336px 20M dataset

### 2. Detection Models
**Location**: `models/detection/yolo/`
**File Format**: `.pt` (PyTorch models)
**Models**:
- `yolov8n.pt` - YOLOv8 nano (~6MB)
- `yolov8s.pt` - YOLOv8 small (~22MB)
- `yolov8m.pt` - YOLOv8 medium (~52MB)
- `yolov8l.pt` - YOLOv8 large (~88MB)
- `yolov8x.pt` - YOLOv8 extra large (~136MB)

### 3. Segmentation Models
**Location**: `models/detection/sam/`
**File Format**: `.pt` (PyTorch models)
**Models**:
- `sam2_t.pt` - SAM2 tiny (~39MB)
- `sam2_s.pt` - SAM2 small (~46MB)
- `sam2_b.pt` - SAM2 base (~159MB)
- `sam2_l.pt` - SAM2 large (~224MB)

### 4. Language Models
**Location**: `models/language/`
**File Format**: Downloaded automatically by transformers library
**Models**:
- `bert-base-uncased` - BERT base model
- `bert-large-uncased` - BERT large model
- `roberta-base` - RoBERTa base model
- `roberta-large` - RoBERTa large model

### 5. Legacy Models
**Location**: `models/legacy/`
**File Format**: `.pth` (PyTorch checkpoints)
**Models**: Any existing ConZIC or legacy model files

## Migration Process

### Step 1: Run the Migration Script

```bash
# List current models
python migrate_models.py --list

# Organize existing models into new structure
python migrate_models.py

# Download all available models
python migrate_models.py --download-all

# Clean up old structure after migration
python migrate_models.py --cleanup
```

### Step 2: Verify Migration

Check that models are in the correct locations:

```bash
# Check directory structure
ls -la models/
ls -la models/alpha_clip/checkpoints/
ls -la models/detection/yolo/
ls -la models/detection/sam/
```

### Step 3: Update Configuration

The configuration automatically uses the new paths. No manual changes needed.

## Configuration Integration

The new structure is automatically integrated into the configuration system:

```python
from src.config import Config

config = Config()

# Model paths are automatically managed
print(config.model_paths.alpha_clip_checkpoints)  # models/alpha_clip/checkpoints/
print(config.model_paths.yolo_models)            # models/detection/yolo/
print(config.model_paths.sam_models)             # models/detection/sam/
```

## Benefits

### 1. **Conflict Prevention**
- All models are in one location
- No duplicate model files
- Clear separation of model types

### 2. **Easy Deployment**
- Single `models/` directory to copy to server
- Predictable file structure
- No scattered model files

### 3. **Version Control**
- Models can be excluded from git (add to .gitignore)
- Easy to track which models are available
- Simple backup and restore process

### 4. **Maintenance**
- Centralized model management
- Easy to update specific model types
- Clear inventory of available models

## Server Deployment

### Option 1: Copy Models Directory
```bash
# Copy entire models directory to server
scp -r models/ user@server:/path/to/GET_CAPTION/

# Or use rsync for large transfers
rsync -avz models/ user@server:/path/to/GET_CAPTION/models/
```

### Option 2: Download on Server
```bash
# On server, run migration script to download models
python migrate_models.py --download-all
```

### Option 3: Use Model Manager
```python
from src.utils.model_manager import ModelManager

# Initialize and download models
manager = ModelManager()
manager.download_all_models()
```

## Troubleshooting

### Common Issues

1. **Model Not Found**
   - Check if model exists in correct directory
   - Verify file permissions
   - Run `python migrate_models.py --list` to see inventory

2. **Download Failures**
   - Check internet connection
   - Verify disk space
   - Check file permissions in models directory

3. **Import Errors**
   - Ensure `src/` is in Python path
   - Check that all dependencies are installed
   - Verify model file integrity

### Debug Mode

Enable debug logging to see detailed model loading information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Then run your pipeline
from src.main_pipeline import GetCaptionPipeline
pipeline = GetCaptionPipeline()
```

## File Size Estimates

| Model Type | Size Range | Notes |
|------------|------------|-------|
| YOLOv8 | 6MB - 136MB | Detection models |
| SAM2 | 39MB - 224MB | Segmentation models |
| AlphaCLIP | 300MB - 800MB | Vision-language models |
| Language Models | 100MB - 500MB | Downloaded automatically |

**Total estimated size**: 1-2GB for complete model set

## Best Practices

### 1. **Model Selection**
- Use smaller models for development/testing
- Use larger models for production quality
- Balance speed vs. accuracy based on requirements

### 2. **Storage Management**
- Monitor disk space usage
- Consider using SSD for faster model loading
- Implement model cleanup for unused models

### 3. **Backup Strategy**
- Backup models directory regularly
- Use version control for configuration
- Document model versions and sources

### 4. **Performance Optimization**
- Keep frequently used models on fast storage
- Use model caching when possible
- Consider model quantization for deployment

## Future Enhancements

### Planned Features
- Model versioning and updates
- Automatic model validation
- Model performance benchmarking
- Cloud storage integration
- Model compression and optimization

### Contributing
To add new model types or improve the organization:
1. Update `ModelPathsConfig` in `src/config.py`
2. Add model URLs to `ModelManager` in `src/utils/model_manager.py`
3. Update this documentation
4. Test with existing pipeline

## Support

For issues with model organization:
1. Check the migration log: `model_migration.log`
2. Run `python migrate_models.py --list` to verify structure
3. Check file permissions and disk space
4. Review this documentation for troubleshooting steps

---

**Last Updated**: August 2024
**Version**: 1.0
**Maintainer**: GET_CAPTION Team
