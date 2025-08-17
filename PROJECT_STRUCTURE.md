# GET_CAPTION Project Structure

## Overview
This document describes the clean, reorganized structure of the GET_CAPTION project.

## Directory Structure

```
GET_CAPTION/
├── README.md                    # Main project documentation
├── requirements.txt             # Python dependencies
├── setup.py                    # Package installation script
├── .gitignore                  # Git ignore rules
├── PROJECT_STRUCTURE.md        # This file
│
├── src/                        # Main source code
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── main_pipeline.py        # Main pipeline entry point
│   │
│   ├── utils/                  # Utility modules
│   │   ├── __init__.py
│   │   ├── logger.py           # Logging utilities
│   │   ├── image_utils.py      # Image processing utilities
│   │   └── text_utils.py       # Text processing utilities
│   │
│   ├── detection/              # Object detection & segmentation
│   │   ├── __init__.py
│   │   ├── detector.py         # YOLOv8 object detection
│   │   ├── segmentation.py     # SAM2 mask generation
│   │   └── pipeline.py         # Combined detection pipeline
│   │
│   └── captioning/             # Caption generation
│       ├── __init__.py
│       ├── alpha_clip_wrapper.py    # Clean AlphaCLIP interface
│       ├── text_generator.py       # ConZIC-style text generation
│       └── caption_pipeline.py     # Caption generation pipeline
│
├── AlphaCLIP/                  # AlphaCLIP submodule (unchanged)
│   ├── alpha_clip/
│   ├── checkpoints/
│   └── ...
│
├── examples/                   # Example scripts and sample data
│   └── simple_example.py       # Simple usage example
│
├── legacy/                     # Legacy/backup directory
│   └── ...                     # Old ConZIC files (if moved)
│
└── logger_legacy/              # Old log files
    └── ...
```

## Key Components

### 1. Main Pipeline (`src/main_pipeline.py`)
- **GetCaptionPipeline**: Main orchestrator class
- Handles single image and batch processing
- Manages configuration and result saving
- Command-line interface

### 2. Configuration (`src/config.py`)
- **Config**: Main configuration class with nested configs
- **ModelConfig**: Model selection and device settings
- **GenerationConfig**: Text generation parameters
- **ProcessingConfig**: Image processing and I/O settings

### 3. Detection Pipeline (`src/detection/`)
- **ObjectDetector**: YOLOv8-based object detection
- **MaskGenerator**: SAM2/bbox mask generation with fallbacks
- **DetectionPipeline**: Combined detection and segmentation

### 4. Captioning Pipeline (`src/captioning/`)
- **AlphaCLIPWrapper**: Clean AlphaCLIP interface with mask support
- **TextGenerator**: ConZIC-style masked language modeling
- **CaptionPipeline**: End-to-end caption generation

### 5. Utilities (`src/utils/`)
- **logger.py**: Colored logging with file and console output
- **image_utils.py**: Image loading, saving, mask conversion
- **text_utils.py**: Text cleaning, formatting, reporting

## Data Flow

```
Input Image
    ↓
ObjectDetector (YOLOv8)
    ↓
MaskGenerator (SAM2/bbox)
    ↓
AlphaCLIPWrapper (masked encoding)
    ↓
TextGenerator (ConZIC approach)
    ↓
Results & Visualizations
```

## Usage Patterns

### Command Line
```bash
python -m src.main_pipeline image.jpg -o results/
python -m src.main_pipeline images/ --batch -o results/
```

### Python API
```python
from src.main_pipeline import GetCaptionPipeline
from src.config import Config

pipeline = GetCaptionPipeline()
results = pipeline.process_single_image("image.jpg")
```

### Custom Configuration
```python
config = Config()
config.model.alpha_clip_model = "ViT-L/14"
config.generation.sentence_len = 10
pipeline = GetCaptionPipeline(config)
```

## Key Improvements Over Original

1. **Clean Architecture**: Modular design with clear separation of concerns
2. **Comprehensive Configuration**: Centralized, type-safe configuration
3. **Robust Error Handling**: Graceful fallbacks and detailed error reporting
4. **Batch Processing**: Efficient processing of multiple images
5. **Rich Output**: JSON results, visualizations, summaries
6. **Easy Extension**: Plugin-friendly architecture for new models
7. **Production Ready**: Proper logging, documentation, testing structure

## Dependencies

### Core
- PyTorch, torchvision, PIL, numpy, scipy
- transformers, tokenizers (for BERT/RoBERTa)
- ultralytics (for YOLOv8 and SAM2)

### Utilities  
- colorlog (colored logging)
- matplotlib (visualizations)
- tqdm (progress bars)

### Development
- pytest, black, flake8 (testing and formatting)

## Configuration Examples

### Fast Processing
```python
config.model.detection_model = "yolov8n.pt"
config.model.alpha_clip_model = "ViT-B/32"  
config.model.lm_model = "bert-base-uncased"
config.generation.num_iterations = 10
```

### High Quality
```python
config.model.detection_model = "yolov8l.pt"
config.model.alpha_clip_model = "ViT-L/14"
config.model.lm_model = "roberta-base"
config.generation.num_iterations = 20
```

## Extension Points

1. **New Detection Models**: Implement detector interface
2. **New Segmentation Models**: Implement mask generator interface  
3. **New Caption Models**: Implement text generator interface
4. **Custom Configurations**: Extend Config classes
5. **Output Formats**: Add new result formatters

This structure provides a solid foundation for research, development, and production use of the GET_CAPTION system.
