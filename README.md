# GET_CAPTION: Unified Image Captioning with Object Detection

A comprehensive pipeline that combines object detection, segmentation, and image captioning using AlphaCLIP and ConZIC methodologies. This project detects objects in images, generates precise masks, and produces detailed captions for each detected object using masked image regions.

## Features

- **Object Detection**: YOLOv8-based object detection with configurable confidence thresholds
- **Advanced Segmentation**: SAM2 integration for precise object masks (with bbox fallback)
- **Masked Image Captioning**: AlphaCLIP + ConZIC approach for region-specific captions
- **Clean Architecture**: Modular, extensible design with comprehensive configuration
- **Batch Processing**: Process single images or entire directories
- **Rich Output**: JSON results, visualizations, and formatted text summaries

## Architecture Overview

```
Input Image → Object Detection → Mask Generation → AlphaCLIP Encoding → ConZIC Captioning → Results
     ↓              ↓                   ↓                  ↓                    ↓             ↓
   PIL Image    YOLOv8 Bboxes    SAM2/Bbox Masks    Alpha Masks      BERT+CLIP Fusion    Captions
```

### Key Components

1. **Detection Pipeline** (`src/detection/`)
   - `ObjectDetector`: YOLOv8-based object detection
   - `MaskGenerator`: SAM2 or bbox-based mask generation
   - `DetectionPipeline`: Combined detection and segmentation

2. **Captioning Pipeline** (`src/captioning/`)
   - `AlphaCLIPWrapper`: Clean AlphaCLIP interface with mask support
   - `TextGenerator`: ConZIC-style masked language modeling
   - `CaptionPipeline`: End-to-end caption generation

3. **Main Pipeline** (`src/main_pipeline.py`)
   - `GetCaptionPipeline`: Complete processing pipeline
   - Configuration management and result handling

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git LFS (for AlphaCLIP checkpoints)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd GET_CAPTION
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install the package** (optional):
```bash
pip install -e .
```

### Model Downloads

The pipeline will automatically download required models on first use:
- **YOLOv8**: Downloaded via ultralytics
- **SAM2**: Downloaded via ultralytics
- **AlphaCLIP**: Uses included checkpoints in `AlphaCLIP/`
- **BERT/RoBERTa**: Downloaded via transformers

## Quick Start

### Command Line Usage

**Process a single image**:
```bash
python -m src.main_pipeline path/to/image.jpg -o results/
```

**Process a directory**:
```bash
python -m src.main_pipeline path/to/images/ -o results/ --batch
```

**With custom device**:
```bash
python -m src.main_pipeline image.jpg --device cuda -o results/
```

### Python API

```python
from src.main_pipeline import GetCaptionPipeline
from src.config import Config

# Initialize with default config
pipeline = GetCaptionPipeline()

# Process single image
results = pipeline.process_single_image("path/to/image.jpg")

# Process batch
results = pipeline.process_batch("path/to/images/")

# Custom configuration
config = Config()
config.model.alpha_clip_model = "ViT-L/14"
config.generation.sentence_len = 10
pipeline = GetCaptionPipeline(config)
```

## Configuration

### Model Configuration

```python
config.model.alpha_clip_model = "ViT-L/14"  # ViT-B/32, ViT-B/16, ViT-L/14, RN50
config.model.lm_model = "bert-base-uncased"  # or roberta-base
config.model.detection_model = "yolov8n.pt"  # yolov8n/s/m/l.pt
config.model.sam_model = "sam2_b.pt"  # sam2_t/s/b/l.pt
config.model.device = "cuda"
```

### Generation Configuration

```python
config.generation.sentence_len = 8          # Caption length
config.generation.candidate_k = 50          # Top-k candidates
config.generation.num_iterations = 15       # Generation iterations
config.generation.alpha = 0.8               # Language model weight
config.generation.beta = 1.5                # CLIP similarity weight
config.generation.order = "shuffle"         # Generation order
```

### Processing Configuration

```python
config.processing.min_object_area = 1000    # Minimum object size
config.processing.max_objects_per_image = 10
config.processing.samples_num = 3           # Caption samples per object
```

## Output Format

### Results Structure

```json
{
  "image_name": "example.jpg",
  "processing_time": 12.34,
  "detection_results": {
    "detections": [...],
    "alpha_masks": [...],
    "visualization": "PIL Image"
  },
  "caption_results": {
    "object_captions": [
      {
        "success": true,
        "caption": "A red car parked on the street",
        "clip_score": 0.87,
        "object_info": {
          "class_name": "car",
          "confidence": 0.95,
          "bbox": [100, 150, 300, 250]
        }
      }
    ],
    "full_image_caption": {...},
    "summary": {...}
  }
}
```

### Generated Files

- `{image}_results.json`: Complete processing results
- `{image}_captions.txt`: Human-readable captions
- `{image}_detection_visualization.jpg`: Detection visualization
- `batch_summary_{timestamp}.txt`: Batch processing summary

## Performance Tips

### Hardware Optimization

- **GPU**: Use CUDA for 3-5x speedup
- **Memory**: 8GB+ GPU memory recommended for large models
- **CPU**: Multi-core CPU for batch processing

### Model Selection

- **Fast**: YOLOv8n + ViT-B/32 + bert-base
- **Balanced**: YOLOv8s + ViT-L/14 + bert-base  
- **Quality**: YOLOv8l + ViT-L/14 + roberta-base

### Parameter Tuning

- **More image-faithful captions**: Increase `beta` (1.5→3.0)
- **More fluent text**: Increase `alpha` (0.8→1.2) and `lm_temperature` (0.3→0.5)
- **Faster generation**: Reduce `num_iterations` (15→10) and `candidate_k` (50→30)

## Technical Details

### AlphaCLIP Integration

- Uses alpha masks for region-specific image encoding
- Supports all AlphaCLIP model variants (ViT-B/32, ViT-B/16, ViT-L/14, RN50)
- Automatic fallback to ViT-B/32 for invalid model names
- Proper score normalization for fusion with language model

### ConZIC Methodology

- Iterative masked token generation using BERT/RoBERTa
- CLIP-guided token selection with configurable fusion weights
- Multiple generation orders: sequential, shuffle, random, span
- Stop word filtering and position-aware token masking

### Mask Generation

- **Primary**: SAM2 from ultralytics for precise segmentation
- **Fallback**: Bounding box masks when SAM2 unavailable
- Alpha mask smoothing with configurable blur radius
- Automatic resizing for AlphaCLIP compatibility

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use smaller models
2. **SAM2 not available**: Pipeline automatically falls back to bbox masks
3. **AlphaCLIP import error**: Ensure AlphaCLIP directory is in project root
4. **Tokenizer vocab error**: Some RoBERTa models may need tokenizer fixes

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code passes linting (`black`, `flake8`)
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{get_caption_2024,
  title={GET_CAPTION: Unified Image Captioning with Object Detection},
  author={GET_CAPTION Team},
  year={2024},
  url={https://github.com/your-repo/GET_CAPTION}
}
```

## Acknowledgments

- **AlphaCLIP**: For masked image-text understanding
- **ConZIC**: For controlled image captioning methodology
- **Ultralytics**: For YOLOv8 and SAM2 integration
- **Hugging Face**: For transformer models and tokenizers
