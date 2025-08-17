# GET_CAPTION Installation Guide

## Quick Setup

### 1. Clone and Setup Environment
```bash
git clone <repository-url>
cd GET_CAPTION
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Test Installation
```bash
# Place a test image in examples/
python examples/simple_example.py
```

## Server Deployment

### Moving Code to Server (Checkpoints Cleanup)

When deploying to a server, you should **remove all checkpoints** to save space and ensure clean deployment:

```bash
# Remove all model checkpoints before pushing to server
rm -rf AlphaCLIP/checkpoints/*
rm -f *.pt *.pth
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/torch/
rm -rf ~/.cache/ultralytics/

# Clean up any downloaded models
find . -name "*.pt" -delete
find . -name "*.pth" -delete
```

### Post-Deployment Checkpoint Downloads

After deploying to your server, download all required checkpoints:

#### 1. AlphaCLIP Checkpoints (Required)

```bash
# Create checkpoints directory
mkdir -p AlphaCLIP/checkpoints

# Download all available AlphaCLIP models
cd AlphaCLIP/checkpoints

# GRIT-1M trained models
wget -O clip_b16_grit1m_fultune_8xe.pth "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_b16_grit1m_fultune_8xe.pth"
wget -O clip_l14_grit1m_fultune_8xe.pth "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_grit1m_fultune_8xe.pth"
wget -O clip_l14_336_grit1m_fultune_8xe.pth "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_336_grit1m_fultune_8xe.pth"

# GRIT-20M trained models (higher quality)
wget -O clip_b16_grit20m_fultune_2xe.pth "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_b16_grit20m_fultune_2xe.pth"
wget -O clip_l14_grit20m_fultune_2xe.pth "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_grit20m_fultune_2xe.pth"
wget -O clip_l14_336_grit20m_fultune_2xe.pth "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_336_grit20m_fultune_2xe.pth"

# Alternative: Use curl if wget not available
# curl -L -o clip_b16_grit1m_fultune_8xe.pth "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_b16_grit1m_fultune_8xe.pth"

cd ../..
```

**Total AlphaCLIP size**: ~2.5GB

#### 2. YOLOv8 Models (Auto-downloaded)

YOLOv8 models are automatically downloaded on first use, but you can pre-download them:

```bash
# Create models directory
mkdir -p models

# Download all YOLOv8 variants
cd models

# Detection models
wget -O yolov8n.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"      # ~6MB
wget -O yolov8s.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"      # ~22MB
wget -O yolov8m.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"      # ~52MB
wget -O yolov8l.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt"      # ~88MB
wget -O yolov8x.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"      # ~136MB

# Segmentation models
wget -O yolov8n-seg.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt"  # ~6MB
wget -O yolov8s-seg.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt"  # ~22MB
wget -O yolov8m-seg.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt"  # ~52MB
wget -O yolov8l-seg.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt"  # ~88MB

cd ..
```

**Total YOLOv8 size**: ~500MB

#### 3. SAM2 Models (Auto-downloaded)

SAM2 models are automatically downloaded, but you can pre-download them:

```bash
cd models

# Download all SAM2 variants
wget -O sam2_t.pt "https://github.com/ultralytics/assets/releases/download/v8.1.0/sam2_t.pt"      # ~39MB
wget -O sam2_s.pt "https://github.com/ultralytics/assets/releases/download/v8.1.0/sam2_s.pt"      # ~46MB
wget -O sam2_b.pt "https://github.com/ultralytics/assets/releases/download/v8.1.0/sam2_b.pt"      # ~159MB
wget -O sam2_l.pt "https://github.com/ultralytics/assets/releases/download/v8.1.0/sam2_l.pt"      # ~224MB

cd ..
```

**Total SAM2 size**: ~470MB

#### 4. Language Models (Auto-downloaded)

Transformers models are automatically downloaded, but you can pre-download them:

```bash
# Pre-download all language models
python -c "
from transformers import AutoTokenizer, AutoModel

# BERT variants
print('Downloading BERT models...')
AutoTokenizer.from_pretrained('bert-base-uncased')
AutoModel.from_pretrained('bert-base-uncased')
AutoTokenizer.from_pretrained('bert-large-uncased')
AutoModel.from_pretrained('bert-large-uncased')

# RoBERTa variants
print('Downloading RoBERTa models...')
AutoTokenizer.from_pretrained('roberta-base')
AutoModel.from_pretrained('roberta-base')
AutoTokenizer.from_pretrained('roberta-large')
AutoModel.from_pretrained('roberta-large')

# DistilBERT (faster alternative)
print('Downloading DistilBERT...')
AutoTokenizer.from_pretrained('distilbert-base-uncased')
AutoModel.from_pretrained('distilbert-base-uncased')

print('All language models downloaded!')
"
```

**Total Language Models size**: ~2.5GB

#### 5. Verification Script

Create a verification script to check all downloads:

```bash
# Create verification script
cat > verify_downloads.py << 'EOF'
#!/usr/bin/env python3
import os
import torch
from pathlib import Path

def check_file_size(filepath, expected_mb_min):
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        status = "✓" if size_mb >= expected_mb_min else "✗"
        print(f"{status} {filepath}: {size_mb:.1f}MB")
        return True
    else:
        print(f"✗ {filepath}: NOT FOUND")
        return False

def main():
    print("=== GET_CAPTION Model Verification ===\n")
    
    # Check AlphaCLIP models
    print("AlphaCLIP Models:")
    check_file_size("AlphaCLIP/checkpoints/clip_b16_grit1m_fultune_8xe.pth", 300)
    check_file_size("AlphaCLIP/checkpoints/clip_l14_grit1m_fultune_8xe.pth", 800)
    check_file_size("AlphaCLIP/checkpoints/clip_l14_336_grit1m_fultune_8xe.pth", 800)
    
    print("\nYOLOv8 Models:")
    check_file_size("models/yolov8n.pt", 5)
    check_file_size("models/yolov8s.pt", 20)
    check_file_size("models/yolov8m.pt", 50)
    check_file_size("models/yolov8l.pt", 80)
    
    print("\nSAM2 Models:")
    check_file_size("models/sam2_t.pt", 35)
    check_file_size("models/sam2_s.pt", 40)
    check_file_size("models/sam2_b.pt", 150)
    check_file_size("models/sam2_l.pt", 200)
    
    print("\nLanguage Models (check cache):")
    cache_dir = Path.home() / ".cache" / "huggingface"
    if cache_dir.exists():
        cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        cache_mb = cache_size / (1024 * 1024)
        print(f"✓ HuggingFace cache: {cache_mb:.1f}MB")
    else:
        print("✗ HuggingFace cache not found")
    
    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    main()
EOF

# Make executable and run
chmod +x verify_downloads.py
python verify_downloads.py
```

#### 6. Complete Download Script

For convenience, create a complete download script:

```bash
cat > download_all_models.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting GET_CAPTION model downloads..."

# Create directories
mkdir -p AlphaCLIP/checkpoints
mkdir -p models

# Download AlphaCLIP models
echo "📥 Downloading AlphaCLIP models..."
cd AlphaCLIP/checkpoints
wget -O clip_b16_grit1m_fultune_8xe.pth "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_b16_grit1m_fultune_8xe.pth"
wget -O clip_l14_grit1m_fultune_8xe.pth "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_grit1m_fultune_8xe.pth"
wget -O clip_l14_336_grit1m_fultune_8xe.pth "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_336_grit1m_fultune_8xe.pth"
cd ../..

# Download YOLOv8 models
echo "📥 Downloading YOLOv8 models..."
cd models
wget -O yolov8n.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
wget -O yolov8s.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
wget -O yolov8m.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"
wget -O yolov8l.pt "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt"
cd ..

# Download SAM2 models
echo "📥 Downloading SAM2 models..."
cd models
wget -O sam2_t.pt "https://github.com/ultralytics/assets/releases/download/v8.1.0/sam2_t.pt"
wget -O sam2_s.pt "https://github.com/ultralytics/assets/releases/download/v8.1.0/sam2_s.pt"
wget -O sam2_b.pt "https://github.com/ultralytics/assets/releases/download/v8.1.0/sam2_b.pt"
wget -O sam2_l.pt "https://github.com/ultralytics/assets/releases/download/v8.1.0/sam2_l.pt"
cd ..

# Download language models
echo "📥 Downloading language models..."
python -c "
from transformers import AutoTokenizer, AutoModel
models = ['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased']
for model in models:
    print(f'Downloading {model}...')
    AutoTokenizer.from_pretrained(model)
    AutoModel.from_pretrained(model)
print('All language models downloaded!')
"

echo "✅ All models downloaded successfully!"
echo "📊 Total size: ~6GB"
echo "🔍 Run 'python verify_downloads.py' to verify all downloads"
EOF

chmod +x download_all_models.sh
./download_all_models.sh
```

**Total download size**: ~6GB

## Detailed Installation

### System Requirements
- **Python**: 3.8+ (3.9+ recommended)
- **GPU**: CUDA-compatible GPU with 6GB+ VRAM (optional but recommended)
- **RAM**: 8GB+ system RAM
- **Storage**: 5GB+ for models and checkpoints

### Dependencies Breakdown

#### Core ML Libraries
```bash
pip install torch>=2.0.0 torchvision>=0.15.0
pip install transformers>=4.20.0 tokenizers>=0.13.0
```

#### Computer Vision
```bash
pip install ultralytics>=8.0.0  # YOLOv8 and SAM2
pip install opencv-python>=4.5.0
pip install pillow>=9.0.0
```

#### Scientific Computing
```bash
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install matplotlib>=3.5.0
```

#### Utilities
```bash
pip install colorlog>=6.0.0
pip install tqdm>=4.60.0
pip install pyyaml>=6.0
```

### Model Downloads

Models are downloaded automatically on first use:

#### YOLOv8 Models (via ultralytics)
- `yolov8n.pt` (~6MB) - Fastest
- `yolov8s.pt` (~22MB) - Small  
- `yolov8m.pt` (~52MB) - Medium
- `yolov8l.pt` (~88MB) - Large

#### SAM2 Models (via ultralytics)
- `sam2_t.pt` (~39MB) - Tiny
- `sam2_s.pt` (~46MB) - Small
- `sam2_b.pt` (~159MB) - Base
- `sam2_l.pt` (~224MB) - Large

#### AlphaCLIP Models (included)
- Pre-downloaded in `AlphaCLIP/checkpoints/`
- ViT-B/32, ViT-B/16, ViT-L/14, RN50 variants

#### Language Models (via transformers)
- `bert-base-uncased` (~440MB)
- `roberta-base` (~500MB)
- Downloaded to `~/.cache/huggingface/`

### Verification

#### Test Core Components
```python
# Test AlphaCLIP
from src.captioning import AlphaCLIPWrapper
clip = AlphaCLIPWrapper("ViT-B/32", device="cpu")
print("✓ AlphaCLIP loaded")

# Test Detection
from src.detection import ObjectDetector  
detector = ObjectDetector("yolov8n.pt", device="cpu")
print("✓ YOLOv8 loaded")

# Test Language Model
from src.captioning import TextGenerator
generator = TextGenerator("bert-base-uncased", device="cpu") 
print("✓ BERT loaded")
```

#### Full Pipeline Test
```python
from src.main_pipeline import GetCaptionPipeline
from src.config import Config

config = Config()
config.model.device = "cpu"  # or "cuda"
pipeline = GetCaptionPipeline(config)
print("✓ Full pipeline initialized")
```

## Troubleshooting

### Quick Fix for SHA256 Error
If you're getting "SHA256 checksum does not match" error right now:

```bash
# 1. Clear corrupted downloads
rm -rf ~/.cache/clip/*
rm -rf AlphaCLIP/checkpoints/*

# 2. Download with curl (more reliable)
mkdir -p AlphaCLIP/checkpoints
cd AlphaCLIP/checkpoints
curl -L -o clip_b16_grit1m_fultune_8xe.pth "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_b16_grit1m_fultune_8xe.pth"
cd ../..

# 3. Test if it works
python run_example.py
```

### Common Issues

#### 1. CUDA Out of Memory
```python
# Reduce model sizes
config.model.detection_model = "yolov8n.pt"  # Instead of yolov8l.pt
config.model.alpha_clip_model = "ViT-B/32"   # Instead of ViT-L/14
config.processing.max_objects_per_image = 5  # Instead of 10
```

#### 2. Ultralytics Import Error
```bash
pip install ultralytics>=8.0.0
# Or for development version:
pip install git+https://github.com/ultralytics/ultralytics.git
```

#### 3. AlphaCLIP Import Error
- Ensure `AlphaCLIP/` directory exists in project root
- Check that `AlphaCLIP/alpha_clip/__init__.py` exists
- Verify Python path includes AlphaCLIP directory

#### 4. SAM2 Not Available
- Pipeline automatically falls back to bounding box masks
- No action needed, but segmentation quality will be reduced

#### 5. Transformers Model Download Issues
```bash
# Pre-download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base-uncased')"
```

#### 6. AlphaCLIP SHA256 Checksum Error
If you encounter "SHA256 checksum does not match" errors:

```bash
# Clear corrupted downloads
rm -rf ~/.cache/clip/*
rm -rf AlphaCLIP/checkpoints/*

# Alternative 1: Use direct download with curl (more reliable)
cd AlphaCLIP/checkpoints
curl -L -o clip_b16_grit1m_fultune_8xe.pth "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_b16_grit1m_fultune_8xe.pth"
curl -L -o clip_l14_grit1m_fultune_8xe.pth "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_grit1m_fultune_8xe.pth"

# Alternative 2: Use aria2c for better download reliability
aria2c -x 16 -s 16 "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_b16_grit1m_fultune_8xe.pth"
aria2c -x 16 -s 16 "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_grit1m_fultune_8xe.pth"

# Alternative 3: Manual download from browser
# Visit: https://openxlab.org.cn/models/detail/SunzeY/AlphaCLIP
# Download manually and place in AlphaCLIP/checkpoints/

cd ../..

# Verify file integrity
ls -la AlphaCLIP/checkpoints/
# Expected sizes:
# clip_b16_grit1m_fultune_8xe.pth: ~330MB
# clip_l14_grit1m_fultune_8xe.pth: ~800MB
```

#### 7. Network/Download Issues
For slow or unreliable connections:

```bash
# Use multiple download attempts
for i in {1..3}; do
    echo "Download attempt $i..."
    wget --timeout=300 --tries=3 -O clip_b16_grit1m_fultune_8xe.pth "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_b16_grit1m_fultune_8xe.pth" && break
    echo "Attempt $i failed, retrying..."
    sleep 5
done

# Or use rsync if available
rsync -av --progress rsync://openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/ AlphaCLIP/checkpoints/
```

#### 8. Bypass SHA256 Verification (Use with caution)
If checksum issues persist, you can temporarily bypass verification:

```python
# Edit AlphaCLIP/alpha_clip/alpha_clip.py
# Find line ~70 and comment out the SHA256 check:

# Before (line ~70):
# if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
#     raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

# After (comment out):
# if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
#     # raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")
#     print(f"Warning: SHA256 mismatch for {download_target}")
#     pass
```

**Note**: Only bypass SHA256 verification if you trust the download source and understand the security implications.

### Performance Optimization

#### GPU Setup
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
```

#### Memory Management
```python
# Enable memory efficient attention
import torch
torch.backends.cuda.enable_flash_sdp(True)

# Clear cache between batches  
torch.cuda.empty_cache()
```

#### Model Selection for Different Hardware

**Low-end (4GB GPU or CPU)**:
```python
config.model.detection_model = "yolov8n.pt"
config.model.sam_model = "sam2_t.pt" 
config.model.alpha_clip_model = "ViT-B/32"
config.model.lm_model = "bert-base-uncased"
config.processing.max_objects_per_image = 3
```

**Mid-range (8GB GPU)**:
```python
config.model.detection_model = "yolov8s.pt"
config.model.sam_model = "sam2_s.pt"
config.model.alpha_clip_model = "ViT-B/16" 
config.model.lm_model = "bert-base-uncased"
config.processing.max_objects_per_image = 5
```

**High-end (16GB+ GPU)**:
```python
config.model.detection_model = "yolov8l.pt"
config.model.sam_model = "sam2_l.pt"
config.model.alpha_clip_model = "ViT-L/14"
config.model.lm_model = "roberta-base"
config.processing.max_objects_per_image = 10
```

## Development Setup

### Additional Dev Dependencies
```bash
pip install pytest>=7.0.0 black>=22.0.0 flake8>=4.0.0
```

### Pre-commit Hooks
```bash
pip install pre-commit
pre-commit install
```

### Running Tests
```bash
pytest tests/ -v
```

### Code Formatting
```bash
black src/ examples/
flake8 src/ examples/
```

## Docker Setup (Optional)

```dockerfile
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "-m", "src.main_pipeline", "--help"]
```

## Support

If you encounter issues:
1. Check this guide for common solutions
2. Verify system requirements
3. Test individual components
4. Check GitHub issues
5. Create new issue with error details and system info
