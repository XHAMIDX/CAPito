#!/usr/bin/env python3
"""
CAPito Installation and Test Script
==================================

Quick setup and verification for the refactored CAPito system.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description, check=True):
    """Run a shell command with error handling."""
    print(f"⚡ {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(f"   {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr.strip()}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Requires Python 3.8+")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available - {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing Dependencies")
    print("=" * 40)
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements"):
        return False
    
    # Install CAPito in development mode
    if not run_command(f"{sys.executable} -m pip install -e .", "Installing CAPito"):
        return False
    
    return True


def setup_alphclip():
    """Setup AlphaCLIP dependency."""
    print("\n🔍 Setting up AlphaCLIP")
    print("=" * 40)
    
    alpha_clip_dir = Path("AlphaCLIP")
    if not alpha_clip_dir.exists():
        print("❌ AlphaCLIP directory not found")
        return False
    
    # Install AlphaCLIP
    os.chdir(alpha_clip_dir)
    success = run_command(f"{sys.executable} -m pip install -e .", "Installing AlphaCLIP")
    os.chdir("..")
    
    return success


def download_essential_models():
    """Download essential models for testing."""
    print("\n📥 Downloading Essential Models")
    print("=" * 40)
    
    # Check if models already exist
    models_dir = Path("models")
    essential_models = [
        "clip_b16_grit1m_fultune_8xe.pth",
        "yolov8n.pt", 
        "sam2_t.pt"
    ]
    
    existing_models = []
    missing_models = []
    
    for model in essential_models:
        model_path = models_dir / model
        if model_path.exists():
            existing_models.append(model)
            print(f"✅ {model} - Already available")
        else:
            missing_models.append(model)
            print(f"❌ {model} - Missing")
    
    if not missing_models:
        print("✅ All essential models are available!")
        return True
    
    print(f"\n⚠️  {len(missing_models)} models need to be downloaded")
    print("This may take several minutes depending on your internet connection...")
    
    # Use model management script
    try:
        return run_command(f"{sys.executable} examples/model_management.py", "Downloading models", check=False)
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return False


def run_basic_test():
    """Run basic functionality test."""
    print("\n🧪 Running Basic Test")
    print("=" * 40)
    
    # Create a simple test script
    test_script = """
import sys
sys.path.insert(0, '.')

try:
    from capito import get_default_config
    print("[SUCCESS] Import successful")
    
    config = get_default_config()
    print("[SUCCESS] Configuration created")
    
    # Test model paths
    vlm_path = config.get_model_path("vlm", config.vlm.model_name)
    detection_path = config.get_model_path("detection", config.detection.model_name)
    segmentation_path = config.get_model_path("segmentation", config.segmentation.model_name)
    
    print(f"[SUCCESS] VLM model path: {vlm_path}")
    print(f"[SUCCESS] Detection model path: {detection_path}")
    print(f"[SUCCESS] Segmentation model path: {segmentation_path}")
    
    print("[SUCCESS] Basic test passed!")
    
except Exception as e:
    print(f"[ERROR] Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    
    # Write and run test
    with open("test_capito.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    success = run_command(f"{sys.executable} test_capito.py", "Running basic test")
    
    # Cleanup test file
    try:
        os.remove("test_capito.py")
    except:
        pass
    
    return success


def create_example_image():
    """Create a simple test image if none exists."""
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Check if any example images exist
    image_extensions = ['.png', '.jpg', '.jpeg']
    existing_images = []
    for ext in image_extensions:
        existing_images.extend(examples_dir.glob(f"*{ext}"))
    
    if existing_images:
        print(f"✅ Found {len(existing_images)} example images")
        return True
    
    # Create a simple colored image for testing
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        
        # Create a 256x256 image with some simple shapes
        img = Image.new('RGB', (256, 256), 'lightblue')
        draw = ImageDraw.Draw(img)
        
        # Draw some simple shapes
        draw.rectangle([50, 50, 150, 150], fill='red', outline='black', width=2)
        draw.ellipse([120, 120, 220, 220], fill='green', outline='black', width=2)
        draw.polygon([(128, 50), (100, 100), (156, 100)], fill='yellow', outline='black', width=2)
        
        # Save the test image
        test_image_path = examples_dir / "test_image.png"
        img.save(test_image_path)
        
        print(f"✅ Created test image: {test_image_path}")
        return True
        
    except Exception as e:
        print(f"⚠️  Could not create test image: {e}")
        print("   Please add an image to the examples/ directory for testing")
        return False


def show_usage_instructions():
    """Show how to use CAPito after installation."""
    print("\n🎉 Installation Complete!")
    print("=" * 50)
    
    print("\n📋 Quick Usage Guide:")
    print("-" * 20)
    
    print("\n1. Command Line Interface:")
    print("   python main.py examples/test_image.png")
    print("   python main.py examples/ --config quality")
    print("   python main.py image.jpg --vlm-model 'ViT-L/14' --max-length 8")
    
    print("\n2. Python API:")
    print("   from capito import CAPito, get_default_config")
    print("   with CAPito(get_default_config()) as capito:")
    print("       results = capito.process_image('image.jpg')")
    
    print("\n3. Examples:")
    print("   python examples/basic_example.py")
    print("   python examples/advanced_example.py")
    print("   python examples/model_management.py")
    
    print("\n4. Configuration Presets:")
    print("   --config fast      # Optimized for speed")
    print("   --config default   # Balanced performance")
    print("   --config quality   # Best quality results")
    print("   --config vlm       # VLM-focused processing")
    
    print("\n📚 Documentation:")
    print("   README.md - Comprehensive guide")
    print("   examples/ - Usage examples")
    print("   capito/ - Source code with docstrings")


def main():
    """Main installation function."""
    print("🚀 CAPito Installation & Setup")
    print("=" * 50)
    
    # Check prerequisites
    print("\n🔍 Checking Prerequisites")
    print("=" * 30)
    
    if not check_python_version():
        return 1
    
    check_cuda()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return 1
    
    # Setup AlphaCLIP
    if not setup_alphclip():
        print("❌ Failed to setup AlphaCLIP")
        return 1
    
    # Download models
    if not download_essential_models():
        print("⚠️  Model download incomplete - you may need to download manually")
    
    # Run basic test
    if not run_basic_test():
        print("❌ Basic test failed")
        return 1
    
    # Create example image
    create_example_image()
    
    # Show usage instructions
    show_usage_instructions()
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
