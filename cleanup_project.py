#!/usr/bin/env python3
"""
Project Cleanup Script
=====================

Removes legacy files and organizes the refactored CAPito project.
"""

import os
import shutil
from pathlib import Path


def cleanup_legacy_files():
    """Remove legacy files that are no longer needed."""
    
    # Files to remove
    legacy_files = [
        "demo.py",
        "run.py", 
        "run_example.py",
        "gen_utils.py",
        "control_gen_utils.py",
        "utils.py",
        "compute_n_div.py",
        "POS_classifier.py",
        "sentiments_classifer.py",
        "migrate_models.py",
        "test_model_organization.py",
        "h origin main",  # Looks like a git artifact
        "sam2_b.pt",  # Duplicate model file
        "yolov8n.pt",  # Duplicate model file
    ]
    
    # Directories to remove  
    legacy_dirs = [
        "src",  # Old structure
        "clip",  # Legacy clip module
        "logger",  # Legacy logging
        "results",  # Old results (will use outputs/)
    ]
    
    # Documentation files to remove (keep only essential)
    docs_to_remove = [
        "INSTALLATION_GUIDE.md",
        "MODEL_ORGANIZATION.md", 
        "PROJECT_STRUCTURE.md",
    ]
    
    print("Cleaning up legacy files...")
    
    # Remove legacy files
    for file_name in legacy_files:
        file_path = Path(file_name)
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"✓ Removed: {file_name}")
            except Exception as e:
                print(f"✗ Failed to remove {file_name}: {e}")
    
    # Remove legacy directories
    for dir_name in legacy_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            try:
                shutil.rmtree(dir_path)
                print(f"✓ Removed directory: {dir_name}")
            except Exception as e:
                print(f"✗ Failed to remove directory {dir_name}: {e}")
    
    # Remove documentation files
    for doc_name in docs_to_remove:
        doc_path = Path(doc_name)
        if doc_path.exists():
            try:
                doc_path.unlink()
                print(f"✓ Removed documentation: {doc_name}")
            except Exception as e:
                print(f"✗ Failed to remove {doc_name}: {e}")


def create_outputs_directory():
    """Create outputs directory structure."""
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    subdirs = ["basic_example", "advanced_example", "batch_processing"]
    for subdir in subdirs:
        (outputs_dir / subdir).mkdir(exist_ok=True)
    
    print(f"✓ Created outputs directory structure")


def create_gitignore():
    """Create comprehensive .gitignore file."""
    gitignore_content = """# CAPito .gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyTorch
*.pth
*.pt
*.ckpt

# Models (except essential ones)
models/
!models/.gitkeep

# Outputs
outputs/
logs/
results/

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Data
*.jpg
*.jpeg
*.png
*.bmp
*.tiff
*.gif
!examples/*.jpg
!examples/*.jpeg  
!examples/*.png

# Temporary files
*.tmp
*.temp
*.log
*.cache

# Jupyter
.ipynb_checkpoints/

# Documentation builds
docs/_build/
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("✓ Created .gitignore file")


def create_models_gitkeep():
    """Create .gitkeep file in models directory."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    gitkeep_path = models_dir / ".gitkeep"
    gitkeep_path.touch()
    
    print("✓ Created models/.gitkeep")


def show_final_structure():
    """Display the final project structure."""
    print("\nFinal Project Structure:")
    print("=" * 50)
    
    structure = """
CAPito/
├── capito/                 # Main package
│   ├── core/              # Core pipeline and config
│   ├── vlm/               # AlphaCLIP VLM
│   ├── detection/         # Object detection & segmentation
│   ├── captioning/        # Caption generation
│   └── utils/             # Utilities
├── models/                # Centralized model storage
├── examples/              # Usage examples
├── outputs/               # Generated outputs
├── AlphaCLIP/            # AlphaCLIP submodule (VLM)
├── main.py               # CLI interface
├── requirements.txt      # Dependencies
├── setup.py              # Package setup
├── README.md             # Documentation
└── LICENSE               # MIT License
"""
    
    print(structure)


def main():
    """Main cleanup function."""
    print("CAPito Project Cleanup")
    print("=" * 30)
    
    # Confirm cleanup
    response = input("This will remove legacy files. Continue? (y/N): ").lower().strip()
    if response != 'y':
        print("Cleanup cancelled.")
        return
    
    try:
        cleanup_legacy_files()
        create_outputs_directory()
        create_gitignore()
        create_models_gitkeep()
        
        print("\n" + "=" * 50)
        print("✅ Cleanup completed successfully!")
        print("✅ Project has been refactored to CAPito v2.0.0")
        print("=" * 50)
        
        show_final_structure()
        
        print("\nNext Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Download models: python examples/model_management.py")
        print("3. Test installation: python examples/basic_example.py")
        print("4. Use CLI: python main.py examples/cat.png")
        
    except Exception as e:
        print(f"\n❌ Cleanup failed: {e}")
        print("Please check the error and try again.")


if __name__ == "__main__":
    main()
