#!/usr/bin/env python3
"""
Test script for the new model organization structure.

This script verifies that:
1. The new directory structure is created correctly
2. Configuration can access model paths
3. Model manager can organize existing models
4. No conflicts exist in the current setup
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_directory_structure():
    """Test that the new directory structure exists."""
    print("Testing directory structure...")
    
    required_dirs = [
        "models",
        "models/alpha_clip",
        "models/alpha_clip/checkpoints",
        "models/detection",
        "models/detection/yolo",
        "models/detection/sam",
        "models/language",
        "models/legacy"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} - MISSING")
            return False
    
    return True

def test_configuration():
    """Test that configuration can access model paths."""
    print("\nTesting configuration...")
    
    try:
        from config import Config, ModelPathsConfig
        
        config = Config()
        model_paths = config.model_paths
        
        print(f"  ✓ Models root: {model_paths.models_root}")
        print(f"  ✓ AlphaCLIP checkpoints: {model_paths.alpha_clip_checkpoints}")
        print(f"  ✓ YOLO models: {model_paths.yolo_models}")
        print(f"  ✓ SAM2 models: {model_paths.sam_models}")
        print(f"  ✓ Language models: {model_paths.language_models_root}")
        print(f"  ✓ Legacy models: {model_paths.legacy_root}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Configuration error: {e}")
        return False

def test_model_manager():
    """Test that model manager can be initialized."""
    print("\nTesting model manager...")
    
    try:
        from utils.model_manager import ModelManager
        
        manager = ModelManager()
        print(f"  ✓ Model manager initialized")
        print(f"  ✓ Models root: {manager.model_paths.models_root}")
        
        # Test listing available models
        available = manager.list_available_models()
        print(f"  ✓ Available models: {len(available)} categories")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Model manager error: {e}")
        return False

def test_existing_models():
    """Test that existing models can be found and organized."""
    print("\nTesting existing models...")
    
    try:
        from utils.model_manager import setup_model_environment
        
        manager = setup_model_environment()
        print(f"  ✓ Model environment setup completed")
        
        # Check what models were moved
        available = manager.list_available_models()
        total_models = sum(len(models) for models in available.values())
        print(f"  ✓ Total models found: {total_models}")
        
        for category, models in available.items():
            if models:
                print(f"    - {category}: {', '.join(models)}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Model environment error: {e}")
        return False

def test_no_conflicts():
    """Test that no conflicts exist in the current setup."""
    print("\nTesting for conflicts...")
    
    conflicts = []
    
    # Check for duplicate model files
    model_files = []
    
    # Check AlphaCLIP checkpoints
    alpha_clip_old = Path("AlphaCLIP/checkpoints")
    alpha_clip_new = Path("models/alpha_clip/checkpoints")
    
    if alpha_clip_old.exists() and alpha_clip_new.exists():
        old_files = set(f.name for f in alpha_clip_old.glob("*.pth"))
        new_files = set(f.name for f in alpha_clip_new.glob("*.pth"))
        
        if old_files and new_files:
            conflicts.append(f"AlphaCLIP checkpoints exist in both old and new locations")
            print(f"  ⚠ AlphaCLIP checkpoints in both locations:")
            print(f"    Old: {list(old_files)}")
            print(f"    New: {list(new_files)}")
    
    # Check for scattered model files
    scattered_locations = [
        "AlphaCLIP/checkpoints",
        "legacy/ConZIC",
        "checkpoints",
        "models"
    ]
    
    for location in scattered_locations:
        if os.path.exists(location):
            model_files.extend(list(Path(location).rglob("*.pth")))
            model_files.extend(list(Path(location).rglob("*.pt")))
    
    if len(model_files) > 0:
        print(f"  ✓ Found {len(model_files)} model files across project")
    
    if not conflicts:
        print("  ✓ No conflicts detected")
        return True
    else:
        print(f"  ⚠ {len(conflicts)} potential conflicts found")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("GET_CAPTION Model Organization Test")
    print("=" * 60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Configuration", test_configuration),
        ("Model Manager", test_model_manager),
        ("Existing Models", test_existing_models),
        ("Conflict Check", test_no_conflicts)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Model organization is working correctly.")
        print("\nNext steps:")
        print("1. Run: python migrate_models.py --list")
        print("2. Run: python migrate_models.py --download-all")
        print("3. Run: python migrate_models.py --cleanup")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
