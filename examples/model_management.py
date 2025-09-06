#!/usr/bin/env python3
"""
Model Management Example
=======================

Example demonstrating model management capabilities.
"""

from capito.utils import ModelManager


def main():
    """Example of model management."""
    
    # Initialize model manager
    manager = ModelManager()
    
    print("CAPito Model Management")
    print("=" * 40)
    
    # List available models
    print("\nModel Status:")
    models_status = manager.list_available_models()
    
    for model_name, status in models_status.items():
        available = "✓" if status["available"] else "✗"
        size_info = f" ({status['size_mb']:.1f}MB)" if status["size_mb"] else ""
        print(f"  {available} {model_name} [{status['type']}]{size_info}")
    
    # Download missing models
    missing_models = [
        name for name, status in models_status.items() 
        if not status["available"]
    ]
    
    if missing_models:
        print(f"\nMissing models: {len(missing_models)}")
        
        download = input("Download missing models? (y/n): ").lower().strip()
        if download == 'y':
            print("\nDownloading models...")
            
            for model_name in missing_models:
                try:
                    print(f"Downloading {model_name}...")
                    manager.download_model(model_name)
                    print(f"✓ {model_name} downloaded successfully")
                except Exception as e:
                    print(f"✗ Failed to download {model_name}: {e}")
    
    else:
        print("\n✓ All models are available!")
    
    # Show model details
    print("\nModel Details:")
    for model_name in ["clip_b16_grit1m_fultune_8xe.pth", "yolov8n.pt", "sam2_t.pt"]:
        try:
            info = manager.get_model_info(model_name)
            print(f"\n{model_name}:")
            print(f"  Type: {info['type']}")
            print(f"  Available: {info['available']}")
            if info['available']:
                print(f"  Size: {info['size_mb']:.1f} MB")
                print(f"  Path: {info['path']}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Migration example
    print("\nMigration from legacy locations:")
    print("This would check for models in:")
    print("  - Model/")
    print("  - AlphaCLIP/checkpoints/") 
    print("  - .")
    print("And copy them to the centralized models/ directory")
    
    migrate = input("Run migration? (y/n): ").lower().strip()
    if migrate == 'y':
        manager.migrate_models("Model")
        manager.migrate_models("AlphaCLIP/checkpoints")
        manager.migrate_models(".")
        print("Migration completed!")


if __name__ == "__main__":
    main()
