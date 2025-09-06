#!/usr/bin/env python3
"""
Advanced CAPito Example
======================

Advanced example demonstrating configuration customization and batch processing.
"""

from pathlib import Path
from capito import CAPito, get_quality_config
from capito.core.config import CaptioningConfig


def main():
    """Advanced example with custom configuration."""
    
    # Start with quality configuration
    config = get_quality_config()
    
    # Customize captioning settings
    config.captioning = CaptioningConfig(
        max_length=8,
        candidate_k=40,
        num_iterations=12,
        alpha=0.3,  # Lower fluency weight
        beta=2.5,   # Higher image weight
        temperature=0.15,  # More focused generation
        generation_order="span",
        prompt_template="A detailed photograph showing",
        enable_control=True,
        control_type="sentiment",
        sentiment_type="positive"
    )
    
    # Enable all outputs
    config.system.save_visualizations = True
    config.system.save_results = True
    
    print("Configuration:")
    print(f"  VLM: {config.vlm.model_name}")
    print(f"  Detection: {config.detection.model_name}")
    print(f"  Segmentation: {config.segmentation.model_name}")
    print(f"  Caption Length: {config.captioning.max_length}")
    print(f"  Control: {config.captioning.control_type} ({config.captioning.sentiment_type})")
    
    # Initialize CAPito
    print("\nInitializing CAPito with custom configuration...")
    with CAPito(config) as capito:
        
        # Process multiple images
        image_dir = Path("examples")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        image_paths = [
            p for p in image_dir.iterdir() 
            if p.suffix.lower() in image_extensions
        ]
        
        if not image_paths:
            print("No example images found. Please add images to the examples/ directory")
            return
        
        print(f"\nProcessing {len(image_paths)} images...")
        
        # Batch processing
        results = capito.process_batch(
            image_paths=image_paths,
            output_dir="outputs/advanced_example"
        )
        
        # Analyze results
        print("\nBatch Processing Results:")
        print("=" * 50)
        
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            total_time = sum(r["processing_time"] for r in successful)
            avg_time = total_time / len(successful)
            print(f"Average processing time: {avg_time:.2f} seconds")
            
            # Show sample captions
            print("\nSample Generated Captions:")
            for result in successful[:3]:  # Show first 3 results
                image_name = Path(result["image_path"]).name
                print(f"\n{image_name}:")
                for i, caption in enumerate(result["captions"][:2], 1):
                    print(f"  {i}. {caption}")
        
        if failed:
            print("\nFailed Images:")
            for result in failed:
                image_name = Path(result["image_path"]).name
                print(f"  - {image_name}: {result['error']}")
        
        print(f"\nDetailed results saved to: outputs/advanced_example")
        
        # Show model information
        print("\nModel Information:")
        model_info = capito.get_model_info()
        for component, info in model_info.items():
            print(f"  {component}: {info}")


if __name__ == "__main__":
    main()
