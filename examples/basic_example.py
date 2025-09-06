#!/usr/bin/env python3
"""
Basic CAPito Example
===================

Simple example demonstrating basic image captioning with CAPito.
"""

from pathlib import Path
from capito import CAPito, get_default_config


def main():
    """Basic example of using CAPito."""
    
    # Setup configuration
    config = get_default_config()
    
    # Initialize CAPito
    print("Initializing CAPito...")
    capito = CAPito(config)
    print("CAPito ready!")
    
    # Process example image
    image_path = "D:\\Deep_VAD\\GET_CAPTION\\examples\\cat.png"  # Assuming this exists

    if not Path(image_path).exists():
        print(f"Example image not found: {image_path}")
        print("Please provide a valid image path")
        return
    
    print(f"Processing image: {image_path}")
    
    # Generate captions
    results = capito.process_image(
        image_path=image_path,
        output_dir="outputs/basic_example"
    )
    
    # Display results
    print("\nResults:")
    print("=" * 50)
    
    print("Generated Captions:")
    for i, caption in enumerate(results["captions"], 1):
        print(f"  {i}. {caption}")
    
    print(f"\nDetected Objects:")
    for detection in results["detections"]:
        print(f"  - {detection.class_name} (confidence: {detection.confidence:.2f})")
    
    print(f"\nProcessing Time: {results['processing_time']:.2f} seconds")
    print(f"Results saved to: outputs/basic_example")
    
    # Cleanup
    capito.cleanup()


if __name__ == "__main__":
    main()
