"""Simple example of using GET_CAPTION pipeline."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from main_pipeline import GetCaptionPipeline
from config import Config


def main():
    """Run a simple example."""
    # Setup configuration
    config = Config()
    config.model.device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    config.processing.samples_num = 1  # One caption per object for speed
    config.generation.num_iterations = 10  # Fewer iterations for speed
    
    print("Initializing GET_CAPTION pipeline...")
    pipeline = GetCaptionPipeline(config)
    
    # Example image path
    image_path = "examples/cat.png"
    
    if not os.path.exists(image_path):
        print(f"Image not found at: {image_path}")
        print("Please place an image in the examples/ directory.")
        return
    
    print(f"Processing image: {image_path}")
    
    # Process the image
    results = pipeline.process_single_image(
        image_path=image_path,
        output_dir="examples/results",
        save_intermediate=True
    )
    
    # Print results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    
    print(f"Processing time: {results['processing_time']:.2f}s")
    
    # Detection summary
    detection_summary = results['caption_results']['summary']
    print(f"Objects detected: {detection_summary['total_objects_detected']}")
    print(f"Successful captions: {detection_summary['successful_captions']}")
    
    # Full image caption
    if results['caption_results']['full_image_caption']:
        full_cap = results['caption_results']['full_image_caption']
        if full_cap.get('success'):
            print(f"\nFull image caption: {full_cap['caption']}")
    
    # Object captions
    print("\nObject captions:")
    for i, obj_result in enumerate(results['caption_results']['object_captions']):
        if obj_result.get('success'):
            obj_info = obj_result['object_info']
            print(f"  {i+1}. {obj_info['class_name']} ({obj_info['confidence']:.2f}): {obj_result['caption']}")
        else:
            print(f"  {i+1}. Failed: {obj_result.get('error', 'Unknown error')}")
    
    print(f"\nResults saved to: examples/results/")
    print("Check the generated files for detailed results and visualizations.")


if __name__ == "__main__":
    main()
