"""Quick test script for GET_CAPTION pipeline."""

import sys
from pathlib import Path


sys.path.append(".")

from src.main_pipeline import GetCaptionPipeline
from src.config import Config

def main():
    # Setup configuration
    config = Config()
    config.model.device = "cpu"  # Change to "cuda" if you have GPU
    # Keep memory low
    config.processing.samples_num = 1
    config.processing.max_objects_per_image = 3
    config.generation.num_iterations = 3
    config.generation.candidate_k = 10
    config.generation.sentence_len = 8
    
    # Initialize pipeline
    print("🚀 Initializing GET_CAPTION pipeline...")
    pipeline = GetCaptionPipeline(config)
    
    # Process the example image
    image_path = "examples/Screenshot 2025-07-15 173624.png"
    print(f"📸 Processing: {image_path}")
    
    # Run the pipeline
    results = pipeline.process_single_image(
        image_path=image_path,
        output_dir="results",
        save_intermediate=True
    )
    
    # Print results
    print("\n" + "="*60)
    print(" RESULTS")
    print("="*60)
    
    print(f" Processing time: {results['processing_time']:.2f}s")
    
    # Detection summary
    detection_summary = results['caption_results']['summary']
    print(f" Objects detected: {detection_summary['total_objects_detected']}")
    print(f" Successful captions: {detection_summary['successful_captions']}")
    
    # Full image caption
    if results['caption_results']['full_image_caption']:
        full_cap = results['caption_results']['full_image_caption']
        if full_cap.get('success'):
            print(f"\n Full image: {full_cap['caption']}")
    
    # Object captions
    print(f"\n Object captions:")
    for i, obj_result in enumerate(results['caption_results']['object_captions']):
        if obj_result.get('success'):
            obj_info = obj_result['object_info']
            confidence = obj_info['confidence']
            class_name = obj_info['class_name']
            caption = obj_result['caption']
            print(f"   {i+1}. {class_name} ({confidence:.2f}): {caption}")
        else:
            print(f"   {i+1}. Failed: {obj_result.get('error', 'Unknown error')}")
    
    print(f"\n Results saved to: results/")
    print("   - Check the JSON files for detailed results")
    print("   - Check the visualization images")
    print("\n Done!")

if __name__ == "__main__":
    main()
