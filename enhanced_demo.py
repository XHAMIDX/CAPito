#!/usr/bin/env python3
"""
Enhanced CAPito Demo
===================

Comprehensive demonstration of the enhanced CAPito system with:
- Object detection and segmentation
- Depth estimation and pose analysis
- Individual object captioning  
- Scene graph generation
- Relationship analysis
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from capito import CAPito
from capito.core.config import get_default_config, get_fast_config, get_quality_config


def main():
    parser = argparse.ArgumentParser(description="Enhanced CAPito Demo")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--output-dir", default="outputs/enhanced_demo", 
                       help="Output directory for results")
    parser.add_argument("--config", choices=["default", "fast", "quality"], 
                       default="default", help="Configuration preset")
    parser.add_argument("--save-graph", action="store_true", 
                       help="Save graph visualization")
    parser.add_argument("--save-results", action="store_true", default=True,
                       help="Save JSON results")
    parser.add_argument("--save-visualizations", action="store_true", default=True,
                       help="Save visualization images")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select configuration
    print(f"Using {args.config} configuration...")
    if args.config == "fast":
        config = get_fast_config()
    elif args.config == "quality":
        config = get_quality_config()
    else:
        config = get_default_config()
    
    # Enable graph generation
    config.graph.enable_graph = True
    config.graph.save_graph_visualization = args.save_graph
    
    # Override output directory
    config.system.output_dir = args.output_dir
    config.system.save_results = args.save_results
    config.system.save_visualizations = args.save_visualizations
    
    print("=" * 60)
    print("ENHANCED CAPITO DEMO")
    print("=" * 60)
    print(f"Input image: {args.image_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Configuration: {args.config}")
    print()
    
    try:
        # Initialize CAPito pipeline
        print("Initializing enhanced CAPito pipeline...")
        capito = CAPito(config)
        print("✓ CAPito pipeline initialized successfully")
        print()
        
        # Process the image
        print("Processing image...")
        results = capito.process_image(
            image_path=args.image_path,
            output_dir=args.output_dir,
            save_visualizations=args.save_visualizations,
            save_results=args.save_results
        )
        
        # Display results
        print()
        print("=" * 60)
        print("PROCESSING RESULTS")
        print("=" * 60)
        
        # Basic statistics
        processing_time = results.get("processing_time", 0)
        num_objects = results["metadata"].get("num_objects", 0)
        num_humans = results["metadata"].get("num_humans", 0)
        has_graph = results["metadata"].get("has_graph", False)
        has_depth = results["metadata"].get("has_depth", False)
        
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Objects detected: {num_objects}")
        print(f"Humans detected: {num_humans}")
        print(f"Depth estimation: {'✓' if has_depth else '✗'}")
        print(f"Scene graph: {'✓' if has_graph else '✗'}")
        print()
        
        # Object details
        if results.get("enhanced_objects"):
            print("DETECTED OBJECTS:")
            print("-" * 40)
            for i, obj in enumerate(results["enhanced_objects"], 1):
                class_name = obj.get("class_name", "unknown")
                confidence = obj.get("confidence", 0)
                depth = obj.get("depth", 0)
                caption = obj.get("caption", "No caption")
                has_pose = obj.get("pose") is not None
                
                print(f"{i}. {class_name} (confidence: {confidence:.3f})")
                print(f"   Depth: {depth:.3f}")
                if has_pose:
                    print(f"   Pose: Available")
                print(f"   Caption: {caption}")
                print()
        
        # Scene summary
        scene_summary = results.get("scene_summary", "")
        if scene_summary:
            print("SCENE SUMMARY:")
            print("-" * 40)
            print(scene_summary)
            print()
        
        # Graph analysis
        graph_analysis = results.get("graph_analysis")
        if graph_analysis:
            print("GRAPH ANALYSIS:")
            print("-" * 40)
            
            basic_metrics = graph_analysis.get("basic_metrics", {})
            object_dist = graph_analysis.get("object_distribution", {})
            centrality = graph_analysis.get("centrality_metrics", {})
            
            print(f"Graph nodes: {basic_metrics.get('num_nodes', 0)}")
            print(f"Graph edges: {basic_metrics.get('num_edges', 0)}")
            print(f"Graph density: {basic_metrics.get('density', 0):.3f}")
            print(f"Connected: {'Yes' if basic_metrics.get('is_connected', False) else 'No'}")
            
            most_common = object_dist.get("most_common_class", "")
            if most_common:
                print(f"Most common object: {most_common}")
            
            central_object = centrality.get("most_central_object", "")
            if central_object:
                print(f"Central object: {central_object}")
            print()
        
        # File outputs
        print("OUTPUT FILES:")
        print("-" * 40)
        output_path = Path(args.output_dir)
        image_stem = Path(args.image_path).stem
        
        if args.save_results:
            json_file = output_path / f"{image_stem}_results.json"
            print(f"✓ Results saved: {json_file}")
        
        if args.save_visualizations:
            viz_file = output_path / f"{image_stem}_visualization.jpg"
            print(f"✓ Visualization saved: {viz_file}")
        
        if args.save_graph and has_graph:
            graph_file = output_path / f"{image_stem}_graph.png"
            print(f"✓ Graph visualization saved: {graph_file}")
        
        print()
        print("=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        # Cleanup
        capito.cleanup()
        
        return 0
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
