"""Text processing utilities."""

import re
from typing import List, Dict, Any


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Fix common punctuation issues
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([,.!?;:])\s*([,.!?;:])', r'\1', text)
    
    # Ensure sentence ends with punctuation
    if text and text[-1] not in '.!?':
        text += '.'
    
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    
    return text


def format_caption(
    caption: str, 
    object_name: str = None, 
    confidence: float = None
) -> str:
    """Format caption with optional object information."""
    caption = clean_text(caption)
    
    if object_name and confidence:
        # Add object information as prefix
        prefix = f"[{object_name} ({confidence:.2f})] "
        caption = prefix + caption
    
    return caption


def create_summary_report(results: List[Dict[str, Any]]) -> str:
    """Create a summary report from captioning results."""
    if not results:
        return "No results to summarize."
    
    report_lines = [
        "=== GET_CAPTION Results Summary ===",
        f"Total images processed: {len(results)}",
        ""
    ]
    
    for i, result in enumerate(results, 1):
        report_lines.append(f"Image {i}: {result.get('image_name', 'Unknown')}")
        
        if 'objects' in result:
            report_lines.append(f"  Objects detected: {len(result['objects'])}")
            for j, obj in enumerate(result['objects'], 1):
                obj_name = obj.get('class_name', 'Unknown')
                confidence = obj.get('confidence', 0.0)
                caption = obj.get('caption', 'No caption')
                report_lines.append(f"    {j}. {obj_name} ({confidence:.2f}): {caption}")
        
        if 'full_caption' in result:
            report_lines.append(f"  Full image caption: {result['full_caption']}")
        
        report_lines.append("")
    
    return "\n".join(report_lines)
