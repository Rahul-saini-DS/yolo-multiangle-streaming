"""
Universal postprocessing module for YOLO inference results
Uses Ultralytics built-in methods to minimize custom logic
Handles visualization and structured output extraction
"""
import cv2
import numpy as np
import time
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path


def draw_results(results: List[Any], save: bool = False, project: str = "runs/predict", name: str = "exp") -> List[np.ndarray]:
    """
    Minimal result drawing using Ultralytics built-in methods
    
    Args:
        results: List of YOLO Results objects
        save: Whether to save annotated images using Ultralytics
        project: Project directory for saving
        name: Experiment name for saving
        
    Returns:
        List of annotated images (BGR format ready for OpenCV/Streamlit)
    """
    # Use Ultralytics built-in save if requested
    if save:
        for r in results:
            if r is not None:
                r.save(dir=f"{project}/{name}")
    
    # Use Ultralytics built-in plot - returns BGR ready for display
    return [r.plot() for r in results if r is not None]


def summarize_results(results: List[Any], want_ultralytics_json: bool = True) -> List[Dict[str, Any]]:
    """
    Minimal structured data extraction using current Ultralytics methods
    
    Args:
        results: List of YOLO Results objects  
        want_ultralytics_json: Use Ultralytics built-in methods (recommended)
        
    Returns:
        List of summary dictionaries 
    """
    if want_ultralytics_json:
        # Use current Ultralytics approach - direct attribute access is most reliable
        summaries = []
        for i, result in enumerate(results):
            if result is None:
                summaries.append({'source_index': i, 'detections': [], 'total_detections': 0})
                continue
            
            # Use direct Ultralytics result attributes (most stable approach)
            summary = {
                'source_index': i,
                'detections': [],
                'total_detections': 0,
                'task_type': 'unknown'
            }
            
            # Detection/Segmentation task
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                summary['task_type'] = 'detection'
                summary['total_detections'] = len(result.boxes)
                
                # Extract detection data using Ultralytics attributes
                boxes_data = result.boxes.xyxy.cpu().numpy()
                conf_data = result.boxes.conf.cpu().numpy()
                cls_data = result.boxes.cls.cpu().numpy()
                names = result.names if hasattr(result, 'names') else {}
                
                for j in range(len(boxes_data)):
                    detection = {
                        'bbox': boxes_data[j].tolist(),
                        'confidence': float(conf_data[j]),
                        'class_id': int(cls_data[j]),
                        'name': names.get(int(cls_data[j]), f"class_{int(cls_data[j])}")
                    }
                    summary['detections'].append(detection)
                
                # Check for segmentation masks
                if hasattr(result, 'masks') and result.masks is not None:
                    summary['task_type'] = 'segmentation'
                    summary['masks_count'] = len(result.masks)
            
            # Pose estimation task
            elif hasattr(result, 'keypoints') and result.keypoints is not None and len(result.keypoints) > 0:
                summary['task_type'] = 'pose'
                summary['total_detections'] = len(result.keypoints)
                summary['keypoints_count'] = len(result.keypoints)
            
            # Classification task  
            elif hasattr(result, 'probs') and result.probs is not None:
                summary['task_type'] = 'classification'
                summary['total_detections'] = 1
                if hasattr(result.probs, 'top1'):
                    summary['top1_class'] = int(result.probs.top1)
                    summary['top1_conf'] = float(result.probs.top1conf)
            
            summaries.append(summary)
        return summaries
    
    # Fallback minimal custom format (only if you need specific format)
    summaries = []
    for i, result in enumerate(results):
        if result is None:
            summaries.append({'source_index': i, 'detections': [], 'total_detections': 0})
            continue
            
        summary = {
            'source_index': i,
            'total_detections': 0,
            'task_type': 'unknown'
        }
        
        # Detect task and get basic counts
        if hasattr(result, 'boxes') and result.boxes is not None:
            summary['total_detections'] = len(result.boxes)
            summary['task_type'] = 'segmentation' if hasattr(result, 'masks') and result.masks is not None else 'detection'
        elif hasattr(result, 'keypoints') and result.keypoints is not None:
            summary['total_detections'] = len(result.keypoints)
            summary['task_type'] = 'pose'
        elif hasattr(result, 'probs') and result.probs is not None:
            summary['total_detections'] = 1
            summary['task_type'] = 'classification'
            
        summaries.append(summary)
    
    return summaries


def create_batch_summary(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Minimal batch summary from Ultralytics JSON or custom summaries
    
    Args:
        summaries: List of result summaries (from summarize_results)
        
    Returns:
        Batch summary dictionary
    """
    if not summaries:
        return {'total_sources': 0, 'total_detections': 0}
    
    # Handle both Ultralytics JSON format and custom format
    total_detections = 0
    detected_classes = set()
    
    for summary in summaries:
        # Ultralytics .tojson() format
        if isinstance(summary, list):  # Ultralytics returns list of detections
            total_detections += len(summary)
            for detection in summary:
                if 'name' in detection:
                    detected_classes.add(detection['name'])
        # Custom format
        elif 'total_detections' in summary:
            total_detections += summary.get('total_detections', 0)
            if 'classes_detected' in summary:
                detected_classes.update(summary['classes_detected'])
    
    return {
        'total_sources': len(summaries),
        'total_detections': total_detections,
        'unique_classes': len(detected_classes),
        'detected_classes': list(detected_classes)
    }


def save_summaries(summaries: List[Dict[str, Any]], out_dir: str = "runs/predict/json") -> None:
    """
    Simple summary saver (only needed if you don't use Ultralytics save=True)
    
    Args:
        summaries: List of summary dictionaries
        out_dir: Output directory for JSON files
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    
    for i, summary in enumerate(summaries):
        json_file = Path(out_dir) / f"summary_{i}_{ts}.json"
        json_file.write_text(json.dumps(summary, indent=2))


def process_results(results: List[Any], 
                   save: bool = False, 
                   project: str = "runs/predict", 
                   name: str = "exp", 
                   want_json: bool = True) -> Tuple[List[np.ndarray], Optional[List[Dict[str, Any]]]]:
    """
    Minimal all-in-one result processor using Ultralytics best practices
    
    Args:
        results: List of YOLO Results objects
        save: Use Ultralytics built-in saving
        project: Project directory 
        name: Experiment name
        want_json: Whether to extract JSON summaries
        
    Returns:
        Tuple of (annotated_images, summaries)
    """
    # 1) Get annotated images using Ultralytics
    annotated = draw_results(results, save=save, project=project, name=name)
    
    # 2) Get structured data using Ultralytics
    summaries = summarize_results(results, want_ultralytics_json=True) if want_json else None
    
    return annotated, summaries


if __name__ == "__main__":
    # Demo usage of minimal postprocessing
    print("📊 Minimal postprocessing demo using Ultralytics best practices")
    
    # In real usage, results would come from YOLO inference
    dummy_results = []  # Would contain actual YOLO Results objects
    
    # Test minimal functions
    annotated_images, summaries = process_results(
        dummy_results, 
        save=False, 
        project="runs/predict", 
        name="demo", 
        want_json=True
    )
    
    batch_summary = create_batch_summary(summaries or [])
    
    print(f"✅ Processed {len(annotated_images)} images")
    print(f"📈 Batch summary: {batch_summary}")
    print("✅ Minimal postprocessing ready")
