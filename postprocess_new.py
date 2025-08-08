"""
Universal postprocessing module for YOLO inference results
Uses Ultralytics built-in methods to minimize custom logic
Handles visualization and structured output extraction
"""
import cv2
import numpy as np
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


def draw_results(results: List[Any]) -> List[np.ndarray]:
    """
    Universal result drawing using Ultralytics built-in plot method
    
    Args:
        results: List of YOLO Results objects
        
    Returns:
        List of annotated images (BGR format for display)
    """
    annotated_images = []
    
    for result in results:
        try:
            if result is None:
                continue
            
            # Use Ultralytics built-in plot method
            annotated = result.plot(
                conf=True,      # Show confidence
                labels=True,    # Show class labels
                boxes=True,     # Show bounding boxes
                line_width=2,   # Line thickness
                font_size=1.0,  # Font size
                pil=False       # Return numpy array
            )
            
            # Ultralytics plot returns RGB, convert to BGR for OpenCV display
            if annotated is not None:
                annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                annotated_images.append(annotated_bgr)
            
        except Exception as e:
            print(f"⚠️ Failed to plot result: {e}")
            # Return original image if available
            if hasattr(result, 'orig_img') and result.orig_img is not None:
                annotated_images.append(result.orig_img)
    
    return annotated_images


def summarize_results(results: List[Any]) -> List[Dict[str, Any]]:
    """
    Extract structured data from YOLO results using Ultralytics attributes
    
    Args:
        results: List of YOLO Results objects
        
    Returns:
        List of summary dictionaries
    """
    summaries = []
    
    for i, result in enumerate(results):
        summary = {
            'source_index': i,
            'detections': [],
            'total_detections': 0,
            'classes_detected': [],
            'avg_confidence': 0.0
        }
        
        try:
            if result is None:
                summaries.append(summary)
                continue
            
            # Extract detections using Ultralytics structure
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                # Get all detection data at once
                if len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()      # Bounding boxes
                    conf = boxes.conf.cpu().numpy()      # Confidence scores
                    cls = boxes.cls.cpu().numpy().astype(int)  # Class indices
                    
                    # Get class names
                    names = result.names if hasattr(result, 'names') else {}
                    
                    # Build detection list
                    detections = []
                    confidences = []
                    classes = set()
                    
                    for j in range(len(cls)):
                        class_name = names.get(cls[j], f"class_{cls[j]}")
                        detection = {
                            'bbox': xyxy[j].tolist(),  # [x1, y1, x2, y2]
                            'confidence': float(conf[j]),
                            'class_id': int(cls[j]),
                            'class_name': class_name
                        }
                        detections.append(detection)
                        confidences.append(float(conf[j]))
                        classes.add(class_name)
                    
                    # Update summary
                    summary['detections'] = detections
                    summary['total_detections'] = len(detections)
                    summary['classes_detected'] = list(classes)
                    summary['avg_confidence'] = np.mean(confidences) if confidences else 0.0
            
            # Handle segmentation if available
            if hasattr(result, 'masks') and result.masks is not None:
                summary['has_masks'] = True
                summary['mask_count'] = len(result.masks)
            else:
                summary['has_masks'] = False
                summary['mask_count'] = 0
                
        except Exception as e:
            print(f"⚠️ Failed to summarize result {i}: {e}")
        
        summaries.append(summary)
    
    return summaries


def create_batch_summary(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create overall summary for batch of results
    
    Args:
        summaries: List of individual result summaries
        
    Returns:
        Batch summary dictionary
    """
    total_detections = sum(s['total_detections'] for s in summaries)
    all_classes = set()
    all_confidences = []
    
    for summary in summaries:
        all_classes.update(summary['classes_detected'])
        for det in summary['detections']:
            all_confidences.append(det['confidence'])
    
    batch_summary = {
        'total_sources': len(summaries),
        'total_detections': total_detections,
        'unique_classes': len(all_classes),
        'detected_classes': list(all_classes),
        'per_source_detections': [s['total_detections'] for s in summaries],
        'overall_avg_confidence': np.mean(all_confidences) if all_confidences else 0.0
    }
    
    if all_confidences:
        batch_summary['confidence_range'] = {
            'min': float(np.min(all_confidences)),
            'max': float(np.max(all_confidences))
        }
    
    return batch_summary


class ResultSaver:
    """Optional result saver using Ultralytics patterns"""
    
    def __init__(self, save_dir: str = "results", save_images: bool = True, save_json: bool = True):
        self.save_dir = Path(save_dir)
        self.save_images = save_images
        self.save_json = save_json
        
        if save_images or save_json:
            self.save_dir.mkdir(exist_ok=True)
            print(f"💾 Results will be saved to: {self.save_dir}")
    
    def save_batch(self, 
                  annotated_images: List[np.ndarray], 
                  summaries: List[Dict[str, Any]], 
                  source_names: List[str] = None) -> None:
        """Save batch of annotated images and summaries"""
        
        if not (self.save_images or self.save_json):
            return
        
        timestamp = int(time.time() * 1000)
        
        for i, (image, summary) in enumerate(zip(annotated_images, summaries)):
            source_name = source_names[i] if source_names else f"source_{i}"
            
            if self.save_images and image is not None:
                # Save annotated image
                image_filename = f"{source_name}_{timestamp}.jpg"
                image_path = self.save_dir / image_filename
                cv2.imwrite(str(image_path), image)
            
            if self.save_json:
                # Save detection data
                json_filename = f"{source_name}_{timestamp}.json"
                json_path = self.save_dir / json_filename
                
                with open(json_path, 'w') as f:
                    json.dump(summary, f, indent=2)


# Legacy class for compatibility
class ResultProcessor:
    """Simplified result processor using universal methods"""
    
    def __init__(self, save_results: bool = False, save_dir: str = "results"):
        self.save_results = save_results
        self.saver = ResultSaver(save_dir) if save_results else None
        self.processed_frames = 0
        self.processing_times = []
    
    def process_batch_results(self, 
                            results: List[Any],
                            source_names: List[str] = None) -> List[Dict[str, Any]]:
        """Process batch using universal methods"""
        start_time = time.time()
        
        # Use universal functions
        annotated_images = draw_results(results)
        summaries = summarize_results(results)
        
        # Save if requested
        if self.save_results and self.saver:
            self.saver.save_batch(annotated_images, summaries, source_names)
        
        # Track performance
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.processed_frames += len(results)
        
        # Combine results
        processed_results = []
        for i, (image, summary) in enumerate(zip(annotated_images, summaries)):
            processed_results.append({
                'source_name': source_names[i] if source_names else f"source_{i}",
                'annotated_image': image,
                'summary': summary,
                'raw_result': results[i] if i < len(results) else None
            })
        
        return processed_results
    
    def get_processing_stats(self) -> Dict[str, float]:
        """Get processing statistics"""
        if not self.processing_times:
            return {
                'avg_processing_time_ms': 0.0,
                'total_frames_processed': self.processed_frames,
                'processing_fps': 0.0
            }
        
        avg_time = np.mean(self.processing_times)
        processing_fps = 1.0 / avg_time if avg_time > 0 else 0.0
        
        return {
            'avg_processing_time_ms': avg_time * 1000,
            'total_frames_processed': self.processed_frames,
            'processing_fps': processing_fps
        }


if __name__ == "__main__":
    # Demo usage of universal postprocessing
    print("📊 Universal postprocessing demo")
    
    # In real usage, results would come from YOLO inference
    dummy_results = []  # Would contain actual YOLO Results objects
    
    # Test universal functions
    annotated_images = draw_results(dummy_results)
    summaries = summarize_results(dummy_results)
    batch_summary = create_batch_summary(summaries)
    
    print(f"✅ Processed {len(annotated_images)} images")
    print(f"📈 Batch summary: {batch_summary}")
    
    # Test result saver
    saver = ResultSaver("demo_results", save_images=False, save_json=True)
    print("✅ Postprocessing ready")
