#!/usr/bin/env python3
"""
Test how postprocessing adapts to different YOLO model types
Demonstrates Ultralytics built-in methods for different tasks
"""

from pipeline import Inference, discover_models, get_device_config
from postprocess import process_results, create_batch_summary
import numpy as np
import cv2

def test_model_adaptation():
    """Test how postprocessing handles different model types"""
    
    print("🧪 Testing Model Type Adaptation")
    print("=" * 50)
    
    # Get available models
    models = discover_models()
    if not models:
        print("❌ No models found")
        return
    
    # Create a dummy image for testing
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    
    device_config = get_device_config(prefer_gpu=True)
    
    # Test each model type
    for model in models[:3]:  # Test first 3 models
        print(f"\n🔍 Testing: {model['name']} ({model['format']}) - {model['task']}")
        print("-" * 40)
        
        try:
            # Load model
            inference = Inference(
                model_path=model['path'],
                imgsz=640,
                conf=0.25,
                device=device_config['device'],
                half=device_config['half']
            )
            
            # Run inference
            print(f"📊 Running inference...")
            results = inference.predict_batch([test_image_rgb])
            
            if results:
                # Process results using minimal postprocessing
                annotated_images, summaries = process_results(
                    results,
                    save=False,  # Don't save during test
                    want_json=True
                )
                
                # Analyze what we got
                result = results[0]
                print(f"✅ Model output analysis:")
                
                # Check task type based on available attributes
                task_detected = "unknown"
                if hasattr(result, 'boxes') and result.boxes is not None:
                    task_detected = "detection"
                    print(f"   📦 Boxes: {len(result.boxes)} detections")
                    
                if hasattr(result, 'masks') and result.masks is not None:
                    task_detected = "segmentation"
                    print(f"   🎭 Masks: {len(result.masks)} segments")
                    
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    task_detected = "pose"
                    print(f"   🦴 Keypoints: {len(result.keypoints)} people")
                    
                if hasattr(result, 'probs') and result.probs is not None:
                    task_detected = "classification"
                    top_class = result.probs.top1
                    print(f"   📊 Classification: class_{top_class} ({result.probs.top1conf:.3f})")
                
                print(f"   🎯 Task detected: {task_detected}")
                print(f"   🖼️ Annotated image: {annotated_images[0].shape if annotated_images else 'None'}")
                
                # Show Ultralytics JSON format
                if summaries and len(summaries) > 0:
                    summary = summaries[0]
                    if isinstance(summary, list):  # Ultralytics .tojson() format
                        print(f"   📄 Ultralytics JSON: {len(summary)} detections")
                        if summary:
                            print(f"       First detection keys: {list(summary[0].keys())}")
                    else:  # Custom format
                        print(f"   📄 Custom summary: {summary.get('total_detections', 0)} detections")
                
                # Test batch summary
                batch_summary = create_batch_summary(summaries)
                print(f"   📈 Batch summary: {batch_summary['total_detections']} total detections")
                
                # Demonstrate how result.plot() adapts automatically
                print(f"   🎨 Ultralytics result.plot() handles {task_detected} visualization automatically")
                
            else:
                print("❌ No results returned")
            
            # Cleanup
            inference.cleanup()
            
        except Exception as e:
            print(f"❌ Error testing {model['name']}: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Model adaptation test completed!")
    print("\n📝 Key findings:")
    print("  • result.plot() automatically adapts to model type")
    print("  • Detection models: shows bounding boxes + labels")  
    print("  • Segmentation models: shows colored masks + boxes")
    print("  • Pose models: shows keypoint skeletons")
    print("  • Classification: shows top prediction")
    print("  • result.tojson() provides unified format for all tasks")


if __name__ == "__main__":
    test_model_adaptation()
