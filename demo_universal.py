"""
Universal Multi-Angle YOLO Pipeline Demo
Demonstrates the key improvements using Ultralytics best practices
"""
import time
from preprocess import open_cam, make_views_rgb, detect_webcams
from pipeline import Inference, get_device_config, discover_models  
from postprocess import draw_results, summarize_results, create_batch_summary


def main():
    print("🎯 Universal Multi-Angle YOLO Pipeline Demo")
    print("=" * 50)
    
    # 1. Universal Model Discovery
    print("\n📁 Discovering models...")
    models = discover_models()
    if not models:
        print("❌ No models found! Place .pt or .onnx files in 'models' directory")
        return
    
    print(f"Found {len(models)} models:")
    for i, model in enumerate(models):
        print(f"  {i+1}. {model['name']} ({model['format']}) - {model['task']}")
    
    # Use first model (prioritize .pt over .onnx)
    pt_models = [m for m in models if m['format'] == 'pt']
    selected_model = pt_models[0] if pt_models else models[0]
    
    print(f"\n🚀 Using: {selected_model['name']} ({selected_model['format']})")
    
    # 2. Universal Device Configuration  
    print("\n🔧 Configuring device...")
    device_config = get_device_config(prefer_gpu=True)
    
    # 3. Universal Inference Setup
    print(f"\n⚡ Initializing inference pipeline...")
    inference = Inference(
        model_path=selected_model['path'],
        imgsz=640,  # Standard size
        conf=0.5,
        device=device_config['device'],
        half=device_config['half']
    )
    
    # 4. Universal Preprocessing
    print("\n📹 Testing webcam capture...")
    webcams = detect_webcams()
    
    if not webcams:
        print("❌ No webcams detected")
        return
        
    cap = open_cam(webcams[0])
    if cap is None:
        print("❌ Failed to open webcam")
        return
    
    print("✅ Webcam opened successfully")
    
    # 5. Main Processing Loop
    print("\n🔄 Running inference loop (5 iterations)...")
    
    for i in range(5):
        start_time = time.time()
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Failed to capture frame {i+1}")
            continue
        
        # Universal preprocessing: BGR → 4 RGB views at 640x640
        views = make_views_rgb(frame, target=(640, 640))
        
        # Universal inference: single batch call with ONNX fallback
        results = inference.predict_batch(views)
        
        if results:
            # Universal postprocessing using Ultralytics methods
            annotated_images = draw_results(results)
            summaries = summarize_results(results)
            batch_summary = create_batch_summary(summaries)
            
            # Performance metrics
            processing_time = time.time() - start_time
            total_detections = batch_summary['total_detections']
            
            print(f"Frame {i+1:2d}: {total_detections:2d} detections | "
                  f"{processing_time*1000:5.1f}ms | "
                  f"Classes: {batch_summary['detected_classes']}")
        else:
            print(f"Frame {i+1:2d}: Processing failed")
        
        time.sleep(0.1)  # Brief pause
    
    # 6. Performance Summary
    print("\n📊 Performance Summary:")
    stats = inference.get_performance_stats()
    print(f"  Average FPS: {stats['avg_fps']:.1f}")
    print(f"  Average inference time: {stats['avg_inference_time_ms']:.1f}ms")
    print(f"  Total frames processed: {stats['total_frames']}")
    
    # Cleanup
    cap.release()
    inference.cleanup()
    print("\n✅ Demo completed successfully!")
    
    print("\n🎯 Key Universal Features Demonstrated:")
    print("  ✓ Model format agnostic (.pt, .onnx)")
    print("  ✓ Device optimization (CPU/GPU)")
    print("  ✓ Batch processing with ONNX fallback")
    print("  ✓ Universal preprocessing (BGR → 4 RGB views)")
    print("  ✓ Ultralytics native postprocessing")
    print("  ✓ Performance tracking")


if __name__ == "__main__":
    main()
