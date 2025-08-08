"""
Universal YOLO inference pipeline
Works with PyTorch (.pt), ONNX (.onnx), TensorRT (.engine), OpenVINO
Optimized for both CPU and GPU using Ultralytics best practices
"""
import torch
from ultralytics import YOLO
from typing import List, Dict, Any, Optional, Union
import numpy as np
import time
from pathlib import Path
import gc


class Inference:
    """
    Universal YOLO inference class using Ultralytics
    Supports .pt, .onnx, .engine, .torchscript formats with same API
    """
    
    def __init__(self, 
                 model_path: str,
                 imgsz: int = 480,
                 conf: float = 0.5,
                 iou: float = 0.45,
                 device: str = 'auto',
                 half: bool = False):
        """
        Initialize universal YOLO inference
        
        Args:
            model_path: Path to model (.pt, .onnx, .engine, etc.)
            imgsz: Input image size (single value for square)
            conf: Confidence threshold
            iou: IoU threshold for NMS
            device: Device ('auto', 'cpu', 0 for cuda:0, etc.)
            half: Use FP16 (ignored for most ONNX unless model is FP16)
        """
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.half = half
        
        # Performance tracking
        self.inference_times = []
        self.total_frames = 0
        
        # Load model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model with optimizations"""
        try:
            print(f"⏳ Loading model: {self.model_path}")
            
            # Universal loader - works for .pt, .onnx, .engine, .torchscript
            self.model = YOLO(self.model_path)
            
            # Device optimization
            if self.device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            if self.device != 'cpu' and torch.cuda.is_available():
                print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
                self.model.to(self.device)
            else:
                print("🔧 Using CPU")
                self.device = 'cpu'
            
            # Warmup
            self._warmup()
            
            print(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _warmup(self, num_warmup: int = 3):
        """Warmup model with dummy inputs - ONNX compatible"""
        print("🔥 Warming up model...")
        
        # Create single dummy frame for warmup (works better with ONNX)
        dummy_frame = np.random.randint(0, 255, (self.imgsz, self.imgsz, 3), dtype=np.uint8)
        
        for i in range(num_warmup):
            try:
                # Try single frame first (ONNX compatible)
                _ = self.model.predict(
                    dummy_frame,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    device=self.device,
                    half=self.half,
                    verbose=False,
                    show=False,
                    save=False
                )
                print(f"   Warmup {i+1}/{num_warmup} completed")
            except Exception as e:
                print(f"   Warmup {i+1}/{num_warmup} failed: {e}")
        
        # Clear warmup stats
        self.inference_times.clear()
        self.total_frames = 0
        
        print("✅ Model warmup completed")
    
    def predict_batch(self, frames_rgb: List[np.ndarray], warmup: bool = False) -> List[Any]:
        """
        Universal batch prediction with ONNX fallback
        
        Args:
            frames_rgb: List of RGB frames (from universal preprocessing)
            warmup: Whether this is a warmup call
            
        Returns:
            List of YOLO Results objects
        """
        if not frames_rgb:
            return []
        
        start_time = time.time()
        
        # First try batch processing
        try:
            # Single batched call using Ultralytics predict method
            results = self.model.predict(
                frames_rgb,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                half=self.half,
                verbose=False,
                stream=False,  # Get all results at once for batch processing
                show=False,
                save=False
            )
            
            # Track performance (skip warmup)
            if not warmup:
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)
                self.total_frames += len(frames_rgb)
                
                # Keep only last 100 measurements
                if len(self.inference_times) > 100:
                    self.inference_times.pop(0)
            
            return results if isinstance(results, list) else [results]
            
        except Exception as e:
            if not warmup:  # Only print errors during actual inference
                print(f"⚠️ Batch inference failed, falling back to individual calls")
            
            # Fallback: Process frames individually (works better with ONNX)
            try:
                results = []
                for frame in frames_rgb:
                    result = self.model.predict(
                        frame,
                        imgsz=self.imgsz,
                        conf=self.conf,
                        iou=self.iou,
                        device=self.device,
                        half=self.half,
                        verbose=False,
                        show=False,
                        save=False
                    )
                    results.append(result[0] if isinstance(result, list) else result)
                
                # Track performance (skip warmup)
                if not warmup:
                    inference_time = time.time() - start_time
                    self.inference_times.append(inference_time)
                    self.total_frames += len(frames_rgb)
                    
                    # Keep only last 100 measurements
                    if len(self.inference_times) > 100:
                        self.inference_times.pop(0)
                
                return results
                
            except Exception as e2:
                if not warmup:
                    print(f"❌ Individual inference also failed: {e2}")
                return []
    
    def predict_single(self, frame_rgb: np.ndarray) -> Any:
        """Single frame prediction wrapper"""
        results = self.predict_batch([frame_rgb])
        return results[0] if results else None
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.inference_times:
            return {
                'avg_inference_time_ms': 0.0,
                'avg_fps': 0.0,
                'total_frames': self.total_frames,
                'total_batches': 0
            }
        
        avg_time = np.mean(self.inference_times)
        avg_fps = (self.total_frames / len(self.inference_times)) / avg_time if avg_time > 0 else 0
        
        return {
            'avg_inference_time_ms': avg_time * 1000,
            'avg_fps': avg_fps,
            'total_frames': self.total_frames,
            'total_batches': len(self.inference_times)
        }
    
    def update_thresholds(self, conf: float = None, iou: float = None):
        """Update inference thresholds"""
        if conf is not None:
            self.conf = conf
        if iou is not None:
            self.iou = iou
        
        print(f"📊 Updated thresholds: conf={self.conf}, iou={self.iou}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.model is not None:
            del self.model
            self.model = None
        
        # Clear GPU memory if using CUDA
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
            gc.collect()
        
        print("🧹 Pipeline cleaned up")


def discover_models(models_dir: str = "models") -> List[Dict[str, str]]:
    """
    Discover available YOLO models in the models directory
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        List of model info dictionaries
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    
    models = []
    supported_extensions = ['.pt', '.onnx', '.engine', '.torchscript']
    
    for model_file in models_path.iterdir():
        if model_file.suffix.lower() in supported_extensions:
            # Determine task based on filename
            name = model_file.stem.lower()
            if 'seg' in name:
                task = 'segmentation'
            elif 'pose' in name:
                task = 'pose'
            elif 'cls' in name or 'classify' in name:
                task = 'classification'
            else:
                task = 'detection'  # Default
            
            models.append({
                'name': model_file.stem,
                'path': str(model_file),
                'format': model_file.suffix[1:],  # Remove dot
                'task': task,
                'size': model_file.stat().st_size
            })
    
    return sorted(models, key=lambda x: x['name'])


def get_device_config(prefer_gpu: bool = True) -> Dict[str, Any]:
    """
    Get optimal device configuration
    
    Args:
        prefer_gpu: Whether to prefer GPU over CPU
        
    Returns:
        Device configuration dictionary
    """
    if prefer_gpu and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🚀 Using GPU: {device_name} ({memory_gb:.1f}GB)")
        
        return {
            'device': 0,  # CUDA device 0
            'half': True,  # Use FP16 for GPU
            'imgsz': 640,  # Standard size for GPU
            'batch_optimal': True
        }
    else:
        print("🔧 Using CPU")
        return {
            'device': 'cpu',
            'half': False,  # Keep FP32 for CPU
            'imgsz': 640,  # Standard size works for CPU too
            'batch_optimal': False
        }


if __name__ == "__main__":
    # Demo usage with universal approach
    from preprocess import open_cam, make_views_rgb
    
    models = discover_models()
    print(f"📁 Found {len(models)} models:")
    
    for model in models:
        print(f"  {model['name']} ({model['format']}) - {model['task']}")
    
    if models:
        # Get device config
        device_config = get_device_config(prefer_gpu=True)
        
        # Test with first model using universal approach
        model_path = models[0]['path']
        
        # Initialize inference with optimal settings
        inference = Inference(
            model_path=model_path,
            imgsz=device_config['imgsz'],
            conf=0.5,
            device=device_config['device'],
            half=device_config['half']
        )
        
        # Test with webcam capture
        cap = open_cam(0)
        if cap is not None:
            print("📹 Testing universal pipeline...")
            
            for i in range(3):  # Test 3 frames
                ret, frame = cap.read()
                if ret:
                    # Universal preprocessing: BGR → 4 RGB views
                    views = make_views_rgb(frame, target=(640, 640))
                    
                    # Single batched inference call
                    results = inference.predict_batch(views)
                    
                    print(f"Frame {i+1}: Processed {len(results)} views")
                else:
                    break
            
            cap.release()
            
            # Show performance stats
            stats = inference.get_performance_stats()
            print(f"📊 Performance: {stats['avg_fps']:.1f} FPS")
            
            inference.cleanup()
            print("✅ Demo completed")
        else:
            print("❌ No webcam available for demo")
