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
from pathlib import Path
import gc


class Inference:
    """
    Universal YOLO inference class using Ultralytics
    Supports .pt, .onnx, .engine, .torchscript formats with same API
    Always uses 640x640 input size as per Ultralytics YOLO standard
    """
    
    def __init__(self, 
                 model_path: str,
                 imgsz: int = 640,  # Always use 640 for YOLO models
                 conf: float = 0.5,
                 iou: float = 0.45,
                 device: str = 'auto',
                 half: bool = False,
                 task: str = None):
        """
        Initialize universal YOLO inference with enhanced device support
        
        Args:
            model_path: Path to model (.pt, .onnx, .engine, .openvino, etc.)
            imgsz: Input image size (single value for square)
            conf: Confidence threshold
            iou: IoU threshold for NMS
            device: Device string (supports 'cpu', 'cuda:0', 'mps', 'intel:gpu', 'intel:cpu', etc.)
            half: Use FP16 (auto-managed for OpenVINO/TensorRT models)
            task: Model task ('detect', 'segment', 'classify', 'pose', 'obb'). Auto-detected if None.
        """
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.half = half
        self.task = task
        
        # Enhanced device information
        self.device_type = self._detect_device_type(device)
        self.supports_batching = self._check_batch_support()
        
        # Performance tracking
        self.inference_times = []
        self.total_frames = 0
        
        # Load model
        self.model = None
        self._load_model()
    
    def _detect_device_type(self, device: str) -> str:
        """Detect the type of device for optimized handling"""
        device_str = str(device).lower()
        
        if device_str.startswith('intel:'):
            return 'openvino'
        elif device_str.startswith('cuda') or device_str.isdigit():
            return 'cuda'
        elif device_str == 'mps':
            return 'mps'
        elif device_str == 'cpu':
            return 'cpu'
        else:
            return 'unknown'
    
    def _check_batch_support(self) -> bool:
        """Check if model format likely supports batch processing"""
        model_path_lower = self.model_path.lower()
        
        # PyTorch models generally support batching well
        if model_path_lower.endswith('.pt'):
            return True
        # ONNX may or may not support batching depending on export
        elif model_path_lower.endswith('.onnx'):
            return False  # Conservative - try individual frames first
        # TensorRT engines depend on how they were built
        elif model_path_lower.endswith('.engine'):
            return False  # Conservative
        # OpenVINO models prefer individual processing
        elif 'openvino' in model_path_lower or model_path_lower.endswith('.xml'):
            return False
        else:
            return False  # Conservative for unknown formats
    
    def _get_model_info(self) -> Dict[str, str]:
        """Get detailed model information for verification"""
        model_path_lower = self.model_path.lower()
        
        # Determine format
        if model_path_lower.endswith('.pt'):
            format_type = 'PyTorch'
        elif model_path_lower.endswith('.onnx'):
            format_type = 'ONNX'
        elif model_path_lower.endswith('.engine'):
            format_type = 'TensorRT'
        elif model_path_lower.endswith('.torchscript'):
            format_type = 'TorchScript'
        elif 'openvino' in model_path_lower or model_path_lower.endswith('.xml'):
            format_type = 'OpenVINO'
        else:
            format_type = 'Unknown'
        
        # Determine task from filename
        filename = Path(self.model_path).stem.lower()
        if 'seg' in filename:
            task_type = 'segmentation'
        elif 'pose' in filename:
            task_type = 'pose estimation'
        elif 'cls' in filename or 'classify' in filename:
            task_type = 'classification'
        else:
            task_type = 'object detection'
        
        return {
            'format': format_type,
            'task': task_type,
            'filename': Path(self.model_path).name,
            'size_mb': round(Path(self.model_path).stat().st_size / (1024*1024), 1) if Path(self.model_path).exists() else 0
        }

    def _load_model(self):
        """Load YOLO model with enhanced device handling and verification"""
        try:
            print(f"⏳ Loading model: {self.model_path}")
            
            # Universal loader - works for .pt, .onnx, .engine, .torchscript, OpenVINO
            # Pass task parameter if specified to avoid auto-detection warnings
            if self.task:
                self.model = YOLO(self.model_path, task=self.task)
                print(f"🎯 Using explicit task: {self.task}")
            else:
                self.model = YOLO(self.model_path)
                print(f"🔍 Auto-detecting task...")
            
            # Model verification and info
            model_info = self._get_model_info()
            print(f"✅ Model loaded successfully:")
            print(f"   📁 Path: {self.model_path}")
            print(f"   🏷️ Format: {model_info['format']}")
            print(f"   🎯 Task: {model_info['task']}")
            print(f"   📏 Input size: {self.imgsz}x{self.imgsz}")
            
            # Enhanced device handling for different backends
            if self.device == 'auto':
                if torch.cuda.is_available():
                    self.device = 'cuda:0'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = 'mps'
                else:
                    self.device = 'cpu'
            
            # Handle different device types appropriately
            if self.device_type == 'openvino':
                # OpenVINO devices: let Ultralytics handle device string directly
                print(f"⚡ Using OpenVINO device: {self.device}")
                # Don't call model.to() for OpenVINO - Ultralytics handles it internally
                
            elif self.device_type == 'cuda':
                if torch.cuda.is_available():
                    device_id = 0 if self.device == 'cuda' else int(self.device.split(':')[1]) if ':' in str(self.device) else int(self.device)
                    device_name = torch.cuda.get_device_name(device_id)
                    memory_gb = torch.cuda.get_device_properties(device_id).total_memory / 1e9
                    print(f"🚀 Using CUDA GPU {device_id}: {device_name} ({memory_gb:.1f}GB)")
                    
                    # Only call model.to() for PyTorch-based models
                    if not self.model_path.lower().endswith(('.onnx', '.engine')):
                        self.model.to(device_id)
                else:
                    print("⚠️ CUDA requested but not available, falling back to CPU")
                    self.device = 'cpu'
                    self.device_type = 'cpu'
                    
            elif self.device_type == 'mps':
                print("🍎 Using Apple MPS (Metal Performance Shaders)")
                # Let Ultralytics handle MPS device assignment
                
            else:  # CPU or unknown
                print(f"🔧 Using device: {self.device}")
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
    
    def predict_batch(self, frames_rgb: List[np.ndarray], warmup: bool = False, stream: bool = False, 
                     save: bool = False, project: str = "runs/predict", name: str = "exp") -> List[Any]:
        """
        Enhanced batch prediction with Ultralytics built-in saving support
        
        Args:
            frames_rgb: List of RGB frames (from universal preprocessing)
            warmup: Whether this is a warmup call
            stream: Use streaming mode for memory efficiency (good for long sequences)
            save: Use Ultralytics built-in saving (images + labels if save_txt=True)
            project: Project directory for saving
            name: Experiment name for saving
            
        Returns:
            List of YOLO Results objects or generator if stream=True
        """
        if not frames_rgb:
            return []
        
        start_time = time.time()
        
        # Smart processing strategy based on model format and device
        if self.supports_batching and len(frames_rgb) > 1:
            # Try batch processing for supported formats
            try:
                results = self.model.predict(
                    frames_rgb,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    device=self.device,
                    half=self.half,
                    verbose=False,
                    stream=stream,  # Support streaming for memory efficiency
                    show=False,
                    save=save,  # Use Ultralytics built-in saving
                    project=project,
                    name=name,
                    exist_ok=True
                )
                
                # Handle streaming vs batch results
                if stream:
                    return results  # Return generator as-is
                else:
                    results_list = results if isinstance(results, list) else [results]
                    
                    # Accurate GPU timing with synchronization
                    if self.device_type == 'cuda' and torch.cuda.is_available():
                        torch.cuda.synchronize()  # Ensure GPU work is complete
                    
                    # Track performance (skip warmup)
                    if not warmup:
                        inference_time = time.time() - start_time
                        self.inference_times.append(inference_time)
                        self.total_frames += len(frames_rgb)
                        
                        # Keep only last 100 measurements for rolling average
                        if len(self.inference_times) > 100:
                            self.inference_times.pop(0)
                    
                    return results_list
                    
            except Exception as batch_error:
                if not warmup:
                    print(f"⚠️ Batch processing failed ({batch_error}), falling back to individual frames")
        
        # Fallback: Individual frame processing
        results = []
        for i, frame in enumerate(frames_rgb):
            try:
                result = self.model.predict(
                    frame,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    device=self.device,
                    half=self.half,
                    verbose=False,
                    stream=False,  # Individual frames don't need streaming
                    show=False,
                    save=False
                )[0]  # Get first (and only) result
                results.append(result)
            except Exception as frame_error:
                print(f"⚠️ Frame {i} processing failed: {frame_error}")
                results.append(None)
        
        # GPU synchronization for accurate timing
        if self.device_type == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Track performance (skip warmup)
        if not warmup:
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.total_frames += len(frames_rgb)
            
            # Keep only last 100 measurements
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
        
        return results
    
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
                'total_batches': 0,
                'last_inference_time_ms': 0.0
            }
        
        avg_time = np.mean(self.inference_times)
        avg_fps = (self.total_frames / len(self.inference_times)) / avg_time if avg_time > 0 else 0
        last_time = self.inference_times[-1] if self.inference_times else 0.0
        
        return {
            'avg_inference_time_ms': avg_time * 1000,
            'avg_fps': avg_fps,
            'total_frames': self.total_frames,
            'total_batches': len(self.inference_times),
            'last_inference_time_ms': last_time * 1000
        }
    
    def get_last_inference_time_ms(self) -> float:
        """Get the last inference time in milliseconds (Ultralytics style)"""
        return self.inference_times[-1] * 1000 if self.inference_times else 0.0
    
    def update_thresholds(self, conf: float = None, iou: float = None):
        """Update inference thresholds"""
        if conf is not None:
            self.conf = conf
        if iou is not None:
            self.iou = iou
        
        print(f"📊 Updated thresholds: conf={self.conf}, iou={self.iou}")
    
    def verify_model_usage(self) -> Dict[str, Any]:
        """Verify which model is actually being used during inference"""
        model_info = self._get_model_info()
        
        verification = {
            'model_loaded': self.model is not None,
            'model_path': self.model_path,
            'model_info': model_info,
            'device': self.device,
            'device_type': self.device_type,
            'supports_batching': self.supports_batching,
            'input_size': f"{self.imgsz}x{self.imgsz}",
            'confidence_threshold': self.conf,
            'iou_threshold': self.iou,
            'half_precision': self.half
        }
        
        print("🔍 Model Verification:")
        print(f"   ✅ Model: {model_info['filename']} ({model_info['format']})")
        print(f"   🎯 Task: {model_info['task']}")
        print(f"   🔧 Device: {self.device} ({self.device_type})")
        print(f"   📏 Input: {self.imgsz}x{self.imgsz}")
        print(f"   ⚙️ Batch Support: {self.supports_batching}")
        
        return verification

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
    
    # Check for OpenVINO IR format (directories with .xml and .bin files)
    for model_dir in models_path.iterdir():
        if model_dir.is_dir():
            xml_files = list(model_dir.glob('*.xml'))
            bin_files = list(model_dir.glob('*.bin'))
            
            if xml_files and bin_files:
                # Found OpenVINO IR format
                name = model_dir.name.lower()
                if 'seg' in name:
                    task = 'segmentation'
                elif 'pose' in name:
                    task = 'pose'
                elif 'cls' in name or 'classify' in name:
                    task = 'classification'
                else:
                    task = 'detection'
                
                # Calculate total size of IR files
                total_size = sum(f.stat().st_size for f in xml_files + bin_files)
                
                models.append({
                    'name': model_dir.name,
                    'path': str(model_dir),
                    'format': 'openvino',
                    'task': task,
                    'size': total_size
                })
    
    return sorted(models, key=lambda x: x['name'])


def get_device_config(prefer_gpu: bool = True) -> Dict[str, Any]:
    """
    Get optimal device configuration with support for multiple platforms
    Supports CUDA, MPS (Apple Silicon), OpenVINO, and CPU
    
    Args:
        prefer_gpu: Whether to prefer GPU over CPU
        
    Returns:
        Device configuration dictionary with optimized settings
    """
    # Check for CUDA (NVIDIA GPU)
    if prefer_gpu and torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_id = 0  # Use first GPU by default
        device_name = torch.cuda.get_device_name(device_id)
        memory_gb = torch.cuda.get_device_properties(device_id).total_memory / 1e9
        print(f"🚀 Using CUDA GPU {device_id}: {device_name} ({memory_gb:.1f}GB)")
        
        return {
            'device': f'cuda:{device_id}',
            'half': True,  # FP16 for GPU speedup
            'imgsz': 640,
            'batch_optimal': True,
            'device_type': 'cuda',
            'device_name': device_name
        }
    
    # Check for MPS (Apple Silicon GPU)
    elif prefer_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("🍎 Using Apple MPS (Metal Performance Shaders)")
        return {
            'device': 'mps',
            'half': False,  # MPS doesn't support FP16 in all cases
            'imgsz': 640,
            'batch_optimal': True,
            'device_type': 'mps',
            'device_name': 'Apple GPU'
        }
    
    # Check for OpenVINO Intel GPU (if available)
    elif prefer_gpu:
        try:
            # Check if OpenVINO is available
            import openvino
            print("⚡ OpenVINO available - can use intel:gpu for Intel GPU acceleration")
            return {
                'device': 'intel:gpu',
                'half': False,  # OpenVINO manages precision internally
                'imgsz': 640,
                'batch_optimal': False,  # OpenVINO may prefer individual frames
                'device_type': 'openvino_gpu',
                'device_name': 'Intel GPU (OpenVINO)'
            }
        except ImportError:
            pass
    
    # Fallback to CPU (with optional OpenVINO acceleration)
    try:
        import openvino
        print("🔧 Using CPU with OpenVINO acceleration (up to 3x faster)")
        return {
            'device': 'intel:cpu',
            'half': False,  # OpenVINO manages precision
            'imgsz': 640,
            'batch_optimal': False,  # OpenVINO individual frame processing
            'device_type': 'openvino_cpu',
            'device_name': 'Intel CPU (OpenVINO)'
        }
    except ImportError:
        print("🔧 Using standard CPU")
        return {
            'device': 'cpu',
            'half': False,  # Keep FP32 for CPU
            'imgsz': 640,
            'batch_optimal': False,  # CPU prefers individual frames
            'device_type': 'cpu',
            'device_name': 'Standard CPU'
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
