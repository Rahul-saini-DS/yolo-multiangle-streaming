"""
Universal preprocessing module for YOLO inference pipeline
Model-agnostic frame preparation for YOLOv8/YOLOv11, PyTorch/ONNX/TensorRT/OpenVINO
Optimized for Windows with DirectShow backend
"""
import cv2
import numpy as np
from typing import List, Union, Tuple, Optional
import time


def open_cam(idx: int = 0) -> cv2.VideoCapture:
    """
    Universal camera opener optimized for Windows
    Works consistently across all backends
    """
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # DirectShow backend for Windows stability
    if not cap.isOpened():
        print(f"❌ Failed to open camera {idx}")
        return None
    
    # Optimized settings for real-time processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
    
    print(f"✅ Camera {idx} opened successfully (640x480@30fps)")
    return cap


def make_views_rgb(frame_bgr: np.ndarray, target: Tuple[int, int] = (640, 640)) -> List[np.ndarray]:
    """
    Universal view generator: BGR frame → 4 RGB views at 640x640 for YOLO
    Creates 0°, 90°, 180°, 270° rotated views for multi-angle processing
    Always resizes to 640x640 as per Ultralytics YOLO standard
    
    Args:
        frame_bgr: Input BGR frame from camera (any size)
        target: Target size (width, height) - always (640, 640) for YOLO models
    
    Returns:
        List of 4 RGB frames at 640x640 ready for YOLO inference
    """
    # Convert BGR to RGB as required by Ultralytics
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize first, then rotate for efficiency
    resized = cv2.resize(frame_rgb, target)
    
    views = [
        resized,                                                    # 0° (original)
        cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE),              # 90°
        cv2.rotate(resized, cv2.ROTATE_180),                       # 180°
        cv2.rotate(resized, cv2.ROTATE_90_COUNTERCLOCKWISE)        # 270°
    ]
    
    return views


def detect_webcams() -> List[int]:
    """Detect available webcam indices"""
    available_cams = []
    
    # Check first 5 indices (usually sufficient)
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available_cams.append(i)
            cap.release()
    
    return available_cams


class MultiAngleWebcamCapture:
    """
    Optimized multi-angle webcam capture using universal preprocessing
    Produces synchronized 4-view output at 640x640 ready for YOLO inference
    Always uses 640x640 as per Ultralytics YOLO standard
    """
    
    def __init__(self, webcam_index: int = 0, target_size: Tuple[int, int] = (640, 640)):
        """
        Initialize multi-angle webcam capture
        
        Args:
            webcam_index: Index of webcam to use
            target_size: Target size for all views - always (640, 640) for YOLO models
        """
        self.webcam_index = webcam_index
        self.target_size = (640, 640)  # Always use 640x640 for YOLO models
        self.cap = None
        self.angles = [0, 90, 180, 270]
        self.angle_names = ["Front (0°)", "Right (90°)", "Rear (180°)", "Left (270°)"]
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        
    def open(self) -> bool:
        """Open webcam with optimized settings"""
        self.cap = open_cam(self.webcam_index)
        return self.cap is not None
    
    def capture_multi_angle_frames(self) -> Tuple[bool, List[np.ndarray], List[str]]:
        """
        Capture single frame and create 4 synchronized angle views
        Uses universal preprocessing for consistent output
        
        Returns:
            success: Whether capture was successful
            frames: List of 4 RGB frames ready for YOLO inference
            names: List of angle names
        """
        if self.cap is None:
            return False, [], []
        
        # Capture single frame
        ret, frame = self.cap.read()
        if not ret:
            return False, [], []
        
        # Create all 4 rotated versions using universal preprocessing
        views = make_views_rgb(frame, self.target_size)
        
        # Update performance tracking
        self.frame_count += 1
        current_time = time.time()
        if self.frame_count % 30 == 0:  # Update FPS every 30 frames
            elapsed = current_time - self.last_fps_time
            self.fps = 30.0 / elapsed if elapsed > 0 else 0.0
            self.last_fps_time = current_time
        
        return True, views, self.angle_names
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self.fps
    
    def release(self):
        """Release webcam resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __del__(self):
        self.release()


if __name__ == "__main__":
    # Demo usage of universal preprocessing
    print("🔍 Detecting available webcams...")
    webcams = detect_webcams()
    print(f"Available webcams: {webcams}")
    
    if webcams:
        # Test multi-angle capture
        capture = MultiAngleWebcamCapture(webcams[0], target_size=(640, 640))
        
        if capture.open():
            print("📹 Testing multi-angle capture...")
            
            for i in range(5):  # Test 5 frames
                success, views, names = capture.capture_multi_angle_frames()
                if success:
                    print(f"Frame {i+1}: Captured {len(views)} views at {views[0].shape} each")
                    print(f"FPS: {capture.get_fps():.1f}")
                else:
                    print(f"Frame {i+1}: Failed to capture")
                
                time.sleep(0.1)  # Brief pause
            
            capture.release()
            print("✅ Test completed")
        else:
            print("❌ Failed to open webcam")
    else:
        print("❌ No webcams detected")
