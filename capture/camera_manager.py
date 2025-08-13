"""CameraManager scaffolding for migration to multi-real-feed architecture.

This is an optional starter module based on the migration plan.
Currently provides a minimal interface with stub methods so the app can begin
integrating real multi-feed capture without breaking existing functionality.
"""
from __future__ import annotations
import cv2
import threading
import time
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Callable

@dataclass
class FeedConfig:
    id: str
    source: str  # webcam index as string (e.g. "0") or RTSP/FILE URL
    type: str  # 'webcam' | 'rtsp' | 'file'
    resolution: Tuple[int, int] = (640, 480)
    fps_cap: int = 15
    task: Optional[dict] = None  # {'type': 'detect', 'model': 'yolov8n.pt'}

@dataclass
class FeedState:
    config: FeedConfig
    status: str = "stopped"  # stopped|connecting|live|degraded|disconnected
    last_frame = None
    last_frame_time: float = 0.0
    error: Optional[str] = None
    thread: Optional[threading.Thread] = None
    stop_flag: bool = False
    frame_count: int = 0
    consecutive_failures: int = 0

class CameraManager:
    """Manages multiple independent camera feeds (webcam / rtsp / file).

    Adds lightweight resilience & health checking to help diagnose why feeds
    might not transition from 'connecting' to 'live'.
    """
    def __init__(self):
        self._feeds: Dict[str, FeedState] = {}
        self._lock = threading.Lock()

    # ---- Public API -----------------------------------------------------
    def add_feed(self, config: FeedConfig):
        with self._lock:
            if config.id in self._feeds:
                raise ValueError(f"Feed id already exists: {config.id}")
            self._feeds[config.id] = FeedState(config=config)

    def list_feeds(self) -> List[FeedState]:
        with self._lock:
            return list(self._feeds.values())

    def start(self, feed_id: str):
        state = self._feeds.get(feed_id)
        if not state:
            raise KeyError(feed_id)
        if state.status == "live":
            return
        state.stop_flag = False
        t = threading.Thread(target=self._run_feed, args=(state,), daemon=True)
        state.thread = t
        state.status = "connecting"
        t.start()

    def stop(self, feed_id: str):
        state = self._feeds.get(feed_id)
        if not state:
            return
        state.stop_flag = True
        if state.thread and state.thread.is_alive():
            state.thread.join(timeout=2)
        state.status = "stopped"

    def remove(self, feed_id: str):
        self.stop(feed_id)
        with self._lock:
            self._feeds.pop(feed_id, None)

    def get_frame(self, feed_id: str):
        state = self._feeds.get(feed_id)
        if not state:
            return None
        return state.last_frame

    # ---- Internal -------------------------------------------------------
    def _open_capture(self, state: FeedState):
        src = state.config.source
        feed_type = state.config.type

        def _raise(msg: str):
            raise ValueError(msg)

        # Webcam handling -------------------------------------------------
        if feed_type == 'webcam':
            try:
                idx = int(src)
            except ValueError:
                _raise(f"Webcam source is not an integer index: {src}")

            backends = []
            if os.name == 'nt':
                backends.append(cv2.CAP_DSHOW)
            # Always include default backend
            backends.append(None)

            cap = None
            for be in backends:
                if be is None:
                    cap = cv2.VideoCapture(idx)
                else:
                    cap = cv2.VideoCapture(idx, be)
                if cap.isOpened():
                    break
                time.sleep(0.25)

            if not cap or not cap.isOpened():
                _raise(f"Unable to open webcam index {idx}. Is it busy or missing?")

            # Configure
            w, h = state.config.resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, state.config.fps_cap)

            # Probe frames (few attempts)
            probe_ok = False
            for _ in range(5):
                ok, frm = cap.read()
                if ok and frm is not None and frm.size:
                    probe_ok = True
                    break
                time.sleep(0.1)
            if not probe_ok:
                _raise("Opened webcam but couldn't read frames (in use or driver issue)")
            return cap

        # RTSP handling ---------------------------------------------------
        if feed_type == 'rtsp':
            # Multiple retries because network streams can take time
            retries = 5
            cap = None
            for i in range(retries):
                cap = cv2.VideoCapture(src)
                if cap.isOpened():
                    # Wait a moment then test a frame
                    time.sleep(0.4)
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.size:
                        break
                    else:
                        cap.release()
                time.sleep(0.6)
            if not cap or not cap.isOpened():
                _raise(f"Failed to open RTSP after {retries} attempts: {src}")
            return cap

        # File handling ---------------------------------------------------
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            _raise(f"Failed to open file source: {src}")
        return cap

    def _run_feed(self, state: FeedState):
        try:
            # Open the camera/stream
            cap = self._open_capture(state)
            state.status = 'live'
            frame_interval = 1.0 / max(1, state.config.fps_cap)
            w, h = state.config.resolution
            
            # Variables for tracking consecutive failures
            consecutive_failures = 0
            max_consecutive_failures = 5  # Allow 5 consecutive failures before giving up
            
            print(f"Starting feed loop for {state.config.id} - {state.config.source}")
            
            while not state.stop_flag:
                t0 = time.time()
                
                try:
                    ok, frame = cap.read()
                    
                    if not ok or frame is None or frame.size == 0:
                        # Graceful end-of-file handling for file sources after at least one frame
                        if state.config.type == 'file' and state.frame_count > 0:
                            print(f"End of file reached for {state.config.id}; marking as ended")
                            state.status = 'ended'
                            state.error = None
                            break

                        consecutive_failures += 1
                        print(f"Frame read failed ({consecutive_failures}/{max_consecutive_failures}) for {state.config.id}")

                        if consecutive_failures >= max_consecutive_failures:
                            print(f"Too many consecutive failures for {state.config.id}, stopping feed")
                            state.status = 'disconnected'
                            state.error = 'read_failed_multiple_times'
                            break
                            
                        # Short delay before retrying
                        time.sleep(0.1)
                        continue
                    
                    # Reset counter on successful frame
                    consecutive_failures = 0
                    
                    # Resize if needed
                    if (w, h) != (frame.shape[1], frame.shape[0]):
                        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                    
                    # Store the frame
                    state.last_frame = frame
                    state.last_frame_time = time.time()
                    
                    # Update frame counter
                    state.frame_count = state.frame_count + 1 if hasattr(state, 'frame_count') else 1
                    
                    # FPS cap
                    elapsed = time.time() - t0
                    sleep_for = frame_interval - elapsed
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                        
                except Exception as frame_error:
                    print(f"Error processing frame for {state.config.id}: {frame_error}")
                    consecutive_failures += 1
                    time.sleep(0.1)
                    
                    if consecutive_failures >= max_consecutive_failures:
                        state.status = 'disconnected'
                        state.error = f'frame_processing_error: {str(frame_error)}'
                        break
            
            print(f"Feed loop for {state.config.id} has ended")
            
        except Exception as e:
            print(f"Feed {state.config.id} error: {e}")
            state.status = 'disconnected'
            state.error = str(e)
            
        finally:
            # Clean up resources
            try:
                if 'cap' in locals() and cap is not None:
                    cap.release()
                    print(f"Released capture for {state.config.id}")
            except Exception as release_error:
                print(f"Error releasing capture for {state.config.id}: {release_error}")
                
            # Update final status
            if state.status not in ('stopped', 'disconnected'):
                # If ended flag already set (e.g., file playback finished) leave as-is
                if state.status != 'ended':
                    # If loop exited normally without stop flag and without frames
                    if state.frame_count == 0:
                        state.status = 'disconnected'
                        state.error = state.error or 'no_frames_captured'
                    else:
                        state.status = 'stopped'

    # Utility -------------------------------------------------------------
    def ensure_started(self, feed_id: str):
        """Start feed if not live/connecting."""
        state = self._feeds.get(feed_id)
        if not state:
            return
        if state.status in ('stopped', 'disconnected'):
            self.start(feed_id)

    def health_snapshot(self) -> Dict[str, dict]:
        snap = {}
        for s in self.list_feeds():
            snap[s.config.id] = {
                'status': s.status,
                'error': s.error,
                'frames': getattr(s, 'frame_count', 0),
                'last_frame_age': (time.time() - s.last_frame_time) if s.last_frame_time else None,
                'type': s.config.type,
                'source': s.config.source,
            }
        return snap

# Example (manual) usage:
# mgr = CameraManager()
# mgr.add_feed(FeedConfig(id='cam1', source='0', type='webcam'))
# mgr.start('cam1')
# frame = mgr.get_frame('cam1')
