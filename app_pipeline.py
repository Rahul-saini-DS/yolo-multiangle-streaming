"""
Streamlit application for YOLO multi-angle webcam inference
Uses universal preprocessing and batched inference with Ultralytics
Each angle shows input and detection side-by-side with color-coded borders
"""
import streamlit as st
import cv2
import numpy as np
import time

# Import universal modules
from preprocess import open_cam, make_views_rgb, MultiAngleWebcamCapture
from pipeline import Inference, get_device_config, discover_models
from postprocess import process_results, create_batch_summary

# Custom CSS for color-coded borders and uniform sizing
st.markdown("""
<style>
    .angle-container {
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .front-angle {
        border: 4px solid #00ff00;  /* Green */
        background: rgba(0, 255, 0, 0.1);
    }
    .right-angle {
        border: 4px solid #0080ff;  /* Blue */
        background: rgba(0, 128, 255, 0.1);
    }
    .rear-angle {
        border: 4px solid #ff8000;  /* Orange */
        background: rgba(255, 128, 0, 0.1);
    }
    .left-angle {
        border: 4px solid #ff0040;  /* Red */
        background: rgba(255, 0, 64, 0.1);
    }
    .angle-title {
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 15px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .front-title { color: #00ff00; }
    .right-title { color: #0080ff; }
    .rear-title { color: #ff8000; }
    .left-title { color: #ff0040; }
    
    .input-output-container {
        display: flex;
        gap: 20px;
        align-items: center;
    }
    .video-frame {
        border-radius: 8px;
        border: 2px solid rgba(255,255,255,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="Live Multi-Angle YOLO",
    page_icon="📹",
    layout="wide"
)

st.title("🎥 Live Multi-Angle YOLO Detection - Universal Pipeline")

# Initialize session state
if 'inference' not in st.session_state:
    st.session_state.inference = None
if 'capture' not in st.session_state:
    st.session_state.capture = None
if 'current_model_path' not in st.session_state:
    st.session_state.current_model_path = None
if 'current_task' not in st.session_state:
    st.session_state.current_task = None

# Sidebar controls
with st.sidebar:
    st.header("🎛️ Controls")
    
    # Auto-discover models
    models = discover_models()
    if models:
        # Enhanced prioritization: .pt → .onnx → .openvino → .engine → others
        pt_models = [m for m in models if m['format'] == 'pt']
        onnx_models = [m for m in models if m['format'] == 'onnx']
        openvino_models = [m for m in models if m['format'] == 'openvino']
        engine_models = [m for m in models if m['format'] == 'engine']
        other_models = [m for m in models if m['format'] not in ['pt', 'onnx', 'openvino', 'engine']]
        
        sorted_models = pt_models + onnx_models + openvino_models + engine_models + other_models
        model_options = {f"{m['name']} ({m['format']})": m['path'] for m in sorted_models}
        
        selected_model_name = st.selectbox("Select Model:", list(model_options.keys()))
        model_path = model_options[selected_model_name]
        
        # Enhanced format information
        selected_format = next(m['format'] for m in sorted_models if f"{m['name']} ({m['format']})" == selected_model_name)
        detected_task = next(m['task'] for m in sorted_models if f"{m['name']} ({m['format']})" == selected_model_name)
        
        # Task selection with auto-detection
        st.markdown("**🎯 Model Task:**")
        
        # Define available tasks
        task_options = ['detect', 'segment', 'classify', 'pose', 'obb']
        
        # Try to set default based on detected task, fallback to 'detect'
        default_task_index = 0  # Default to 'detect'
        if detected_task and detected_task in task_options:
            default_task_index = task_options.index(detected_task)
        
        selected_task = st.selectbox(
            "Choose task type:",
            task_options,
            index=default_task_index,
            help="Auto-detected from model name, but you can override if needed"
        )
        
        # Show task info
        if detected_task and detected_task != selected_task:
            st.warning(f"⚠️ Auto-detected: {detected_task}, but using: {selected_task}")
        elif detected_task:
            st.info(f"✅ Auto-detected task: {detected_task}")
        else:
            st.info(f"🔧 Using task: {selected_task}")
        
        # Check if model or task changed (need to reinitialize)
        model_changed = (st.session_state.current_model_path != model_path or 
                        st.session_state.current_task != selected_task)
        
        if model_changed and st.session_state.inference is not None:
            st.warning("🔄 Model/task changed - will reinitialize on next start")
            st.session_state.inference = None  # Force reinitialization
        
        # Manual reload button
        if st.button("🔄 Reload Model", help="Force reload the model with current settings"):
            st.session_state.inference = None
            if st.session_state.capture:
                st.session_state.capture.release()
                st.session_state.capture = None
            st.success("✅ Model will be reloaded on next start")
        
        st.markdown("---")
        
        if selected_format == 'pt':
            st.info(f"🚀 PyTorch model ({selected_task}) - optimized for batch processing")
        elif selected_format == 'onnx':
            st.info(f"⚡ ONNX model ({selected_task}) - individual frame fallback")
        elif selected_format == 'openvino':
            st.info(f"⚡ OpenVINO model ({selected_task}) - Intel CPU/GPU optimized (up to 3x faster)")
        elif selected_format == 'engine':
            st.info(f"🏎️ TensorRT engine ({selected_task}) - NVIDIA GPU optimized")
        else:
            st.info(f"🔧 {selected_format.upper()} model ({selected_task})")
    else:
        st.error("❌ No models found in 'models' directory")
        st.stop()
    
    # Enhanced device configuration display
    device_config = get_device_config(prefer_gpu=True)
    device_name = device_config.get('device_name', 'Unknown')
    device_type = device_config.get('device_type', 'unknown')
    
    if device_type == 'cuda':
        st.success(f"� GPU: {device_name} (FP16: {device_config['half']})")
    elif device_type == 'mps':
        st.success(f"🍎 Apple GPU: {device_name}")
    elif device_type == 'openvino_gpu':
        st.success(f"⚡ Intel GPU: {device_name} (OpenVINO accelerated)")
    elif device_type == 'openvino_cpu':
        st.info(f"⚡ CPU: {device_name} (OpenVINO accelerated)")
    else:
        st.info(f"🔧 CPU: {device_name}")
    
    # Confidence threshold
    confidence = st.slider("Confidence Threshold:", 0.1, 0.9, 0.5)
    
    # Always use 640x640 for YOLO models (Ultralytics standard)
    imgsz = 640

# Main layout: Each angle shows input and detection side-by-side
st.markdown("---")

# Create containers for each angle with color coding
angles_data = [
    {"name": "Front View (0°)", "emoji": "🟢", "class": "front-angle", "title_class": "front-title"},
    {"name": "Right View (90°)", "emoji": "🔵", "class": "right-angle", "title_class": "right-title"}, 
    {"name": "Rear View (180°)", "emoji": "🟠", "class": "rear-angle", "title_class": "rear-title"},
    {"name": "Left View (270°)", "emoji": "🔴", "class": "left-angle", "title_class": "left-title"}
]

# Create placeholders for each angle
angle_containers = []
for i, angle_data in enumerate(angles_data):
    with st.container():
        st.markdown(f'<div class="angle-container {angle_data["class"]}">', unsafe_allow_html=True)
        st.markdown(f'<div class="angle-title {angle_data["title_class"]}">{angle_data["emoji"]} {angle_data["name"]}</div>', unsafe_allow_html=True)
        
        # Create two columns for input and output side-by-side
        input_col, output_col = st.columns(2)
        
        with input_col:
            st.markdown("**📹 Live Feed**")
            input_placeholder = st.empty()
        
        with output_col:
            st.markdown("**🎯 YOLO Detection**")
            output_placeholder = st.empty()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Store placeholders
        angle_containers.append({
            'input': input_placeholder,
            'output': output_placeholder
        })

# Performance metrics - Ultralytics style
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
fps_placeholder = metrics_col1.empty()
detections_placeholder = metrics_col2.empty() 
e2e_time_placeholder = metrics_col3.empty()

# Detailed timing breakdown (Ultralytics CLI style)
timing_expander = st.expander("📊 Detailed Performance Metrics", expanded=False)
with timing_expander:
    timing_col1, timing_col2, timing_col3 = st.columns(3)
    preprocess_placeholder = timing_col1.empty()
    inference_placeholder = timing_col2.empty()
    postprocess_placeholder = timing_col3.empty()

# Control buttons
start_col, stop_col = st.columns(2)
with start_col:
    start_stream = st.button("🚀 Start Live Stream", key="start")
with stop_col:
    stop_stream = st.button("🛑 Stop Stream", key="stop")


# Universal inference loop using the new pipeline
if start_stream:
    # Initialize inference pipeline
    if st.session_state.inference is None:
        try:
            with st.spinner("Loading YOLO model..."):
                st.session_state.inference = Inference(
                    model_path=model_path,
                    imgsz=imgsz,
                    conf=confidence,
                    device=device_config['device'],
                    half=device_config['half'],
                    task=selected_task  # Explicitly pass the task
                )
                
                # Store current model/task for change detection
                st.session_state.current_model_path = model_path
                st.session_state.current_task = selected_task
            
            # Show detailed model verification
            model_info = st.session_state.inference._get_model_info()
            st.success("✅ YOLO model loaded successfully!")
            
            # Display which model is actually being used
            with st.expander("📋 Model Details", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**📁 File:** `{model_info['filename']}`")
                    st.write(f"**🏷️ Format:** {model_info['format']}")
                    st.write(f"**📏 Size:** {model_info['size_mb']} MB")
                with col2:
                    st.write(f"**🎯 Task:** {model_info['task']}")
                    st.write(f"**📐 Input:** {imgsz}×{imgsz}")
                    st.write(f"**🔧 Device:** {device_config['device']}")
                
                # Verification button
                if st.button("🔍 Verify Model Usage", help="Confirm which model is actively being used"):
                    verification = st.session_state.inference.verify_model_usage()
                    st.json(verification)
                    
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            st.stop()
    
    # Initialize webcam capture
    if st.session_state.capture is None:
        st.session_state.capture = MultiAngleWebcamCapture(
            webcam_index=0, 
            target_size=(640, 640)  # Standard YOLO input size
        )
        
        if not st.session_state.capture.open():
            st.error("❌ Failed to open webcam")
            st.stop()
        else:
            st.success("✅ Webcam opened successfully!")
    
    # Main inference loop with Ultralytics-style metrics
    frame_count = 0
    last_time = time.time()
    max_frames = 1000  # Reset after 1000 frames to prevent memory issues
    
    # Initialize EMA for stable FPS
    if 'ema_fps' not in st.session_state:
        st.session_state.ema_fps = 0.0
    alpha = 0.1  # EMA smoothing factor
    
    while not stop_stream:
        # === Start timing (Ultralytics style) ===
        t0 = time.perf_counter()
        
        # Reset frame count periodically to prevent memory buildup
        if frame_count > max_frames:
            frame_count = 0
            last_time = time.time()
        
        # === PREPROCESS TIMING ===
        # Capture 4 synchronized views using universal preprocessing
        success, views, view_names = st.session_state.capture.capture_multi_angle_frames()
        t1 = time.perf_counter()
        
        if not success:
            st.error("⚠️ Failed to capture frames")
            break
        
        # === INFERENCE TIMING ===
        # Single batched inference call for all 4 views
        results = st.session_state.inference.predict_batch(views)
        t2 = time.perf_counter()
        
        # Get actual inference time from pipeline (Ultralytics internal timing)
        actual_inference_ms = st.session_state.inference.get_last_inference_time_ms()
        
        if results and len(results) >= 4:
            # === POSTPROCESS TIMING ===
            # Use minimal postprocessing with Ultralytics best practices
            annotated_images, summaries = process_results(
                results, 
                save=False,  # Could enable save=True for debugging
                want_json=True
            )
            t3 = time.perf_counter()
            
            # === Calculate Ultralytics-style timings ===
            preprocess_ms = (t1 - t0) * 1000
            # Use actual Ultralytics inference time (more accurate)
            inference_ms = actual_inference_ms
            postprocess_ms = (t3 - t2) * 1000
            e2e_ms = (t3 - t0) * 1000
            
            # === EMA Smoothed FPS (Ultralytics style) ===
            instant_fps = 1000.0 / e2e_ms if e2e_ms > 0 else 0
            if st.session_state.ema_fps == 0.0:
                st.session_state.ema_fps = instant_fps  # Initialize on first frame
            else:
                st.session_state.ema_fps = alpha * instant_fps + (1 - alpha) * st.session_state.ema_fps
            
            # Display each angle in its container
            for i, (input_frame, output_frame, summary, angle_name) in enumerate(zip(views, annotated_images, summaries, view_names)):
                # Ensure we don't exceed available containers
                if i >= len(angle_containers):
                    break
                
                try:
                    # Convert RGB to display format for input
                    input_display = cv2.cvtColor(input_frame, cv2.COLOR_RGB2BGR) if input_frame is not None else None
                    
                    # Display input frame
                    if input_display is not None:
                        angle_containers[i]['input'].image(
                            input_display, 
                            channels="BGR", 
                            use_container_width=True,
                            caption=f"Live Feed - {angle_name}"
                        )
                    
                    # Display detection output
                    if output_frame is not None:
                        # Handle both Ultralytics JSON format and fallback
                        detection_count = len(summary) if isinstance(summary, list) else summary.get('total_detections', 0)
                        
                        angle_containers[i]['output'].image(
                            output_frame, 
                            channels="BGR", 
                            use_container_width=True,
                            caption=f"Detections: {detection_count}"
                        )
                    else:
                        # Show original frame if no output
                        if input_display is not None:
                            angle_containers[i]['output'].image(
                                input_display, 
                                channels="BGR", 
                                use_container_width=True,
                                caption="Processing..."
                            )
                except Exception as img_error:
                    # Handle image display errors gracefully
                    if "MediaFileStorageError" not in str(img_error):
                        st.error(f"Image display error: {img_error}")
                    continue
            
            # === Update Ultralytics-style Metrics ===
            batch_summary = create_batch_summary(summaries)
            total_detections = batch_summary.get('total_detections', 0)
            
            # Main metrics (like Ultralytics CLI)
            fps_placeholder.metric("🚀 FPS (EMA)", f"{st.session_state.ema_fps:.1f}")
            detections_placeholder.metric("🎯 Total Detections", total_detections)
            e2e_time_placeholder.metric("⏱️ E2E Latency (ms)", f"{e2e_ms:.0f}")
            
            # Detailed breakdown (Speed: X.Xms preprocess, Y.Yms inference, Z.Zms postprocess)
            preprocess_placeholder.metric("� Preprocess", f"{preprocess_ms:.1f}ms")
            inference_placeholder.metric("🧠 Inference", f"{inference_ms:.1f}ms") 
            postprocess_placeholder.metric("🎨 Postprocess", f"{postprocess_ms:.1f}ms")
        else:
            # Handle case where inference failed
            st.warning("⚠️ Inference failed or returned incomplete results")
        
        # Check for stop condition
        stop_stream = st.session_state.get("stop", False)
        
        # Small delay for UI responsiveness
        time.sleep(0.03)  # ~30 FPS max
    
    # Cleanup
    if st.session_state.capture:
        st.session_state.capture.release()
        st.session_state.capture = None
    
    st.success("✅ Stream stopped.")

elif not start_stream:
    st.info("👆 Click 'Start Live Stream' to begin universal multi-angle YOLO detection")
