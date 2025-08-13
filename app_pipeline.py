"""
Streamlit application for multi-camera object detection
Uses universal preprocessi    /* Header container with glow effect */
    .header-container {
        display:    /* Main content area */
    .main .block-container {
        padding-top: 0;
        background: rgba(0,0,0,0.1);
        border-radius: 15px;
        backdrop-filter: blur(5px);
    }
        align-items: center;
        justify-content: space-between;
        padding: 15px 30px;
        background: rgba(28, 30, 38, 0.8);
        border-radius: 20px;
        margin: 0 0 25px 0;
        box-shadow: 0 0 30px rgba(75, 85, 150, 0.3), 0 8px 25px rgba(0,0,0,0.4);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.1);
        position: relative;
    } inference with Ultralytics
Each camera feed shows input and detection side-by-side with color-coded borders
"""
import streamlit as st
import cv2
import numpy as np
import time
import base64
from pathlib import Path
from typing import Callable

# Import universal modules
from preprocess import open_cam, make_views_rgb, MultiAngleWebcamCapture
from capture.camera_manager import CameraManager, FeedConfig
import uuid
from pipeline import Inference, get_device_config, discover_models
from postprocess import process_results, create_batch_summary


def safe_rerun():
    """Trigger a Streamlit rerun in both old and new versions.

    Streamlit <=1.30 used st.experimental_rerun; newer versions expose st.rerun.
    This helper silently no-ops if neither is present (avoids AttributeError).
    """
    fn: Callable | None = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if fn:
        fn()

# Set page config early (must be before first Streamlit UI call)
st.set_page_config(
    page_title="CraftEye - Multi-Camera Vision",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to encode image to base64
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return ""

# Function to create high-quality image HTML
def create_high_quality_image_html(image_path, width, alt_text, css_class=None):
    """Return an <img> tag with embedded base64 PNG.

    Args:
        image_path: path to local PNG asset
        width: target logical width in px
        alt_text: alt attribute for accessibility
        css_class: optional class name(s) to attach for external CSS styling
    """
    img_b64 = get_base64_image(image_path)
    cls_attr = f' class="{css_class}"' if css_class else ''
    if img_b64:
        return (
            f"<img src=\"data:image/png;base64,{img_b64}\"{cls_attr} width=\"{width}\" alt=\"{alt_text}\" "
            f"style=\"max-width: {width}px; height: auto; image-rendering: -webkit-optimize-contrast; filter: drop-shadow(4px 4px 14px rgba(0,0,0,0.75));\">"
        )
    return (
        f'<div{cls_attr} style="width: {width}px; height: 130px; background: rgba(255,255,255,0.05); '
        f'border: 1px dashed rgba(255,255,255,0.2); border-radius: 8px; display: flex; align-items: center; '
        f'justify-content: center; color: #888; font-size: 12px;">{alt_text}</div>'
    )

# Load images
background_b64 = get_base64_image("assets/background.png")
company_logo_path = "assets/company-logo.png"
project_logo_path = "assets/project-logo.png"

# Custom CSS for background, logos, and styling
st.markdown(f"""
<style>
    /* Main background with gradient overlay + image (single rule to avoid overrides) */
    .stApp {{
        /* Removed background image (contained unwanted text 'SEND MESSAGE').
           To restore image overlay, append: , url('data:image/png;base64,{background_b64}') */
        background: linear-gradient(180deg, #1C1E26 0%, #0F1015 100%);
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    /* Header container with glow effect */
    .header-container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 15px 30px;
        background: rgba(28, 30, 38, 0.8);
        border-radius: 20px;
        margin: 15px 0 25px 0;
        box-shadow: 0 0 30px rgba(75, 85, 150, 0.3), 0 8px 25px rgba(0,0,0,0.4);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.1);
        position: relative;
    }}
    
    /* Glow effect behind header */
    .header-container::before {{
        content: '';
        position: absolute;
        top: -10px;
        left: -10px;
        right: -10px;
        bottom: -10px;
        background: linear-gradient(45deg, rgba(75, 85, 150, 0.2), rgba(100, 120, 200, 0.2));
        border-radius: 25px;
        z-index: -1;
        filter: blur(15px);
    }}
    
    .company-logo, .project-logo {{
        height: 130px;
        width: auto;
        max-width: 220px;
        filter: drop-shadow(4px 4px 14px rgba(0,0,0,0.75));
        margin: -8px 18px 0 18px;
        image-rendering: -webkit-optimize-contrast;
        image-rendering: crisp-edges;
        image-rendering: pixelated;
    }}
    
    .main-title {{
        flex-grow: 1;
        text-align: center;
        color: #FFFFFF;
        font-size: 3.2rem;
        font-weight: 900;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.6), 0 0 20px rgba(255,255,255,0.1);
        letter-spacing: 1px;
    }}
    
    .subtitle {{
        color: #CCCCCC;
        font-size: 1.2rem;
        font-weight: 400;
        margin-top: 5px;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
    }}
    
    /* Section divider */
    .section-divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        margin: 20px 0;
    }}
    
    /* Sidebar styling */
    .css-1d391kg {{
        background: rgba(28, 30, 38, 0.95);
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }}
    
    /* Main content area */
    .main .block-container {{
        padding-top: 1rem;
        background: rgba(0,0,0,0.1);
        border-radius: 15px;
        backdrop-filter: blur(5px);
    }}
    
    /* Angle containers with improved colors */
    .angle-container {{
        padding: 20px;
        margin: 15px 0;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        background: rgba(28, 30, 38, 0.85);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.1);
    }}
    .front-angle {{
        border: 5px solid #00FF7F;
        background: linear-gradient(135deg, rgba(0, 255, 127, 0.15), rgba(0, 255, 127, 0.05));
        box-shadow: 0 0 20px rgba(0, 255, 127, 0.2), 0 8px 25px rgba(0,0,0,0.3);
    }}
    .right-angle {{
        border: 5px solid #00BFFF;
        background: linear-gradient(135deg, rgba(0, 191, 255, 0.15), rgba(0, 191, 255, 0.05));
        box-shadow: 0 0 20px rgba(0, 191, 255, 0.2), 0 8px 25px rgba(0,0,0,0.3);
    }}
    .rear-angle {{
        border: 5px solid #ff8000;
        background: linear-gradient(135deg, rgba(255, 128, 0, 0.15), rgba(255, 128, 0, 0.05));
        box-shadow: 0 0 20px rgba(255, 128, 0, 0.2), 0 8px 25px rgba(0,0,0,0.3);
    }}
    .left-angle {{
        border: 5px solid #ff0040;
        background: linear-gradient(135deg, rgba(255, 0, 64, 0.15), rgba(255, 0, 64, 0.05));
        box-shadow: 0 0 20px rgba(255, 0, 64, 0.2), 0 8px 25px rgba(0,0,0,0.3);
    }}
    
    .angle-title {{
        text-align: center;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 15px;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.8);
        letter-spacing: 0.5px;
    }}
    .front-title {{ color: #00FF7F; }}
    .right-title {{ color: #00BFFF; }}
    .rear-title {{ color: #ff8000; }}
    .left-title {{ color: #ff0040; }}
    
    .input-output-container {{
        display: flex;
        gap: 20px;
        align-items: center;
        justify-content: space-around;
    }}
    
    .video-frame {{
        border-radius: 12px;
        border: 3px solid rgba(255,255,255,0.2);
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }}
    
    /* Feed labels styling */
    .feed-label {{
        color: #FFFFFF;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 10px;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
    }}
    
    /* Object Detection pill badge */
    .detection-badge {{
        background: #FF0000;
        color: #FFFFFF;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(255, 0, 0, 0.3);
        text-shadow: none;
        letter-spacing: 0.5px;
    }}
    
    /* Success/info boxes styling */
    .stSuccess, .stInfo, .stWarning {{
        background: rgba(28, 30, 38, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
        color: #FFFFFF;
    }}
    
    /* Streamlit specific overrides */
    .stMarkdown {{
        color: #FFFFFF;
    }}
    
    h1, h2, h3 {{
        color: #FFFFFF !important;
    }}
    
    /* High quality image rendering */
    img {{
        image-rendering: -webkit-optimize-contrast;
        image-rendering: crisp-edges;
        image-rendering: pixelated;
        max-width: 100%;
        height: auto;
    }}
    
    /* Streamlit image container styling */
    .stImage > div {{
        display: flex;
        justify-content: center;
        align-items: center;
    }}
    
    .stImage img {{
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }}
</style>
""", unsafe_allow_html=True)

## (Page config moved to top)

# Custom header with logos
try:
    st.markdown(f"""
    <div class="header-container">
        <div>
            {create_high_quality_image_html(company_logo_path, 150, "CraftifAI Logo", css_class="company-logo")}
        </div>
        <div style="flex: 1; text-align: center; padding: 0 30px;">
            <h1 style="color: #FFFFFF; font-size: 3.2rem; font-weight: 900; margin: 0; text-shadow: 2px 2px 8px rgba(0,0,0,0.6), 0 0 20px rgba(255,255,255,0.1); letter-spacing: 1px;">
                CraftEye
            </h1>
            <div style="color: #CCCCCC; font-size: 1.2rem; font-weight: 400; margin-top: 5px; text-shadow: 1px 1px 3px rgba(0,0,0,0.5);">
                Deploy Today, Evolve Tomorrow.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Add section divider
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

except Exception as e:
    # Fallback if images don't load
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: rgba(28, 30, 38, 0.8); border-radius: 15px; margin: 15px 0;">
        <h1 style="color: #FFFFFF; font-size: 3.2rem; font-weight: 900; margin: 0; text-shadow: 2px 2px 8px rgba(0,0,0,0.6);">
            👁️ CraftEye
        </h1>
        <h3 style="color: #CCCCCC; margin: 5px 0 0 0; font-weight: 400;">
            Deploy Today, Evolve Tomorrow.
        </h3>
    </div>
    <div class="section-divider"></div>
    """, unsafe_allow_html=True)

# Initialize session state
if 'inference' not in st.session_state:
    st.session_state.inference = None
if 'capture' not in st.session_state:
    st.session_state.capture = None
if 'current_model_path' not in st.session_state:
    st.session_state.current_model_path = None
if 'current_task' not in st.session_state:
    st.session_state.current_task = None

############################
# Sidebar controls
############################
with st.sidebar:
    # Remove the main page label ("app pipeline") from sidebar navigation if present
    # Only show controls, not the main page label
    st.header("🎛️ Controls")
    
    # Define use_legacy as False since we're removing that functionality
    use_legacy = False

    # Auto-discover models once (shared by feeds)
    models = discover_models()
    if not models:
        st.error("❌ No models found in 'models' directory")
        st.stop()

    # Sort models (used for initializing defaults but selection moved to main UI)
    pt_models = [m for m in models if m['format'] == 'pt']
    onnx_models = [m for m in models if m['format'] == 'onnx']
    openvino_models = [m for m in models if m['format'] == 'openvino']
    engine_models = [m for m in models if m['format'] == 'engine']
    other_models = [m for m in models if m['format'] not in ['pt', 'onnx', 'openvino', 'engine']]
    sorted_models = pt_models + onnx_models + openvino_models + engine_models + other_models
    model_options = {f"{m['name']} ({m['format']})": m for m in sorted_models}
    
    # Set defaults (used when adding a feed)
    default_model_label = list(model_options.keys())[0] if model_options else ""
    default_model_path = model_options[default_model_label]['path'] if model_options else ""
    default_detected_task = model_options[default_model_label]['task'] or 'detect' if model_options else 'detect'
    task_options = ['detect', 'segment', 'classify', 'pose', 'obb']
    default_task_index = task_options.index(default_detected_task) if default_detected_task in task_options else 0
    default_task = task_options[default_task_index]

    st.markdown("---")
    st.subheader("📹 Camera Management")
    if 'cam_manager' not in st.session_state:
        st.session_state.cam_manager = CameraManager()
    if 'feeds_meta' not in st.session_state:
        st.session_state.feeds_meta = {}  # feed_id -> meta (model selections / inference objects)

    # Simplified camera source selection
    feed_type = st.selectbox("Source Type", ["webcam", "rtsp", "file"], key="add_type")
    
    # Handle different source types
    source_input = "0"  # Default for webcam (index 0 is typically the default/built-in webcam)
    
    # If webcam is selected, show a message about default webcam usage
    if feed_type == "webcam":
        st.info("Using default built-in webcam (index 0). If you have multiple cameras, you can change this below.")
        # Optional: Allow selection of different camera indices if the user has multiple webcams
        webcam_indices = ["0 (Default)", "1", "2", "3"]
        selected_index = st.selectbox("Select webcam index", webcam_indices)
        source_input = selected_index.split()[0]  # Extract just the number
        
    elif feed_type == "rtsp":
        source_input = st.text_input("RTSP URL", 
                                    value="rtsp://username:password@ip:port/stream",
                                    help="RTSP URL for streaming")
    elif feed_type == "file":
        uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            import tempfile
            import os
            
            # Create temp dir if it doesn't exist
            temp_dir = os.path.join(os.getcwd(), "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
                
            # Save file
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            source_input = file_path
            st.success(f"File saved: {file_path}")
        else:
            st.info("Please upload a video file")
    
    # Camera configuration
    st.subheader("Camera Configuration")
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        res_choice = st.selectbox("Resolution", ["640x480", "1280x720", "1920x1080"], index=1)
    with col_res2:
        fps_cap = st.number_input("FPS Cap", min_value=5, max_value=60, value=15, step=1)
        
    camera_name = st.text_input("Camera Name", value=f"Camera {len(st.session_state.feeds_meta)+1}", 
                              help="Give your camera a descriptive name")
    
    add_feed_btn = st.button("➕ Add Camera")
    if add_feed_btn and (feed_type == "webcam" or (feed_type != "webcam" and source_input.strip())):
        feed_id = str(uuid.uuid4())[:8]
        w, h = map(int, res_choice.split('x'))
        cfg = FeedConfig(id=feed_id, source=source_input.strip(), type=feed_type, resolution=(w, h), fps_cap=fps_cap,
                         task={'type': default_task, 'model': default_model_path})
        try:
            # Add the feed to camera manager
            st.session_state.cam_manager.add_feed(cfg)
            
            # Store metadata
            st.session_state.feeds_meta[feed_id] = {
                'name': camera_name,
                'primary_model_path': default_model_path,
                'primary_task': default_task,
                'primary_inference': None,
                'secondary_enabled': False,
                'secondary_model_path': None,
                'secondary_task': None,
                'secondary_inference': None,
            }
            
            # Automatically start the camera feed
            try:
                st.session_state.cam_manager.start(feed_id)
                st.success(f"✅ Added and started camera: {camera_name}")
            except Exception as start_err:
                st.error(f"Camera added but failed to start: {start_err}")
                
        except Exception as e:
            st.error(f"Failed to add camera: {e}")

    st.markdown("---")
    st.subheader("🖥️ Device")
    device_config = get_device_config(prefer_gpu=True)
    device_name = device_config.get('device_name', 'Unknown')
    device_type = device_config.get('device_type', 'unknown')
    if device_type == 'cuda':
        st.success(f"GPU: {device_name} (FP16: {device_config['half']})")
    elif device_type == 'mps':
        st.success(f"Apple GPU: {device_name}")
    elif device_type == 'openvino_gpu':
        st.success(f"Intel GPU: {device_name} (OpenVINO)")
    elif device_type == 'openvino_cpu':
        st.info(f"CPU: {device_name} (OpenVINO)")
    else:
        st.info(f"CPU: {device_name}")

    confidence = st.slider("Confidence Threshold", 0.1, 0.9, 0.5)
    imgsz = 640

    st.markdown("---")
    st.caption("Configure feeds below in main panel. Start Processing controls run loop.")

st.markdown("---")

# Dynamic feed configuration & display 
if 'feed_placeholders' not in st.session_state:
    st.session_state.feed_placeholders = {}

# Main content area
st.subheader("📡 Active Feeds")
if not st.session_state.feeds_meta:
    st.info("Add a camera from the sidebar to begin.")
else:
    for feed_state in st.session_state.cam_manager.list_feeds():
        fid = feed_state.config.id
        meta = st.session_state.feeds_meta.get(fid)
        if not meta:
            continue
        with st.container():
            st.markdown(f'<div class="angle-container">', unsafe_allow_html=True)
            # Header row with name + status + controls
            cols_top = st.columns([3,1,1,1])
            with cols_top[0]:
                meta['name'] = st.text_input("Name", value=meta['name'], key=f"name_{fid}")
            with cols_top[1]:
                if feed_state.status != 'live':
                    if st.button("Start", key=f"start_{fid}"):
                        st.session_state.cam_manager.start(fid)
                else:
                    if st.button("Stop", key=f"stop_{fid}"):
                        st.session_state.cam_manager.stop(fid)
            with cols_top[2]:
                if st.button("Remove", key=f"rem_{fid}"):
                    st.session_state.cam_manager.remove(fid)
                    st.session_state.feeds_meta.pop(fid, None)
                    st.session_state.feed_placeholders.pop(fid, None)
                    safe_rerun()
            with cols_top[3]:
                status = feed_state.status
                if status == "live":
                    st.markdown(f"**Status:** <span style='color:green;font-weight:bold;'>LIVE</span> ({feed_state.frame_count} frames)", unsafe_allow_html=True)
                elif status == "connecting":
                    st.markdown(f"**Status:** <span style='color:orange;font-weight:bold;'>CONNECTING</span>", unsafe_allow_html=True)
                elif status == "disconnected":
                    st.markdown(f"**Status:** <span style='color:red;font-weight:bold;'>DISCONNECTED</span>", unsafe_allow_html=True)
                    st.markdown(f"<span style='color:red;font-size:0.8em;'>Error: {feed_state.error}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Status:** {status}")

            # Model selections
            st.subheader("🤖 Model Selection")
            cols_models = st.columns([2,2,1])
            with cols_models[0]:
                primary_label = st.selectbox("Primary Model", list(model_options.keys()),
                                             index=list(model_options.keys()).index(default_model_label) if meta['primary_model_path']==default_model_path else 0,
                                             key=f"prim_model_{fid}")
                meta['primary_model_path'] = model_options[primary_label]['path']
                meta['primary_task'] = st.selectbox("Primary Task", task_options,
                                                    index=task_options.index(meta['primary_task']) if meta['primary_task'] in task_options else 0,
                                                    key=f"prim_task_{fid}")
            with cols_models[1]:
                meta['secondary_enabled'] = st.checkbox("Enable Secondary Model", value=meta['secondary_enabled'], key=f"sec_enable_{fid}")
                if meta['secondary_enabled']:
                    sec_label = st.selectbox("Secondary Model", list(model_options.keys()), key=f"sec_model_{fid}")
                    meta['secondary_model_path'] = model_options[sec_label]['path']
                    meta['secondary_task'] = st.selectbox("Secondary Task", task_options,
                                                          index=task_options.index(meta['secondary_task']) if meta['secondary_task'] in task_options else 0,
                                                          key=f"sec_task_{fid}")
                else:
                    meta['secondary_model_path'] = None
                    meta['secondary_task'] = None
            with cols_models[2]:
                st.write(" ")
                if st.button("Reload Models", key=f"reload_{fid}"):
                    meta['primary_inference'] = None
                    meta['secondary_inference'] = None
                    st.success("Models reset")

            # Placeholders (create/update)
            model_cols = st.columns(3)
            with model_cols[0]:
                st.markdown("<p class='feed-label' style='text-align: center;'>Live Camera Feed</p>", unsafe_allow_html=True)
            with model_cols[1]:
                primary_task_label = meta['primary_task'].capitalize()
                st.markdown(f"<p class='feed-label' style='text-align: center;'>{primary_task_label} Detection</p>", unsafe_allow_html=True)
            with model_cols[2]:
                if meta['secondary_enabled']:
                    secondary_task_label = meta['secondary_task'].capitalize() if meta['secondary_task'] else "N/A"
                    st.markdown(f"<p class='feed-label' style='text-align: center;'>{secondary_task_label} Detection</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p class='feed-label' style='text-align: center; opacity: 0.5;'>Secondary Model (Disabled)</p>", unsafe_allow_html=True)

            cols_disp = st.columns(3)
            st.session_state.feed_placeholders[fid] = {
                'input': cols_disp[0].empty(),
                'primary': cols_disp[1].empty(),
                'secondary': cols_disp[2].empty()
            }
            st.markdown('</div>', unsafe_allow_html=True)

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

start_col, stop_col = st.columns(2)
with start_col:
    start_stream = st.button("🚀 Start Processing", key="start")
with stop_col:
    stop_stream = st.button("🛑 Stop", key="stop")

# Feed health diagnostics (always visible for debugging source issues)
health_exp = st.expander("🩺 Feed Health / Diagnostics", expanded=False)
with health_exp:
    if 'cam_manager' in st.session_state:
        snap = st.session_state.cam_manager.health_snapshot()
        if not snap:
            st.write("No feeds added.")
        else:
            for fid, info in snap.items():
                cols = st.columns([2,2,2,2,2])
                cols[0].markdown(f"**ID:** `{fid}`")
                cols[1].markdown(f"**Type:** {info['type']}")
                cols[2].markdown(f"**Status:** {info['status']}")
                cols[3].markdown(f"**Frames:** {info['frames']}")
                age = info['last_frame_age']
                age_str = f"{age:.1f}s" if age is not None else "-"
                cols[4].markdown(f"**Last Frame Age:** {age_str}")
                if info['error']:
                    st.warning(f"Error: {info['error']}")
            st.caption("If a webcam stays in 'connecting' >2s or frames remain 0, ensure no other app is using it and index is correct (0 is default). For RTSP ensure the stream is reachable.")


# Universal inference loop
if start_stream:
    # ---- MULTI-FEED PATH -----------------------------------------
    st.info("Running multi-feed processing loop")

    # Auto-start any feeds that are not running yet
    if 'cam_manager' in st.session_state:
        for feed_state in st.session_state.cam_manager.list_feeds():
            if feed_state.status in ('stopped', 'disconnected'):
                try:
                    st.session_state.cam_manager.start(feed_state.config.id)
                except Exception as autostart_err:
                    st.warning(f"Auto-start failed for {feed_state.config.id}: {autostart_err}")

    # Main inference loop (shared timing metrics) ---------------------------
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
        
        # === PREPROCESS (multi-feed): collect live frames ===
        feed_frames = []  # list of (fid, frame_rgb)
        view_names = []
        feeds_list = st.session_state.cam_manager.list_feeds()
        
        # Debug info
        if not feeds_list:
            st.warning("No camera feeds found. Please add a camera from the sidebar.")
            
        waiting_fids = []
        ended_fids = []
        for feed_state in feeds_list:
            if feed_state.status == 'ended':
                ended_fids.append(feed_state.config.id)
                continue
            if feed_state.status != 'live':
                waiting_fids.append(feed_state.config.id)
                continue
                
            if feed_state.last_frame is None:
                st.warning(f"Camera {feed_state.config.id} has no frames yet")
                continue
                
            fid = feed_state.config.id
            frame_bgr = feed_state.last_frame
            frame_rgb = frame_bgr[:, :, ::-1]
            feed_frames.append((fid, frame_rgb))
            view_names.append(fid)
            meta = st.session_state.feeds_meta.get(fid, {})
            
            # Lazy load primary model
            if meta.get('primary_inference') is None:
                try:
                    st.info(f"Loading model for {meta.get('name', fid)}: {meta.get('primary_model_path')}")
                    meta['primary_inference'] = Inference(
                        model_path=meta['primary_model_path'],
                        imgsz=imgsz,
                        conf=confidence,
                        device=device_config['device'],
                        half=device_config['half'],
                        task=meta['primary_task']
                    )
                    st.success(f"Model loaded for {meta.get('name', fid)}")
                except Exception as e:
                    st.error(f"Primary model load failed ({fid}): {e}")
                    
            # Lazy load secondary if enabled
            if meta.get('secondary_enabled') and meta.get('secondary_inference') is None and meta.get('secondary_model_path'):
                try:
                    meta['secondary_inference'] = Inference(
                        model_path=meta['secondary_model_path'],
                        imgsz=imgsz,
                        conf=confidence,
                        device=device_config['device'],
                        half=device_config['half'],
                        task=meta['secondary_task']
                    )
                except Exception as e:
                    st.error(f"Secondary model load failed ({fid}): {e}")
        
        t1 = time.perf_counter()
        
        # Run inference sequentially per live feed (simple baseline)
        per_feed_outputs = {}
        for fid, frame_rgb in feed_frames:
            meta = st.session_state.feeds_meta.get(fid, {})
            primary_result = None
            secondary_result = None
            if meta.get('primary_inference'):
                primary_result = meta['primary_inference'].predict_single(frame_rgb)
            if meta.get('secondary_enabled') and meta.get('secondary_inference'):
                secondary_result = meta['secondary_inference'].predict_single(frame_rgb)
            per_feed_outputs[fid] = (frame_rgb, primary_result, secondary_result)
        
        t2 = time.perf_counter()
        
        # Postprocess timings approximated
        actual_inference_ms = sum([
            meta['primary_inference'].get_last_inference_time_ms() if meta.get('primary_inference') else 0
            for fid, meta in st.session_state.feeds_meta.items()
        ])
        
        # Define processable flag
        processable = True if per_feed_outputs else False

        if waiting_fids or ended_fids:
            # Throttle messages: update every ~1s using time bucket
            bucket = int(time.time())
            last_bucket = st.session_state.get('status_msg_bucket')
            if last_bucket != bucket:
                st.session_state['status_msg_bucket'] = bucket
                if waiting_fids:
                    st.info(f"Waiting for feeds: {', '.join(waiting_fids)}")
                if ended_fids:
                    st.success(f"Completed file feeds: {', '.join(ended_fids)}")
        if not processable and not waiting_fids:
            # If we have frames_list but none processable (shouldn't happen) or no feeds
            if not feed_frames:
                st.warning("⚠️ No active camera feeds available. Check that cameras are started.")

        if processable:
            # === POSTPROCESS TIMING ===
            # For multi-feed, postprocess each result separately
            annotated_images = []
            summaries = []
            for fid, (frame_rgb, prim_res, sec_res) in per_feed_outputs.items():
                prim_ann, prim_sum = ([], [{'total_detections':0}])
                if prim_res is not None:
                    a_imgs, s_sums = process_results([prim_res], save=False, want_json=True)
                    if a_imgs:
                        prim_ann = a_imgs
                    if s_sums:
                        prim_sum = s_sums
                # Store combined (primary only for metrics)
                annotated_images.append((fid, prim_ann[0] if prim_ann else None, prim_sum[0]))

            # Add static display for ended feeds (file sources finished)
            static_ended = []
            for fid in ended_fids:
                feed_state = next((fs for fs in st.session_state.cam_manager.list_feeds() if fs.config.id == fid), None)
                if not feed_state or feed_state.last_frame is None:
                    continue
                meta = st.session_state.feeds_meta.get(fid, {})
                # Ensure model loaded if we want a snapshot
                if meta.get('primary_inference') is None:
                    try:
                        meta['primary_inference'] = Inference(
                            model_path=meta['primary_model_path'],
                            imgsz=imgsz,
                            conf=confidence,
                            device=device_config['device'],
                            half=device_config['half'],
                            task=meta['primary_task']
                        )
                    except Exception as e:
                        st.error(f"Snapshot model load failed ({fid}): {e}")
                        continue
                # Generate snapshot once and cache
                if 'ended_cached' not in meta:
                    try:
                        snap_res = meta['primary_inference'].predict_single(feed_state.last_frame[:, :, ::-1])
                        a_imgs, s_sums = process_results([snap_res], save=False, want_json=True)
                        meta['ended_cached'] = {
                            'image': a_imgs[0] if a_imgs else None,
                            'summary': s_sums[0] if s_sums else {'total_detections':0}
                        }
                    except Exception as e:
                        meta['ended_cached'] = {'image': None, 'summary': {'total_detections':0}}
                        st.warning(f"Snapshot inference failed for ended feed {fid}: {e}")
                cached = meta.get('ended_cached', {})
                annotated_images.append((fid, cached.get('image'), cached.get('summary')))
                static_ended.append(fid)
            t3 = time.perf_counter()
            
            # === Calculate Ultralytics-style timings ===
            preprocess_ms = (t1 - t0) * 1000
            inference_ms = actual_inference_ms  # aggregated or batch
            postprocess_ms = (t3 - t2) * 1000
            e2e_ms = (t3 - t0) * 1000
            
            # === EMA Smoothed FPS (Ultralytics style) ===
            instant_fps = 1000.0 / e2e_ms if e2e_ms > 0 else 0
            if st.session_state.ema_fps == 0.0:
                st.session_state.ema_fps = instant_fps  # Initialize on first frame
            else:
                st.session_state.ema_fps = alpha * instant_fps + (1 - alpha) * st.session_state.ema_fps
            
            # Display each feed in its container via placeholders
            for fid, ann_img, summary in annotated_images:
                placeholders = st.session_state.feed_placeholders.get(fid)
                meta = st.session_state.feeds_meta.get(fid, {})
                feed_state = next((fs for fs in st.session_state.cam_manager.list_feeds() if fs.config.id == fid), None)
                if not placeholders or not feed_state:
                    continue
                # Input frame (latest BGR from state)
                if feed_state.last_frame is not None:
                    cap_text = "Live" if feed_state.status == 'live' else ("Completed" if feed_state.status == 'ended' else feed_state.status)
                    placeholders['input'].image(feed_state.last_frame, channels="BGR", use_container_width=True, caption=cap_text)
                # Primary output
                if ann_img is not None:
                    det_count = summary.get('total_detections', 0) if isinstance(summary, dict) else 0
                    primary_task_label = meta['primary_task'].capitalize()
                    suffix = " (static)" if feed_state.status == 'ended' else ""
                    placeholders['primary'].image(ann_img, channels="BGR", use_container_width=True, 
                                                  caption=f"{primary_task_label} Objects: {det_count}{suffix}")
                # Secondary output (if enabled)
                if meta.get('secondary_enabled'):
                    sec_inf = meta.get('secondary_inference')
                    feed_frame_rgb = feed_state.last_frame[:, :, ::-1] if feed_state.last_frame is not None else None
                    if sec_inf and feed_frame_rgb is not None:
                        sec_res = sec_inf.predict_single(feed_frame_rgb)
                        sec_ann, sec_sum = process_results([sec_res], save=False, want_json=True)
                        det2 = sec_sum[0].get('total_detections', 0) if sec_sum else 0
                        secondary_task_label = meta['secondary_task'].capitalize() if meta['secondary_task'] else "Object"
                        placeholders['secondary'].image(sec_ann[0], channels="BGR", use_container_width=True, 
                                                         caption=f"{secondary_task_label} Objects: {det2}{suffix if feed_state.status == 'ended' else ''}")
            
            # === Update Ultralytics-style Metrics ===
            total_detections = sum([s.get('total_detections',0) for _,_,s in annotated_images if s])
            
            # Main metrics (like Ultralytics CLI)
            fps_placeholder.metric("🚀 FPS (EMA)", f"{st.session_state.ema_fps:.1f}")
            detections_placeholder.metric("🎯 Objects Detected", total_detections)
            e2e_time_placeholder.metric("⏱️ E2E Latency (ms)", f"{e2e_ms:.0f}")
            
            # Detailed breakdown (Speed: X.Xms preprocess, Y.Yms inference, Z.Zms postprocess)
            preprocess_placeholder.metric("� Preprocess", f"{preprocess_ms:.1f}ms")
            inference_placeholder.metric("🧠 Inference", f"{inference_ms:.1f}ms") 
            postprocess_placeholder.metric("🎨 Postprocess", f"{postprocess_ms:.1f}ms")
        else:
            # Handle case where inference failed with more detailed info
            if not feed_frames:
                st.warning("⚠️ No active camera feeds available. Check that cameras are started.")
            else:
                missing_models = []
                for fid, meta in st.session_state.feeds_meta.items():
                    if meta.get('primary_inference') is None:
                        missing_models.append(meta.get('name', fid))
                
                if missing_models:
                    st.warning(f"⚠️ Models not loaded for: {', '.join(missing_models)}. Check error messages above.")
                else:
                    st.warning("⚠️ Inference failed or returned incomplete results. Check camera status.")
        
        # Check for stop condition
        stop_stream = st.session_state.get("stop", False)
        
        # Small delay for UI responsiveness
        time.sleep(0.03)  # ~30 FPS max
    
    # Cleanup
    st.success("✅ Stream stopped.")

elif not start_stream:
    st.info("👆 Add a camera from the sidebar, then click 'Start Processing' to begin object detection")
