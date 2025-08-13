import streamlit as st
from capture.camera_manager import CameraManager, FeedConfig
import uuid
from pipeline import get_device_config

st.session_state.setdefault('step', 4)
st.title('Camera Configuration')

# Initialize camera manager and feeds_meta
if 'cam_manager' not in st.session_state:
    st.session_state.cam_manager = CameraManager()
if 'feeds_meta' not in st.session_state:
    st.session_state.feeds_meta = {}

feed_type = st.selectbox("Source Type", ["webcam", "rtsp", "file"], key="add_type")
source_input = "0"
if feed_type == "webcam":
    st.info("Using default built-in webcam (index 0). If you have multiple cameras, you can change this below.")
    webcam_indices = ["0 (Default)", "1", "2", "3"]
    selected_index = st.selectbox("Select webcam index", webcam_indices)
    source_input = selected_index.split()[0]
elif feed_type == "rtsp":
    source_input = st.text_input("RTSP URL", value="rtsp://username:password@ip:port/stream", help="RTSP URL for streaming")
elif feed_type == "file":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        import tempfile, os
        temp_dir = os.path.join(os.getcwd(), "temp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        source_input = file_path
        st.success(f"File saved: {file_path}")
    else:
        st.info("Please upload a video file")

col_res1, col_res2 = st.columns(2)
with col_res1:
    res_choice = st.selectbox("Resolution", ["640x480", "1280x720", "1920x1080"], index=1)
with col_res2:
    fps_cap = st.number_input("FPS Cap", min_value=5, max_value=60, value=15, step=1)

camera_name = st.text_input("Camera Name", value=f"Camera {len(st.session_state.feeds_meta)+1}", help="Give your camera a descriptive name")
add_feed_btn = st.button("➕ Add Camera")

default_model_path = "models/yolov8n.pt"
default_task = "detect"

if add_feed_btn and (feed_type == "webcam" or (feed_type != "webcam" and source_input.strip())):
    feed_id = str(uuid.uuid4())[:8]
    w, h = map(int, res_choice.split('x'))
    cfg = FeedConfig(id=feed_id, source=source_input.strip(), type=feed_type, resolution=(w, h), fps_cap=fps_cap,
                     task={'type': default_task, 'model': default_model_path})
    try:
        st.session_state.cam_manager.add_feed(cfg)
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
        try:
            st.session_state.cam_manager.start(feed_id)
            st.success(f"✅ Added and started camera: {camera_name}")
        except Exception as start_err:
            st.error(f"Camera added but failed to start: {start_err}")
    except Exception as e:
        st.error(f"Failed to add camera: {e}")

st.markdown("---")
st.subheader("Active Cameras")
feeds = st.session_state.cam_manager.list_feeds()
if feeds:
    for feed in feeds:
        fid = feed.config.id
        meta = st.session_state.feeds_meta.get(fid, {})
        st.write(f"{meta.get('name', fid)} | {feed.config.type} | {feed.config.resolution} @ {feed.config.fps_cap} FPS | Status: {feed.status}")
        cols = st.columns([1,1,1])
        with cols[0]:
            if st.button(f"Start {fid}"):
                st.session_state.cam_manager.start(fid)
        with cols[1]:
            if st.button(f"Stop {fid}"):
                st.session_state.cam_manager.stop(fid)
        with cols[2]:
            if st.button(f"Remove {fid}"):
                st.session_state.cam_manager.remove(fid)
                st.session_state.feeds_meta.pop(fid, None)
                st.experimental_rerun()
else:
    st.info('No cameras added yet.')

if st.button('Next'):
    st.session_state['step'] = 5
