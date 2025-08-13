
import streamlit as st
import time
from pipeline import Inference
from postprocess import process_results

st.session_state.setdefault('step', 6)
st.title('Live Monitoring')
st.markdown('Real-time computer vision analysis')

# Ensure camera manager and feeds_meta are initialized
if 'cam_manager' not in st.session_state or 'feeds_meta' not in st.session_state:
    st.warning('Please configure cameras first.')
    st.stop()

# Set up metrics placeholders
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
fps_placeholder = metrics_col1.empty()
detections_placeholder = metrics_col2.empty()
e2e_time_placeholder = metrics_col3.empty()

# Detailed timing breakdown
timing_expander = st.expander('📊 Detailed Performance Metrics', expanded=False)
with timing_expander:
    timing_col1, timing_col2, timing_col3 = st.columns(3)
    preprocess_placeholder = timing_col1.empty()
    inference_placeholder = timing_col2.empty()
    postprocess_placeholder = timing_col3.empty()

start_col, stop_col = st.columns(2)
with start_col:
    start_stream = st.button('🚀 Start Processing', key='start_monitor')
with stop_col:
    stop_stream = st.button('🛑 Stop', key='stop_monitor')

# Feed health diagnostics
health_exp = st.expander('🩺 Feed Health / Diagnostics', expanded=False)
with health_exp:
    snap = st.session_state.cam_manager.health_snapshot()
    if not snap:
        st.write('No feeds added.')
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

if start_stream:
    st.info('Running multi-feed processing loop')
    # Auto-start any feeds that are not running yet
    for feed_state in st.session_state.cam_manager.list_feeds():
        if feed_state.status in ('stopped', 'disconnected'):
            try:
                st.session_state.cam_manager.start(feed_state.config.id)
            except Exception as autostart_err:
                st.warning(f"Auto-start failed for {feed_state.config.id}: {autostart_err}")

    frame_count = 0
    last_time = time.time()
    max_frames = 1000
    if 'ema_fps' not in st.session_state:
        st.session_state.ema_fps = 0.0
    alpha = 0.1

    while not stop_stream:
        t0 = time.perf_counter()
        if frame_count > max_frames:
            frame_count = 0
            last_time = time.time()

        feed_frames = []
        view_names = []
        feeds_list = st.session_state.cam_manager.list_feeds()
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
                        imgsz=meta.get('imgsz', 640),
                        conf=meta.get('confidence', 0.25),
                        device=meta.get('device', 'cpu'),
                        half=meta.get('half', False),
                        task=meta.get('primary_task', 'detect')
                    )
                    st.success(f"Model loaded for {meta.get('name', fid)}")
                except Exception as e:
                    st.error(f"Primary model load failed ({fid}): {e}")
            # Lazy load secondary if enabled
            if meta.get('secondary_enabled') and meta.get('secondary_inference') is None and meta.get('secondary_model_path'):
                try:
                    meta['secondary_inference'] = Inference(
                        model_path=meta['secondary_model_path'],
                        imgsz=meta.get('imgsz', 640),
                        conf=meta.get('confidence', 0.25),
                        device=meta.get('device', 'cpu'),
                        half=meta.get('half', False),
                        task=meta.get('secondary_task', 'detect')
                    )
                except Exception as e:
                    st.error(f"Secondary model load failed ({fid}): {e}")

        t1 = time.perf_counter()
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

        processable = True if per_feed_outputs else False

        if waiting_fids or ended_fids:
            bucket = int(time.time())
            last_bucket = st.session_state.get('status_msg_bucket')
            if last_bucket != bucket:
                st.session_state['status_msg_bucket'] = bucket
                if waiting_fids:
                    st.info(f"Waiting for feeds: {', '.join(waiting_fids)}")
                if ended_fids:
                    st.success(f"Completed file feeds: {', '.join(ended_fids)}")
        if not processable and not waiting_fids:
            if not feed_frames:
                st.warning("⚠️ No active camera feeds available. Check that cameras are started.")

        if processable:
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
                annotated_images.append((fid, prim_ann[0] if prim_ann else None, prim_sum[0]))

            t3 = time.perf_counter()
            preprocess_ms = (t1 - t0) * 1000
            inference_ms = sum([
                meta['primary_inference'].get_last_inference_time_ms() if meta.get('primary_inference') else 0
                for fid, meta in st.session_state.feeds_meta.items()
            ])
            postprocess_ms = (t3 - t2) * 1000
            e2e_ms = (t3 - t0) * 1000

            instant_fps = 1000.0 / e2e_ms if e2e_ms > 0 else 0
            if st.session_state.ema_fps == 0.0:
                st.session_state.ema_fps = instant_fps
            else:
                st.session_state.ema_fps = alpha * instant_fps + (1 - alpha) * st.session_state.ema_fps

            # Display each feed
            for fid, ann_img, summary in annotated_images:
                st.subheader(f'Feed: {fid}')
                cols_disp = st.columns(2)
                feed_state = next((fs for fs in st.session_state.cam_manager.list_feeds() if fs.config.id == fid), None)
                if feed_state and feed_state.last_frame is not None:
                    cols_disp[0].image(feed_state.last_frame, channels="BGR", use_container_width=True, caption="Input")
                if ann_img is not None:
                    det_count = summary.get('total_detections', 0) if isinstance(summary, dict) else 0
                    cols_disp[1].image(ann_img, channels="BGR", use_container_width=True, caption=f"Detections: {det_count}")

            total_detections = sum([s.get('total_detections',0) for _,_,s in annotated_images if s])
            fps_placeholder.metric("🚀 FPS (EMA)", f"{st.session_state.ema_fps:.1f}")
            detections_placeholder.metric("🎯 Objects Detected", total_detections)
            e2e_time_placeholder.metric("⏱️ E2E Latency (ms)", f"{e2e_ms:.0f}")
            preprocess_placeholder.metric("� Preprocess", f"{preprocess_ms:.1f}ms")
            inference_placeholder.metric("🧠 Inference", f"{inference_ms:.1f}ms")
            postprocess_placeholder.metric("🎨 Postprocess", f"{postprocess_ms:.1f}ms")
        else:
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

        stop_stream = st.session_state.get("stop_monitor", False)
        time.sleep(0.03)
    st.success("✅ Stream stopped.")
elif not start_stream:
    st.info("👆 Add a camera from the configuration step, then click 'Start Processing' to begin object detection.")

st.markdown('---')
st.info('Insights dashboard will be shown here. Switch to dedicated Insights tab for detailed analytics.')

if st.button('Back to Configuration'):
    st.session_state['step'] = 4
