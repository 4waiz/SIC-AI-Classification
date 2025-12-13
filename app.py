import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import io
import json
import threading
import time
from typing import Optional, Dict, Any

import av
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from streamlit_webrtc import webrtc_streamer


@st.cache_resource
def load_model():
    import tensorflow as tf

    return tf.keras.models.load_model("final_model.h5", compile=False)


def load_labels():
    path = os.path.join(os.path.dirname(__file__), "labels.txt")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return [x.strip() for x in f if x.strip()]
    return None


def preprocess(img: Image.Image, size):
    img = img.convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_image(img: Image.Image, model, labels) -> Optional[Dict[str, Any]]:
    """Return prediction dict with label and confidence or None when model is missing."""
    if model is None:
        return None
    x = preprocess(img, (224, 224))
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(np.max(probs))
    label = labels[idx] if labels and idx < len(labels) else f"class_{idx}"
    return {"label": label, "confidence": conf, "index": idx, "probs": probs}


def speak(text, rate=1.0, lang="en-US"):
    components.html(
        f"""
        <script>
        const t = {json.dumps(text)};
        const u = new SpeechSynthesisUtterance(t);
        u.lang = {json.dumps(lang)};
        u.rate = {rate};
        speechSynthesis.cancel();
        speechSynthesis.speak(u);
        </script>
        """,
        height=0,
    )


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
        :root {
            --bg-1: #050912;
            --bg-2: #0b1733;
            --card: rgba(255,255,255,0.04);
            --card-strong: rgba(255,255,255,0.08);
            --accent: #6ae3ff;
            --accent-2: #9b8bff;
            --text: #e8ecf3;
            --muted: #95a3b8;
        }
        .stApp {
            background: radial-gradient(120% 120% at 20% 20%, rgba(106, 227, 255, 0.18), transparent),
                        radial-gradient(120% 120% at 80% 0%, rgba(155, 139, 255, 0.22), transparent),
                        linear-gradient(120deg, var(--bg-1), var(--bg-2));
            color: var(--text);
            font-family: 'Space Grotesk', 'Segoe UI', system-ui, -apple-system, sans-serif;
        }
        h1, h2, h3, h4 {
            font-weight: 700;
            letter-spacing: 0.01em;
        }
        .hero {
            padding: 1.25rem 1.5rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(106, 227, 255, 0.16), rgba(155, 139, 255, 0.10));
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 20px 60px rgba(0,0,0,0.35);
        }
        .hero .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            color: var(--text);
            font-size: 0.85rem;
            letter-spacing: 0.02em;
        }
        .hero-title {
            font-size: 2.2rem;
            margin: 0.35rem 0 0.1rem;
        }
        .hero-subtitle {
            color: var(--muted);
            max-width: 720px;
            line-height: 1.4;
        }
        .neo-card {
            padding: 1rem 1.25rem;
            margin-top: 0.75rem;
            border-radius: 14px;
            background: var(--card);
            border: 1px solid rgba(255,255,255,0.06);
            box-shadow: 0 12px 30px rgba(0,0,0,0.25);
        }
        .neo-card.live {
            border-color: var(--accent);
            box-shadow: 0 16px 40px rgba(106,227,255,0.25);
        }
        .neo-card .label {
            font-size: 1.3rem;
            font-weight: 700;
        }
        .neo-card .meta {
            color: var(--muted);
            font-size: 0.95rem;
            margin-top: 0.2rem;
        }
        .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.6rem;
            border-radius: 10px;
            background: var(--card);
            font-size: 0.9rem;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #2ae077;
            box-shadow: 0 0 0 6px rgba(42, 224, 119, 0.12);
        }
        [data-testid="stTabs"] button {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.07);
            color: var(--text);
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            background: linear-gradient(135deg, rgba(106, 227, 255, 0.28), rgba(155, 139, 255, 0.24));
            border-color: rgba(255,255,255,0.16);
        }
        .stCheckbox>div>label, .stFileUploader label, .stCameraInput label {
            font-weight: 600;
        }
        .footer-muted {
            color: var(--muted);
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def prediction_card_html(title: str, label: str, confidence: Optional[float], live: bool = False) -> str:
    conf_text = f"{confidence * 100:.1f}% confidence" if confidence is not None else "Confidence unavailable"
    return f"""
    <div class="neo-card {'live' if live else ''}">
        <div class="meta">{title}</div>
        <div class="label">{label}</div>
        <div class="meta">{conf_text}</div>
    </div>
    """


def render_prediction_card(title: str, label: str, confidence: Optional[float], live: bool = False):
    st.markdown(prediction_card_html(title, label, confidence, live), unsafe_allow_html=True)


def handle_photo(img: Image.Image, source: str, model, labels, audio_enabled: bool):
    st.image(img, width=min(720, img.width), caption=source)
    result = predict_image(img, model, labels)
    if result:
        render_prediction_card("Prediction", result["label"], result["confidence"])
        if audio_enabled and st.session_state.get("_last_static") != result["label"]:
            speak(result["label"])
            st.session_state["_last_static"] = result["label"]
    else:
        st.warning("Model not loaded. Predictions are unavailable.")


def main():
    st.set_page_config(
        page_title="AI Based Safety Navigation System",
        page_icon=":compass:",
        layout="wide",
    )
    inject_styles()
    st.session_state.setdefault("_last_live", None)
    st.session_state.setdefault("_last_static", None)

    st.markdown(
        """
        <div class="hero">
            <div class="pill">AI Safety | Vision Intelligence</div>
            <div class="hero-title">AI Based Safety Navigation System</div>
            <div class="hero-subtitle">
                Real-time hazard awareness with on-device intelligence. Stream live video, capture moments, or upload footage to classify scenes instantly.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading vision model..."):
        try:
            model = load_model()
        except Exception as exc:
            model = None
            st.error(f"Model could not be loaded: {exc}")
    labels = load_labels()

    col_status, col_audio = st.columns([3, 2])
    with col_status:
        st.write("")
        status_color = "#2ae077" if model is not None else "#ff6b6b"
        shadow_color = "rgba(42, 224, 119, 0.12)" if model is not None else "rgba(255, 107, 107, 0.18)"
        status_text = "Ready" if model is not None else "Unavailable"
        st.markdown(
            f'<div class="status-chip"><div class="status-dot" style="background:{status_color}; box-shadow: 0 0 0 6px {shadow_color};"></div><div>Model {status_text}</div></div>',
            unsafe_allow_html=True,
        )
    with col_audio:
        audio_enabled = st.toggle(
            "Audio feedback",
            value=True,
            help="Speak predictions aloud for accessibility.",
        )

    tab_live, tab_camera, tab_upload = st.tabs(["Live Video", "Camera", "Upload"])

    with tab_live:
        st.subheader("Live Video Stream")
        st.caption("Stream from your webcam for real-time classification. Record up to 150 frames to a GIF.")
        record_enabled = st.checkbox("Record to GIF", value=False)
        lock = threading.Lock()
        shared = {"label": None, "confidence": None, "frames": []}
        max_frames = 150

        def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_image()
            result = predict_image(img, model, labels)
            with lock:
                if result:
                    shared["label"] = result["label"]
                    shared["confidence"] = result["confidence"]
                if record_enabled and len(shared["frames"]) < max_frames:
                    shared["frames"].append(img.copy())
            return frame

        ctx = webrtc_streamer(
            key="live-video",
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
        )

        live_card = st.empty()
        if ctx and ctx.state.playing:
            while ctx.state.playing:
                with lock:
                    name_v = shared.get("label")
                    conf_v = shared.get("confidence")
                if name_v:
                    live_card.markdown(
                        prediction_card_html("Live prediction", name_v, conf_v, live=True),
                        unsafe_allow_html=True,
                    )
                    if audio_enabled and st.session_state.get("_last_live") != name_v:
                        speak(name_v)
                        st.session_state["_last_live"] = name_v
                time.sleep(0.1)
        else:
            live_card.markdown(
                '<div class="neo-card">Start streaming to see live predictions.</div>',
                unsafe_allow_html=True,
            )

        if record_enabled and shared["frames"]:
            if st.button("Save GIF"):
                import imageio

                buf = io.BytesIO()
                imageio.mimsave(
                    buf,
                    [f.resize((320, 240)) for f in shared["frames"]],
                    format="GIF",
                    duration=0.1,
                )
                st.download_button(
                    "Download recording.gif",
                    data=buf.getvalue(),
                    file_name="recording.gif",
                    mime="image/gif",
                )

    with tab_camera:
        st.subheader("Camera Capture")
        st.caption("Take a quick snapshot with your webcam for instant analysis.")
        camera_file = st.camera_input("Capture a photo")
        if camera_file is not None:
            img = Image.open(io.BytesIO(camera_file.getvalue()))
            handle_photo(img, "Camera snapshot", model, labels, audio_enabled)

    with tab_upload:
        st.subheader("Upload")
        st.caption("Drop an image to classify. Supported formats: PNG, JPG, JPEG, BMP.")
        file = st.file_uploader(
            "Upload an image", type=["png", "jpg", "jpeg", "bmp"]
        )
        if file is not None:
            img = Image.open(file)
            handle_photo(img, f"Uploaded file: {file.name}", model, labels, audio_enabled)

    st.markdown(
        '<div class="footer-muted">Tip: Keep subjects centered and well lit for higher confidence.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
