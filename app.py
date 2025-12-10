import os
import io
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import streamlit.components.v1 as components

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("final_model.h5", compile=False)

def load_labels():
    p = os.path.join(os.path.dirname(__file__), "labels.txt")
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            return [x.strip() for x in f if x.strip()]
    return None

def preprocess(img, size):
    img = img.convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

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

def main():
    st.set_page_config(page_title="AI Based Safety Navigation System", page_icon="🧭", layout="centered")
    st.title("AI Based Safety Navigation System")
    audio_enabled = st.toggle("Audio feedback", value=True)
    try:
        model = load_model()
    except Exception:
        model = None
    labels = load_labels()
    tab1, tab2 = st.tabs(["Upload", "Camera"])
    with tab1:
        file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "bmp"])
    with tab2:
        camera_file = st.camera_input("Take a photo")
    img = None
    if file is not None:
        img = Image.open(file)
    elif camera_file is not None:
        img = Image.open(io.BytesIO(camera_file.getvalue()))
    if img is not None:
        st.image(img, width=min(700, img.width))
        if model is not None:
            x = preprocess(img, (224, 224))
            probs = model.predict(x)[0]
            idx = int(np.argmax(probs))
            conf = float(np.max(probs))
            name = labels[idx] if labels and idx < len(labels) else f"class_{idx}"
            st.subheader("Prediction")
            st.write(f"{name} ({conf:.4f})")
            if audio_enabled:
                key = f"{name}-{int(conf*100)}"
                if st.session_state.get("_last_pred") != key:
                    speak(f"{name}. {int(conf*100)} percent")
                    st.session_state["_last_pred"] = key

if __name__ == "__main__":
    main()
