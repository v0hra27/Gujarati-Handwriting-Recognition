import json
import sys
from pathlib import Path

# Ensure UTF-8 console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import importlib
import src.preprocessing
import src.segmentation

importlib.reload(src.preprocessing)
importlib.reload(src.segmentation)
from src.preprocessing import preprocess_character_image, to_model_input
from src.segmentation import segment_word_into_characters

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "gujarati_core_cnn.keras"
LABELS_PATH = MODELS_DIR / "gujarati_core_labels.json"

st.set_page_config(
    page_title="Gujarati Handwritten Text Recognition",
    page_icon="✍️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }
    .char-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    .char-display {
        font-size: 5rem;
        font-weight: 800;
        line-height: 1.1;
        color: #f8fafc;
    }
    .word-card {
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .word-display {
        font-size: 3.5rem;
        font-weight: 800;
        color: #60a5fa;
        letter-spacing: 2px;
    }
    .conf-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 0.6rem;
    }
    .conf-high {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    .conf-med {
        background-color: rgba(234, 179, 8, 0.2);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.4);
    }
    .conf-low {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Model Status")
    st.success("✅ **Core Gujarati CNN Active**\n- 47 Classes (12 Vowels + 35 Base Consonants)\n- Trained with Multi-Font Generalization\n- **~83% Real-World Handwriting Accuracy**")

    st.divider()
    st.markdown("### 👨‍💻 Project Details")
    st.markdown("""
    **Student:** Mohammadkaif Vahora  
    **ID:** `23IT475` | **Division:** `D10`  
    **Course:** Machine Learning (`4IT47`) — Sem 7
    """)

    st.divider()
    st.markdown("### 🛠️ Preprocessing")
    enable_auto_crop = st.checkbox("Auto-crop & center characters", value=True, help="Automatically centers the character and eliminates edge noise.")
    min_contour_area = st.slider("Min character area (noise threshold)", min_value=15, max_value=150, value=40, step=5)
    merge_gap = st.slider("Stroke merge distance (px)", min_value=8, max_value=32, value=20, step=2, help="Merges split strokes belonging to the same letter (e.g. the loop and bar of 'ગ').")
    
    if st.button("🔄 Reload Model & Reset Cache", use_container_width=True):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()


@st.cache_resource
def load_tf_model(path_str, mtime):
    return tf.keras.models.load_model(path_str)


@st.cache_data
def load_label_map(path_str, mtime):
    with open(path_str, "r", encoding="utf-8") as f:
        return json.load(f)


if not MODEL_PATH.exists():
    st.error(f"Model file not found: `{MODEL_PATH}`")
    st.stop()

if not LABELS_PATH.exists():
    st.error(f"Labels file not found: `{LABELS_PATH}`")
    st.stop()

model = load_tf_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
label_data = load_label_map(str(LABELS_PATH), LABELS_PATH.stat().st_mtime)


def get_char_info(idx):
    info = label_data.get(str(idx), {})
    return info.get("character", "?"), info.get("box_key", f"box_{idx}")


def predict_single_crop(crop_pil):
    if enable_auto_crop:
        clean = preprocess_character_image(crop_pil, target_size=(64, 64))
    else:
        clean = ImageOps.grayscale(crop_pil).resize((64, 64))

    inp = to_model_input(clean)
    probs = model.predict(inp, verbose=0)[0]
    best_idx = int(np.argmax(probs))
    char, box = get_char_info(best_idx)
    return char, box, probs, clean


# Main App UI
st.markdown('<div class="main-title">✍️ Gujarati Handwritten Text Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A complete Machine Learning system for recognizing handwritten Gujarati characters and full words into editable digital text.</div>', unsafe_allow_html=True)

tab_char, tab_word = st.tabs(["🔤 Single Character Recognition", "📝 Word & Text Recognition"])

# ----------------- TAB 1: SINGLE CHARACTER -----------------
with tab_char:
    col_u, col_r = st.columns([1, 1], gap="large")

    with col_u:
        st.subheader("1. Select or Upload Character")
        
        char_choices = ["None (Upload your own)"]
        char_box_map = {}
        data_chars_dir = BASE_DIR / "data" / "characters" / "source" / "preprocessed_images"
        
        for _, info in sorted(label_data.items(), key=lambda x: int(x[0])):
            c_symbol = info.get("character", "?")
            b_key = info.get("box_key", "")
            lbl = f"{c_symbol}  ({b_key})"
            char_choices.append(lbl)
            char_box_map[lbl] = b_key

        selected_char_sample = st.selectbox(
            "🖼️ Select any of the 47 Gujarati characters to test:",
            options=char_choices,
            index=1,
            key="sample_char_selector"
        )

        uploaded_char = st.file_uploader(
            "Or upload a custom character image",
            type=["png", "jpg", "jpeg", "bmp"],
            key="char_uploader"
        )

        raw_img = None
        if uploaded_char is not None:
            raw_img = Image.open(uploaded_char)
        elif selected_char_sample != "None (Upload your own)" and selected_char_sample in char_box_map:
            b_folder = data_chars_dir / char_box_map[selected_char_sample]
            if b_folder.exists():
                c_files = sorted(b_folder.glob("*.png"))
                if c_files:
                    raw_img = Image.open(c_files[0])

        if raw_img is not None:
            p1, p2 = st.columns(2)
            with p1:
                st.caption("Selected / Uploaded Character")
                st.image(raw_img, use_container_width=True)

            char, box, probs, clean_preview = predict_single_crop(raw_img)

            with p2:
                st.caption("Model Input (64×64 Centered)")
                st.image(clean_preview, use_container_width=True)

    with col_r:
        st.subheader("2. Recognition Output")
        if raw_img is None:
            st.info("👈 Please select a character from the dropdown or upload an image on the left.")
        else:
            best_idx = int(np.argmax(probs))
            conf = probs[best_idx] * 100

            if conf >= 70:
                badge = ("conf-high", "🟢 High Confidence")
            elif conf >= 30:
                badge = ("conf-med", "🟡 Moderate Confidence")
            else:
                badge = ("conf-low", "🔴 Low Confidence")

            st.markdown(f"""
            <div class="char-card">
                <div class="char-display">{char}</div>
                <div style="font-size: 0.95rem; color: #94a3b8; margin-top: 0.3rem;">Class: {box}</div>
                <div class="conf-badge {badge[0]}">
                    {badge[1]} • {conf:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.markdown("#### Top 5 Alternative Predictions")
            top_indices = np.argsort(probs)[-5:][::-1]
            for rank, idx in enumerate(top_indices, 1):
                idx = int(idx)
                c, b = get_char_info(idx)
                p = float(probs[idx])
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.write(f"**#{rank} {c}**")
                with c2:
                    st.progress(p, text=f"{p * 100:.2f}%")


# ----------------- TAB 2: WORD RECOGNITION -----------------
with tab_word:
    col_wu, col_wr = st.columns([1, 1], gap="large")

    with col_wu:
        st.subheader("1. Upload Handwritten Word")
        st.caption("Upload an image containing a handwritten Gujarati word or sequence of characters.")

        samples_dir = BASE_DIR / "data" / "sample_images"
        sample_options = ["None (Upload your own)"]
        sample_map = {}
        
        sample_display_names = {
            "kamal.png": "કમળ (kamal.png)",
            "saral.png": "સરળ (saral.png)",
            "kalam.png": "કલમ (kalam.png)",
            "vatan.png": "વતન (vatan.png)",
            "bharat.png": "ભરત (bharat.png)",
            "gagan.png": "ગગન (gagan.png)",
            "raman.png": "રમણ (raman.png)",
            "nagar.png": "નગર (nagar.png)",
            "janak.png": "જનક (janak.png)",
            "madad.png": "મદદ (madad.png)",
            "das.png": "દસ (das.png)",
            "pag.png": "પગ (pag.png)",
            "taras.png": "તરસ (taras.png)",
            "pavan.png": "પવન (pavan.png)",
            "nayan.png": "નયન (nayan.png)"
        }

        if samples_dir.exists():
            for p in sorted(samples_dir.glob("*.png")):
                label = sample_display_names.get(p.name, p.name)
                sample_options.append(label)
                sample_map[label] = p

        selected_sample = st.selectbox(
            "🖼️ Or select a ready-to-test sample word:",
            options=sample_options,
            index=1 if len(sample_options) > 1 else 0
        )

        uploaded_word = st.file_uploader(
            "Upload custom word image (PNG, JPG, BMP)",
            type=["png", "jpg", "jpeg", "bmp"],
            key="word_uploader"
        )

        word_pil = None
        input_id = "default"
        if uploaded_word is not None:
            word_pil = Image.open(uploaded_word)
            input_id = f"uploaded_{uploaded_word.name}_{uploaded_word.size}"
        elif selected_sample != "None (Upload your own)" and selected_sample in sample_map:
            word_pil = Image.open(sample_map[selected_sample])
            input_id = f"sample_{selected_sample}"

        if word_pil is not None:
            st.caption("Uploaded Word Image")
            st.image(word_pil, use_container_width=True)

    with col_wr:
        st.subheader("2. Segmentation & Digital Text")

        if word_pil is None:
            st.info("👈 Upload a handwritten word image or select a sample from the left panel.")
        else:
            with st.spinner("Segmenting characters and predicting..."):
                crops, boxes, annotated = segment_word_into_characters(
                    word_pil, min_area=min_contour_area, max_gap_merge=merge_gap
                )

            st.caption("Detected Character Segments (Left to Right)")
            st.image(annotated, use_container_width=True)

            word_predictions = []
            segment_details = []

            for i, crop in enumerate(crops, 1):
                char, box, probs, clean = predict_single_crop(crop)
                conf = float(np.max(probs)) * 100
                word_predictions.append(char)
                segment_details.append((i, char, conf, clean))

            final_text = "".join(word_predictions)

            st.markdown(f"""
            <div class="word-card">
                <div style="font-size: 1rem; color: #94a3b8; font-weight: 600;">RECOGNIZED DIGITAL GUJARATI TEXT</div>
                <div class="word-display">{final_text}</div>
            </div>
            """, unsafe_allow_html=True)

            st.text_input("📋 Editable Digital Output", value=final_text, key=f"digital_out_{input_id}")

            st.markdown("#### Character-by-Character Breakdown")
            seg_cols = st.columns(min(len(segment_details), 6))
            for idx, (seg_num, c, conf, cln) in enumerate(segment_details):
                col_target = seg_cols[idx % len(seg_cols)]
                with col_target:
                    st.image(cln, use_container_width=True)
                    st.markdown(f"<div style='text-align: center; font-size: 1.4rem; font-weight: bold;'>{c}</div>", unsafe_allow_html=True)
                    st.caption(f"#{seg_num} • {conf:.1f}%")
