import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import tensorflow as tf
from PIL import Image

from preprocessing import preprocess_character_image, to_model_input
from segmentation import segment_word_into_characters

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "gujarati_core_cnn.keras"
LABELS_PATH = ROOT / "models" / "gujarati_core_labels.json"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Create a composite word image from real samples:
# 'box_12' ('ક') + 'box_276' ('મ') + 'box_384' ('ળ') -> "કમળ"
char_dirs = [
    ROOT / "data" / "characters" / "source" / "preprocessed_images" / "box_12",  # ક
    ROOT / "data" / "characters" / "source" / "preprocessed_images" / "box_276", # મ
    ROOT / "data" / "characters" / "source" / "preprocessed_images" / "box_384", # ળ
]

loaded_imgs = [
    Image.open(ROOT / "data" / "characters" / "source" / "preprocessed_images" / "box_12" / "1.png"),  # ક
    Image.open(ROOT / "data" / "characters" / "source" / "preprocessed_images" / "box_276" / "1.png"), # મ
    Image.open(ROOT / "data" / "characters" / "source" / "preprocessed_images" / "box_384" / "2.png"), # ળ (clean complete glyph)
]

if len(loaded_imgs) == 3:
    # Stitch side-by-side into a single word image with spacing
    word_w = sum(im.width for im in loaded_imgs) + 40
    word_h = max(im.height for im in loaded_imgs) + 20
    word_canvas = Image.new("RGB", (word_w, word_h), "white")

    cur_x = 20
    for im in loaded_imgs:
        word_canvas.paste(im, (cur_x, 10))
        cur_x += im.width + 10

    test_word_path = OUTPUT_DIR / "sample_word_kamal.png"
    word_canvas.save(test_word_path)
    print(f"Created sample word image: {test_word_path.resolve()}")

    # 2. Run Segmentation Engine
    crops, boxes, annotated = segment_word_into_characters(word_canvas)
    annotated_path = OUTPUT_DIR / "sample_word_segmented.png"
    annotated.save(annotated_path)
    print(f"Segmented into {len(crops)} character boxes. Saved to {annotated_path.resolve()}")

    # 3. Recognize each segmented character using the Core CNN
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    recognized_letters = []
    print("\n--- Recognition per Segment ---")
    for i, crop in enumerate(crops, 1):
        clean_crop = preprocess_character_image(crop)
        inp = to_model_input(clean_crop)
        probs = model.predict(inp, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        char = labels.get(str(top_idx), {}).get("character", "?")
        conf = probs[top_idx] * 100
        recognized_letters.append(char)
        print(f"  Segment {i}: '{char}' (Confidence: {conf:.2f}%)")

    final_word = "".join(recognized_letters)
    print(f"\nFinal Assembled Word: {final_word}")
    print("Pipeline test successful!")
