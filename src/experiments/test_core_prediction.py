import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import tensorflow as tf
from preprocessing import preprocess_character_image, to_model_input

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "gujarati_core_cnn.keras"
LABELS_PATH = ROOT / "models" / "gujarati_core_labels.json"

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)

print(f"Loaded Core Model with {model.output_shape[-1]} classes.\n")

# Test images from box_0 (which corresponds to 'અ')
p = ROOT / "data" / "characters" / "source" / "preprocessed_images" / "box_0"
for img_path in sorted(p.glob("*.png"))[:5]:
    processed = preprocess_character_image(img_path)
    inp = to_model_input(processed)
    probs = model.predict(inp, verbose=0)[0]
    top3 = np.argsort(probs)[-3:][::-1]

    print(f"Image: {img_path.name}")
    for rank, idx in enumerate(top3, 1):
        info = labels.get(str(idx), {})
        char = info.get("character", "Unknown")
        box = info.get("box_key", "")
        print(f"  #{rank}: {char} ({box}) — {probs[idx] * 100:.2f}%")
    print()
