import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "gujarati_character_cnn.keras"
LABELS_PATH = ROOT / "models" / "gujarati_labels.json"
IMAGE_DIR = (
    ROOT / "data" / "characters" / "source"
    / "preprocessed_images" / "box_0"
)

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)

images = sorted(IMAGE_DIR.glob("*.png"))

if not images:
    raise FileNotFoundError(f"No PNG images found in {IMAGE_DIR}")

print("Model output classes:", model.output_shape[-1])
print("Images found in box_0:", len(images))
print("Expected label for box_0:", labels.get("box_0"))

for image_path in images[:3]:
    with Image.open(image_path) as im:
        image = im.convert("L").resize((64, 64))
        array = np.asarray(image, dtype=np.float32) / 255.0

    array = array[np.newaxis, ..., np.newaxis]
    probabilities = model.predict(array, verbose=0)[0]

    top_indices = np.argsort(probabilities)[-5:][::-1]

    print(f"\nImage: {image_path.name}")
    print("Top 5 predictions:")

    for index in top_indices:
        folder_key = f"box_{int(index)}"
        print(
            f"  index={index}, "
            f"label={labels.get(folder_key, 'UNKNOWN')}, "
            f"probability={probabilities[index] * 100:.4f}%"
        )
