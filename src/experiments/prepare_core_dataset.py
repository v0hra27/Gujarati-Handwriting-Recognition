import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_character_image

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "characters" / "source" / "preprocessed_images"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# 12 vowels + 35 base consonants = 47 classes
CORE_BOX_IDS = list(range(12)) + [12 + 12 * i for i in range(35)]

# Load the full 432 label dictionary
with open(MODELS_DIR / "gujarati_labels.json", "r", encoding="utf-8") as f:
    full_labels = json.load(f)

# Build the 47-class mapping: class_id (0 to 46) -> character string
core_labels = {}
box_to_class_id = {}

for new_id, box_idx in enumerate(CORE_BOX_IDS):
    box_key = f"box_{box_idx}"
    char_symbol = full_labels.get(box_key, f"box_{box_idx}")
    core_labels[str(new_id)] = {
        "box_key": box_key,
        "character": char_symbol
    }
    box_to_class_id[box_key] = new_id

# Save the 47 core labels
core_labels_path = MODELS_DIR / "gujarati_core_labels.json"
with open(core_labels_path, "w", encoding="utf-8") as f:
    json.dump(core_labels, f, ensure_ascii=False, indent=2)

print(f"Saved 47 core labels to: {core_labels_path}")


def augment_image(pil_img, seed=None):
    """
    Applies small, realistic handwriting variations:
    - Slight rotation (-6 to +6 degrees)
    - Small translation (-2 to +2 pixels)
    - Slight contrast variation
    """
    rng = np.random.default_rng(seed)

    # 1. Random rotation
    angle = rng.uniform(-6, 6)
    img = pil_img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=255)

    # 2. Random translation
    dx = int(rng.uniform(-2, 3))
    dy = int(rng.uniform(-2, 3))
    canvas = Image.new("L", pil_img.size, 255)
    canvas.paste(img, (dx, dy))
    img = canvas

    # 3. Slight contrast / stroke enhancement
    contrast_factor = rng.uniform(0.9, 1.2)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)

    return img


def build_dataset(augment_factor=10):
    raw_images = []
    raw_labels = []

    print("Loading and preprocessing source character images...")
    for class_id, box_idx in enumerate(CORE_BOX_IDS):
        folder = DATA_DIR / f"box_{box_idx}"
        if not folder.exists():
            print(f"Warning: {folder} does not exist.")
            continue

        for img_path in sorted(folder.glob("*.png")):
            try:
                processed = preprocess_character_image(img_path, target_size=(64, 64))
                raw_images.append(processed)
                raw_labels.append(class_id)
            except Exception as e:
                print(f"Error reading {img_path}: {e}")

    raw_labels = np.array(raw_labels, dtype=np.int32)
    print(f"Found {len(raw_images)} original preprocessed images across {len(CORE_BOX_IDS)} classes.")

    # Stratified Train/Val split: 80% train, 20% validation on raw images
    indices = np.arange(len(raw_images))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=raw_labels
    )

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []

    # 1. Validation set (only original un-augmented preprocessed images)
    for idx in val_idx:
        arr = np.asarray(raw_images[idx], dtype=np.float32) / 255.0
        X_val_list.append(arr[..., np.newaxis])
        y_val_list.append(raw_labels[idx])

    # 2. Training set (original + augmented variations)
    for idx in train_idx:
        base_img = raw_images[idx]
        base_label = raw_labels[idx]

        # Original
        arr = np.asarray(base_img, dtype=np.float32) / 255.0
        X_train_list.append(arr[..., np.newaxis])
        y_train_list.append(base_label)

        # Augmented variations
        for k in range(augment_factor):
            aug_img = augment_image(base_img, seed=idx * 100 + k)
            arr = np.asarray(aug_img, dtype=np.float32) / 255.0
            X_train_list.append(arr[..., np.newaxis])
            y_train_list.append(base_label)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    X_val = np.array(X_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.int32)

    # Save prepared dataset
    out_path = ROOT / "data" / "characters" / "core_dataset_47.npz"
    np.savez_compressed(
        out_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val
    )

    print("\n--- Chunk 2 Dataset Summary ---")
    print(f"Classes: {len(CORE_BOX_IDS)}")
    print(f"Training samples: {len(X_train)} (shape: {X_train.shape})")
    print(f"Validation samples: {len(X_val)} (shape: {X_val.shape})")
    print(f"Saved dataset archive to: {out_path.resolve()}")


if __name__ == "__main__":
    build_dataset(augment_factor=10)
