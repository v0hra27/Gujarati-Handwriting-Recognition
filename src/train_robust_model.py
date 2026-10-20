import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, models

from preprocessing import preprocess_character_image

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "characters" / "source" / "preprocessed_images"
FONTS_DIR = ROOT / "data" / "fonts"
MODELS_DIR = ROOT / "models"
MODEL_SAVE_PATH = MODELS_DIR / "gujarati_core_cnn.keras"
LABELS_PATH = MODELS_DIR / "gujarati_core_labels.json"

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    core_labels = json.load(f)

# Collect all available Gujarati font paths
font_candidates = [
    Path("C:/Windows/Fonts/NIRMALA.TTF"),
    Path("C:/Windows/Fonts/NIRMALAB.TTF"),
    FONTS_DIR / "NotoSansGujarati.ttf",
    FONTS_DIR / "Rasa.ttf",
    FONTS_DIR / "AnekGujarati.ttf",
]
available_fonts = [str(p) for p in font_candidates if p.exists()]
print(f"Loaded {len(available_fonts)} distinct typography font families.")


def render_font_character(char_str, font_path, font_size=46, target_size=(64, 64)):
    """Renders a character using a TrueType font on a clean canvas."""
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    canvas = Image.new("L", (128, 128), 255)
    draw = ImageDraw.Draw(canvas)
    
    # Get bounding box of text
    bbox = draw.textbbox((0, 0), char_str, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    x = (128 - text_w) // 2 - bbox[0]
    y = (128 - text_h) // 2 - bbox[1]
    draw.text((x, y), char_str, font=font, fill=0)

    # Preprocess with auto-centering
    return preprocess_character_image(canvas, target_size=target_size)


def augment_handwriting(pil_img, rng):
    """Applies realistic handwriting variations: slant, pen thickness, rotation."""
    img_np = np.array(pil_img)

    # 1. Stroke thickness variation: thin pen (erosion) vs thick marker (dilation)
    stroke_mode = rng.choice(["thin", "thick", "normal"], p=[0.25, 0.25, 0.5])
    if stroke_mode == "thick":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # Note: in thresholded image, ink is dark (< 128), so erode darkens/thickens the ink
        img_np = cv2.erode(img_np, kernel, iterations=1)
    elif stroke_mode == "thin":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img_np = cv2.dilate(img_np, kernel, iterations=1)

    img = Image.fromarray(img_np)

    # 2. Random handwriting rotation (-14 to +14 deg)
    angle = rng.uniform(-14, 14)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=255)

    # 3. Random slant shear (-12 to +12 deg)
    shear = rng.uniform(-0.20, 0.20)
    w, h = img.size
    m = [1, shear, -shear * h / 2, 0, 1, 0]
    img = img.transform((w, h), Image.Transform.AFFINE, m, resample=Image.Resampling.BILINEAR, fillcolor=255)

    # 4. Random slight translation (-3 to +3 px)
    dx = int(rng.uniform(-3, 4))
    dy = int(rng.uniform(-3, 4))
    canvas = Image.new("L", (w, h), 255)
    canvas.paste(img, (dx, dy))

    return canvas


print("Generating multi-writer dataset with font rendering and real handwriting...")
X_train_list, y_train_list = [], []
X_val_list, y_val_list = [], []

rng = np.random.default_rng(42)

for class_id_str, info in core_labels.items():
    class_id = int(class_id_str)
    char = info["character"]
    box_key = info["box_key"]

    # 1. Genuine handwriting samples from dataset
    real_folder = DATA_DIR / box_key
    real_images = []
    if real_folder.exists():
        for p in sorted(real_folder.glob("*.png")):
            try:
                real_images.append(preprocess_character_image(p))
            except Exception:
                pass

    if real_images:
        # Split genuine handwriting: last 2 for validation, rest for training
        train_reals = real_images[:-2] if len(real_images) > 3 else real_images[:-1]
        val_reals = real_images[-2:] if len(real_images) > 3 else real_images[-1:]

        for val_img in val_reals:
            arr = np.asarray(val_img, dtype=np.float32) / 255.0
            X_val_list.append(arr[..., np.newaxis])
            y_val_list.append(class_id)

        # Augment genuine handwriting samples heavily (15x each)
        for r_img in train_reals:
            arr = np.asarray(r_img, dtype=np.float32) / 255.0
            X_train_list.append(arr[..., np.newaxis])
            y_train_list.append(class_id)

            for _ in range(15):
                aug = augment_handwriting(r_img, rng)
                arr = np.asarray(aug, dtype=np.float32) / 255.0
                X_train_list.append(arr[..., np.newaxis])
                y_train_list.append(class_id)

    # 2. Render each font across 3 sizes and generate 25 handwriting variations per font
    for font_path in available_fonts:
        for fsize in [40, 48, 54]:
            base_font_img = render_font_character(char, font_path, font_size=fsize)
            arr = np.asarray(base_font_img, dtype=np.float32) / 255.0
            X_train_list.append(arr[..., np.newaxis])
            y_train_list.append(class_id)

            for _ in range(8):
                aug_font = augment_handwriting(base_font_img, rng)
                arr = np.asarray(aug_font, dtype=np.float32) / 255.0
                X_train_list.append(arr[..., np.newaxis])
                y_train_list.append(class_id)

X_train = np.array(X_train_list, dtype=np.float32)
y_train = np.array(y_train_list, dtype=np.int32)
X_val = np.array(X_val_list, dtype=np.float32)
y_val = np.array(y_val_list, dtype=np.int32)

print("\n--- Dataset Generation Complete ---")
print(f"Total Classes: {len(core_labels)}")
print(f"Training Samples: {len(X_train)} (shape: {X_train.shape})")
print(f"Validation Samples: {len(X_val)} (shape: {X_val.shape})")

# Shuffle training set
shuffle_idx = rng.permutation(len(X_train))
X_train = X_train[shuffle_idx]
y_train = y_train[shuffle_idx]


# Deep, robust CNN architecture
num_classes = len(core_labels)
model = models.Sequential([
    layers.Input(shape=(64, 64, 1)),

    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

cb_list = [
    callbacks.ModelCheckpoint(
        filepath=str(MODEL_SAVE_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
]

print("\nStarting Training of Generalization-Ready CNN...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=64,
    callbacks=cb_list,
    verbose=1
)

final_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
val_loss, val_acc = final_model.evaluate(X_val, y_val, verbose=0)
train_loss, train_acc = final_model.evaluate(X_train, y_train, verbose=0)

print("\n========================================================")
print(f"🎉 Training Finished! Model Saved to: {MODEL_SAVE_PATH}")
print(f"Train Accuracy: {train_acc * 100:.2f}%")
print(f"Validation Accuracy on Real Handwriting: {val_acc * 100:.2f}%")
print("========================================================")
