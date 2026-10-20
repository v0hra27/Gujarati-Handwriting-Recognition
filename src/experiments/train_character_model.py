import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = Path("data/characters/source/preprocessed_images")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 20

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

class_dirs = sorted(
    [p for p in DATA_DIR.glob("box_*") if p.is_dir()],
    key=lambda p: int(p.name.split("_")[1])
)

class_names = [p.name for p in class_dirs]
class_to_id = {name: i for i, name in enumerate(class_names)}

images, labels = [], []

print("Loading images...")
for folder in class_dirs:
    label = class_to_id[folder.name]
    for path in sorted(folder.glob("*")):
        if not path.is_file():
            continue
        try:
            with Image.open(path) as img:
                img = img.convert("L").resize(IMAGE_SIZE)
                array = np.asarray(img, dtype=np.float32) / 255.0
            images.append(array[..., np.newaxis])
            labels.append(label)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")

X = np.asarray(images, dtype=np.float32)
y = np.asarray(labels, dtype=np.int32)

if len(X) == 0:
    raise ValueError("No usable images were found.")
if len(np.unique(y)) < 2:
    raise ValueError("At least two classes are required.")

print("Classes:", len(class_names))
print("Images:", len(X))
print("Image tensor:", X.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

print("Training images:", len(X_train))
print("Validation images:", len(X_val))

# Mild augmentation only; large rotations can change Gujarati characters.
augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.04),
    layers.RandomTranslation(0.04, 0.04),
    layers.RandomZoom(0.08),
], name="augmentation")

model = models.Sequential([
    layers.Input(shape=(64, 64, 1)),
    augmentation,
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(len(class_names), activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

best_path = MODEL_DIR / "gujarati_character_cnn.keras"

callbacks = [
    EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    ),
    ModelCheckpoint(
        str(best_path), monitor="val_loss", save_best_only=True
    ),
]

print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

model.save(best_path)

with open(MODEL_DIR / "character_labels.json", "w", encoding="utf-8") as f:
    json.dump(class_to_id, f, ensure_ascii=False, indent=2)

with open(MODEL_DIR / "training_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "classes": len(class_names),
        "images": len(X),
        "train_images": len(X_train),
        "validation_images": len(X_val),
        "best_validation_accuracy": float(max(history.history["val_accuracy"])),
        "best_validation_loss": float(min(history.history["val_loss"])),
    }, f, indent=2)

print("\nTraining finished.")
print("Model saved to:", best_path)
print("Class IDs saved to: models/character_labels.json")
print("Results saved to: models/training_results.json")
print("Note: labels are currently folder IDs, not Gujarati text.")
