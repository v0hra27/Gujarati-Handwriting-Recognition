import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Set reproducible seeds
tf.keras.utils.set_random_seed(42)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "characters" / "core_dataset_47.npz"
MODEL_PATH = ROOT / "models" / "gujarati_core_cnn.keras"
METRICS_PATH = ROOT / "models" / "core_training_results.json"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset archive not found: {DATA_PATH}")

print("Loading dataset from:", DATA_PATH)
data = np.load(DATA_PATH)
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]

num_classes = len(np.unique(np.concatenate([y_train, y_val])))
print(f"Loaded: {len(X_train)} train, {len(X_val)} val samples. Classes: {num_classes}")

# Build CNN Architecture with Batch Normalization & Dropout
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
    layers.Dropout(0.3),

    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Flatten(),
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
        filepath=str(MODEL_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-5,
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
]

print("\nStarting CNN Training on Core 47 Characters...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=35,
    batch_size=32,
    callbacks=cb_list,
    verbose=1
)

# Evaluate best model
final_model = tf.keras.models.load_model(MODEL_PATH)
val_loss, val_acc = final_model.evaluate(X_val, y_val, verbose=0)
train_loss, train_acc = final_model.evaluate(X_train, y_train, verbose=0)

results = {
    "classes": int(num_classes),
    "train_samples": int(len(X_train)),
    "val_samples": int(len(X_val)),
    "best_val_accuracy": float(val_acc),
    "best_val_loss": float(val_loss),
    "train_accuracy": float(train_acc),
    "epochs_trained": len(history.history["loss"])
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n==========================================")
print(f"Training Complete! Saved Model: {MODEL_PATH}")
print(f"Train Accuracy: {train_acc * 100:.2f}%")
print(f"Validation Accuracy: {val_acc * 100:.2f}%")
print("==========================================")
