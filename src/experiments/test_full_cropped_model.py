from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

tf.keras.utils.set_random_seed(42)

root = Path("data/characters/source/preprocessed_images")
images, labels = [], []

def preprocess(path):
    with Image.open(path) as im:
        gray = ImageOps.grayscale(im)
        arr = np.asarray(gray)
        ys, xs = np.where(arr < 180)

        if len(xs):
            gray = gray.crop((
                int(xs.min()), int(ys.min()),
                int(xs.max()) + 1, int(ys.max()) + 1
            ))

        if gray.width == 0 or gray.height == 0:
            return np.zeros((64, 64), dtype=np.float32)

        side = max(gray.width, gray.height)
        canvas = Image.new("L", (side, side), 255)
        canvas.paste(gray, ((side - gray.width) // 2,
                            (side - gray.height) // 2))
        canvas.thumbnail((52, 52), Image.Resampling.LANCZOS)

        final = Image.new("L", (64, 64), 255)
        final.paste(canvas, ((64 - canvas.width) // 2,
                             (64 - canvas.height) // 2))
        return np.asarray(final, dtype=np.float32) / 255.0

for class_id in range(432):
    folder = root / f"box_{class_id}"
    for path in sorted(folder.glob("*.png")):
        images.append(preprocess(path))
        labels.append(class_id)

X = np.array(images)[..., np.newaxis]
y = np.array(labels, dtype=np.int32)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Dataset:", X.shape, "| Training:", len(X_train), "| Validation:", len(X_val))

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 1)),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(432, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        )
    ],
    verbose=1
)

train_pred = np.argmax(model.predict(X_train, verbose=0), axis=1)
val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

print("\n--- FULL DATASET DIAGNOSTIC ---")
print(f"Training accuracy: {accuracy_score(y_train, train_pred) * 100:.2f}%")
print(f"Validation accuracy: {accuracy_score(y_val, val_pred) * 100:.2f}%")
