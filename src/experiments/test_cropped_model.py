from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

tf.keras.utils.set_random_seed(42)

root = Path("data/characters/processed_test")
images, labels = [], []

for class_id in range(10):
    folder = root / f"box_{class_id}"
    for path in sorted(folder.glob("*.png")):
        with Image.open(path) as img:
            arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        images.append(arr)
        labels.append(class_id)

X = np.array(images)[..., np.newaxis]
y = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 1)),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=8,
    verbose=1
)

train_pred = np.argmax(model.predict(X_train, verbose=0), axis=1)
val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

print("\n--- CROPPED DATASET RESULTS ---")
print(f"Training accuracy: {accuracy_score(y_train, train_pred) * 100:.2f}%")
print(f"Validation accuracy: {accuracy_score(y_val, val_pred) * 100:.2f}%")
print(f"Images: {len(X)} | Classes: {len(np.unique(y))}")
