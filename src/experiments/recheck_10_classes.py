from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

tf.keras.utils.set_random_seed(42)
root = Path("data/characters/processed_test")

X, y = [], []
for class_id in range(10):
    for path in sorted((root / f"box_{class_id}").glob("*.png")):
        with Image.open(path) as im:
            X.append(np.asarray(im.convert("L"), dtype=np.float32) / 255.0)
            y.append(class_id)

# Duplicate the 10-class set with distinct labels only for a pipeline
# sanity check? No: instead, use the original 10 classes for comparison.
X = np.array(X)[..., None]
y = np.array(y, dtype=np.int32)

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

model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

train_pred = np.argmax(model.predict(X_train, verbose=0), axis=1)
val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

print("Classes:", len(np.unique(y)))
print("Training accuracy:", round(accuracy_score(y_train, train_pred) * 100, 2), "%")
print("Validation accuracy:", round(accuracy_score(y_val, val_pred) * 100, 2), "%")
