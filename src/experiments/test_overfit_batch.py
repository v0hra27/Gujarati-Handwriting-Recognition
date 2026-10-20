from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

root = Path("data/characters/processed_test")
X, y = [], []

for class_id in range(10):
    paths = sorted((root / f"box_{class_id}").glob("*.png"))
    for path in paths:
        with Image.open(path) as im:
            X.append(np.asarray(im.convert("L"), dtype=np.float32) / 255.0)
            y.append(class_id)

X = np.array(X, dtype=np.float32)[..., np.newaxis]
y = np.array(y, dtype=np.int32)

# Fixed batch: use the first 6 images of each of the 10 classes.
indices = []
for class_id in range(10):
    indices.extend(np.where(y == class_id)[0][:6])

X_small, y_small = X[indices], y[indices]
print("Fixed training batch:", X_small.shape, "classes:", len(np.unique(y_small)))

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 1)),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_small, y_small, epochs=60, batch_size=16, verbose=0)

loss, accuracy = model.evaluate(X_small, y_small, verbose=0)
print(f"Fixed-batch loss: {loss:.4f}")
print(f"Fixed-batch accuracy: {accuracy * 100:.2f}%")
