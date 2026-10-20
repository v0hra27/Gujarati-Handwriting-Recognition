from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

tf.keras.utils.set_random_seed(42)

root = Path("data/characters/source/preprocessed_images")
selected = [root / f"box_{i}" for i in range(10)]

X, y = [], []

for label, folder in enumerate(selected):
    files = sorted(folder.glob("*.png"))
    for path in files:
        with Image.open(path) as img:
            img = img.convert("L").resize((64, 64))
            X.append(np.asarray(img, dtype=np.float32) / 255.0)
            y.append(label)

X = np.asarray(X)[..., np.newaxis]
y = np.asarray(y, dtype=np.int32)

print("Images:", len(X))
print("Image tensor:", X.shape)
print("Images per class:", np.bincount(y))

model = models.Sequential([
    layers.Input(shape=(64, 64, 1)),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(96, 3, padding="same", activation="relu"),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(X, y, epochs=30, batch_size=16, verbose=2)

loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"\nTraining-set accuracy: {accuracy:.2%}")
print("This is a diagnostic test on training data, not a validation score.")
