import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow import keras

DATA = Path("data/characters/source/preprocessed_images")
IMAGE_SIZE = 64
SEED = 42

images, labels = [], []
folders = sorted(
    [p for p in DATA.iterdir() if p.is_dir()],
    key=lambda p: int(p.name.replace("box_", ""))
)

for class_id, folder in enumerate(folders):
    for path in sorted(folder.glob("*.png")):
        try:
            from PIL import Image
            img = Image.open(path).convert("L").resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            )
            images.append(np.array(img, dtype=np.float32) / 255.0)
            labels.append(class_id)
        except Exception as e:
            print("Skipped:", path, e)

X = np.array(images)[..., np.newaxis]
y = np.array(labels)

_, X_val, _, y_val = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

model = keras.models.load_model("models/gujarati_character_cnn.keras")
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)

print(f"Validation accuracy: {accuracy:.4%}")
print(f"Validation loss: {loss:.4f}")

predictions = model.predict(X_val[:10], verbose=0)
for i, probabilities in enumerate(predictions):
    top = np.argsort(probabilities)[-3:][::-1]
    print(
        f"Sample {i + 1}: actual class={y_val[i]}, "
        f"top predictions={[(int(k), round(float(probabilities[k]), 4)) for k in top]}"
    )
