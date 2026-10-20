from pathlib import Path
from PIL import Image, ImageOps
import numpy as np

source = Path("data/characters/source/preprocessed_images")
destination = Path("data/characters/processed_test")

def crop_and_center(image, size=64, margin=6):
    gray = ImageOps.grayscale(image)
    arr = np.asarray(gray)

    # Find dark strokes against the light background.
    mask = arr < 180
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return gray.resize((size, size))

    left, right = xs.min(), xs.max() + 1
    top, bottom = ys.min(), ys.max() + 1
    cropped = gray.crop((left, top, right, bottom))

    available = size - 2 * margin
    scale = min(available / cropped.width, available / cropped.height)
    new_w = max(1, round(cropped.width * scale))
    new_h = max(1, round(cropped.height * scale))
    cropped = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (size, size), 255)
    canvas.paste(cropped, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas

for class_id in range(10):
    src = source / f"box_{class_id}"
    dst = destination / f"box_{class_id}"
    dst.mkdir(parents=True, exist_ok=True)

    for path in sorted(src.glob("*.png")):
        with Image.open(path) as img:
            result = crop_and_center(img)
            result.save(dst / path.name)

print("Processed test dataset saved to:", destination.resolve())
print("Original dataset remains unchanged.")
