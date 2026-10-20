from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import numpy as np

root = Path("data/characters/source/preprocessed_images")
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

cell_w, cell_h = 240, 220
sheet = Image.new("RGB", (cell_w * 4, cell_h * 3), "white")
draw = ImageDraw.Draw(sheet)

for class_id in range(10):
    folder = root / f"box_{class_id}"
    files = sorted(folder.glob("*.png"))
    if not files:
        continue

    # Show up to two samples from each class.
    for sample_index, path in enumerate(files[:2]):
        with Image.open(path) as img:
            gray = img.convert("L")
            arr = np.asarray(gray)
            # Stretch contrast for viewing only.
            low, high = np.percentile(arr, [2, 98])
            if high > low:
                arr = np.clip((arr.astype(float) - low) * 255 / (high - low), 0, 255)
            view = Image.fromarray(arr.astype("uint8"))
            view = ImageOps.autocontrast(view)
            view = view.resize((128, 128))

        col = sample_index * 2
        row = class_id // 4
        col += (class_id % 4)
        x = col * (cell_w // 2)
        y = row * cell_h
        sheet.paste(view.convert("RGB"), (x + 10, y + 25))
        draw.text((x + 10, y + 5), f"Class {class_id}, sample {sample_index + 1}", fill="black")

sheet.save(out_dir / "character_quality_preview.png")
print("Saved:", (out_dir / "character_quality_preview.png").resolve())
