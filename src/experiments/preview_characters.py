from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt

root = Path("data/characters/source/preprocessed_images")
class_ids = [0, 1, 12, 13, 36, 60, 100, 200]
thumbs = []

for class_id in class_ids:
    folder = root / f"box_{class_id}"
    files = sorted(folder.glob("*.png"))

    if not files:
        continue

    img = Image.open(files[0]).convert("L")
    img.thumbnail((128, 128))

    canvas = Image.new("RGB", (160, 175), "white")
    canvas.paste(ImageOps.colorize(img, black="black", white="white"), (16, 8))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 145), f"Class {class_id}", fill="black")
    thumbs.append(canvas)

if not thumbs:
    raise RuntimeError("No preview images found.")

cols = 4
rows = (len(thumbs) + cols - 1) // cols
sheet = Image.new("RGB", (cols * 160, rows * 175), "#dddddd")

for i, img in enumerate(thumbs):
    sheet.paste(img, ((i % cols) * 160, (i // cols) * 175))

out = Path("outputs/character_preview.png")
out.parent.mkdir(exist_ok=True)
sheet.save(out)
print("Saved preview to:", out.resolve())
plt.imshow(sheet)
plt.axis("off")
plt.show()
