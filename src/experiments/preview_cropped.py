from pathlib import Path
from PIL import Image, ImageDraw
import math

root = Path("data/characters/processed_test")
files = []

for class_id in range(10):
    folder = root / f"box_{class_id}"
    for path in sorted(folder.glob("*.png"))[:2]:
        files.append((class_id, path))

cell_w, cell_h = 160, 145
cols = 5
rows = math.ceil(len(files) / cols)
sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
draw = ImageDraw.Draw(sheet)

for i, (class_id, path) in enumerate(files):
    with Image.open(path) as img:
        img = img.convert("L").resize((100, 100))
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        sheet.paste(img.convert("RGB"), (x + 25, y + 25))
        draw.text((x + 5, y + 5), f"Class {class_id}: {path.name}", fill="black")

out = Path("outputs/cropped_character_preview.png")
out.parent.mkdir(exist_ok=True)
sheet.save(out)
print("Preview saved:", out.resolve())
