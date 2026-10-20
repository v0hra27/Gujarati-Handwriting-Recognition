from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

base = Path("data/characters/source")
original = base / "Images"
processed = base / "preprocessed_images"

print("Original Images folder exists:", original.exists())
print("Preprocessed folder exists:", processed.exists())

for folder in [original, processed]:
    if folder.exists():
        print("\n", folder)
        for item in list(folder.iterdir())[:8]:
            print(" ", item.name, "directory" if item.is_dir() else "file")

for class_id in [0, 1, 12, 13]:
    folder = processed / f"box_{class_id}"
    files = sorted(folder.glob("*.png"))
    if files:
        img = Image.open(files[0]).convert("L")
        print(f"box_{class_id}: {len(files)} images, size={img.size}, "
              f"pixel range={img.getextrema()}")
