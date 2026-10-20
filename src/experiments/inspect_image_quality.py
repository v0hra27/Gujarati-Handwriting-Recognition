from pathlib import Path
from PIL import Image
import numpy as np

root = Path("data/characters/source/preprocessed_images")

print("Image statistics for the first 10 classes\n")

for class_id in range(10):
    folder = root / f"box_{class_id}"
    files = sorted(folder.glob("*.png"))
    stats = []

    for path in files:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("L"), dtype=np.uint8)

        stats.append({
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "dark": float((arr < 128).mean() * 100),
            "light": float((arr >= 128).mean() * 100),
        })

    if not stats:
        print(f"Class {class_id}: NO IMAGES")
        continue

    print(
        f"Class {class_id}: {len(stats)} images | "
        f"mean brightness={np.mean([s['mean'] for s in stats]):.1f} | "
        f"contrast std={np.mean([s['std'] for s in stats]):.1f} | "
        f"dark pixels={np.mean([s['dark'] for s in stats]):.1f}%"
    )

print("\nCompare original and preprocessed images:")
for path in [root / "box_0", root / "box_1", root / "box_2"]:
    files = sorted(path.glob("*.png"))
    if files:
        img = Image.open(files[0]).convert("L")
        print(path.name, files[0].name, img.size, img.getextrema())
