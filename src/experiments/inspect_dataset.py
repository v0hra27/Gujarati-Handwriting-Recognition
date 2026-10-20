from pathlib import Path
from collections import Counter
from PIL import Image

root = Path("data/characters/source/preprocessed_images")
folders = sorted(
    root.glob("box_*"),
    key=lambda p: int(p.name.split("_")[1])
)

class_counts = {}
image_sizes = Counter()
bad_images = []

for folder in folders:
    files = list(folder.glob("*"))
    valid_count = 0

    for file in files:
        if not file.is_file():
            continue
        try:
            with Image.open(file) as img:
                img.verify()
            with Image.open(file) as img:
                image_sizes[img.size] += 1
            valid_count += 1
        except Exception:
            bad_images.append(str(file))

    class_counts[folder.name] = valid_count

print("Number of classes:", len(folders))
print("Total valid images:", sum(class_counts.values()))
print("Classes with no valid images:",
      [name for name, count in class_counts.items() if count == 0])
print("Minimum images in a class:", min(class_counts.values(), default=0))
print("Maximum images in a class:", max(class_counts.values(), default=0))
print("Most common image dimensions:", image_sizes.most_common(10))
print("Unreadable images:", len(bad_images))
if bad_images:
    print("Examples:", bad_images[:10])
