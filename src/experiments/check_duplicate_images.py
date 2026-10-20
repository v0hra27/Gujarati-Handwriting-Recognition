from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict

root = Path("data/characters/source/preprocessed_images")
fingerprints = defaultdict(list)
total = 0

for class_id in range(432):
    folder = root / f"box_{class_id}"
    for path in sorted(folder.glob("*.png")):
        with Image.open(path) as im:
            arr = np.asarray(im.convert("L").resize((32, 32)), dtype=np.uint8)

        # Exact fingerprint of the resized pixel array.
        fingerprint = arr.tobytes()
        fingerprints[fingerprint].append((class_id, str(path)))
        total += 1

cross_class_duplicates = []
for entries in fingerprints.values():
    class_ids = {entry[0] for entry in entries}
    if len(class_ids) > 1:
        cross_class_duplicates.append(entries)

print("Total images checked:", total)
print("Unique resized fingerprints:", len(fingerprints))
print("Cross-class duplicate groups:", len(cross_class_duplicates))

for group in cross_class_duplicates[:10]:
    print("\nDuplicate group:")
    for class_id, path in group:
        print(f"  Class {class_id}: {path}")
