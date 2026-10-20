import json
from pathlib import Path
from collections import Counter

path = Path("models/gujarati_labels.json")
labels = json.loads(path.read_text(encoding="utf-8"))

expected_keys = {f"box_{i}" for i in range(432)}
actual_keys = set(labels.keys())

print("Total entries:", len(labels))
print("Missing keys:", sorted(expected_keys - actual_keys)[:20])
print("Unexpected keys:", sorted(actual_keys - expected_keys)[:20])

print("\nSelected labels:")
for i in [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    12, 13, 24, 36, 48, 60, 84, 96, 108, 120,
    132, 168, 228, 276, 324, 372, 384, 396, 408, 420, 431
]:
    print(f"{i}: {labels.get(f'box_{i}', 'MISSING')}")

counts = Counter(labels.values())
duplicates = [label for label, count in counts.items() if count > 1]

print("\nUnique label strings:", len(counts))
print("Duplicate label strings:", duplicates[:20])
print("Verification complete.")
