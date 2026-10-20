import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

root = Path("models")

with open(root / "character_labels.json", encoding="utf-8") as f:
    training_labels = json.load(f)

with open(root / "gujarati_labels.json", encoding="utf-8") as f:
    display_labels = json.load(f)

print("Training label file type:", type(training_labels).__name__)
print("Display label file type:", type(display_labels).__name__)

print("\nFirst 15 training labels:")
if isinstance(training_labels, dict):
    for key, value in list(training_labels.items())[:15]:
        print(repr(key), "=>", repr(value))
else:
    print(training_labels[:15])

print("\nFirst 15 display labels:")
for key, value in list(display_labels.items())[:15]:
    print(repr(key), "=>", repr(value))

print("\nExpected display labels for initial classes:")
for i in range(15):
    print(i, "=>", repr(display_labels.get(f"box_{i}")))
