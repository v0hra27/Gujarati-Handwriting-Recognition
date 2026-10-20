import json
from pathlib import Path

path = Path("models/gujarati_labels.json")
data = json.loads(path.read_text(encoding="utf-8"))

print("Data type:", type(data).__name__)
print("First 10 entries:")
if isinstance(data, dict):
    for i, (key, value) in enumerate(data.items()):
        print(repr(key), "=>", repr(value))
        if i >= 9:
            break
elif isinstance(data, list):
    for i, value in enumerate(data[:10]):
        print(i, "=>", repr(value))
else:
    print(repr(data)[:1000])
