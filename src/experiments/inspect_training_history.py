import json
from pathlib import Path

path = Path("models/training_results.json")

if not path.exists():
    print("Training results file not found.")
    raise SystemExit(1)

data = json.loads(path.read_text(encoding="utf-8"))

print("Training results structure:")
if isinstance(data, dict):
    for key, value in data.items():
        if isinstance(value, list):
            print(f"\n{key}:")
            print("  Number of values:", len(value))
            print("  First 5:", value[:5])
            print("  Last 5:", value[-5:])
        else:
            print(f"{key}: {value}")
else:
    print(type(data).__name__, repr(data)[:2000])
