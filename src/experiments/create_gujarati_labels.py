import json
from pathlib import Path

# Labels for the 12 initial vowel classes.
labels = [
    "અ", "આ", "ઇ", "ઈ", "ઉ", "ઊ",
    "ઋ", "એ", "ઐ", "ઓ", "ઔ", "અં"
]

# Base consonants and conjunct consonants, in README order.
bases = [
    "ક", "ખ", "ગ", "ઘ", "ચ", "છ", "જ", "ઝ",
    "ટ", "ઠ", "ડ", "ઢ", "ણ", "ત", "થ", "દ",
    "ધ", "ન", "પ", "ફ", "બ", "ભ", "મ", "ય",
    "ર", "લ", "વ", "શ", "ષ", "સ", "હ", "ળ",
    "ક્ષ", "ત્ર", "જ્ઞ"
]

# Forms listed in the README for each base.
suffixes = [
    "", "ા", "િ", "ી", "ુ", "ૂ",
    "ે", "ૈ", "ો", "ૌ", "ં", "ઃ"
]

for base in bases:
    for suffix in suffixes:
        labels.append(base + suffix)

if len(labels) != 432:
    raise ValueError(f"Expected 432 labels, got {len(labels)}")

mapping = {f"box_{i}": labels[i] for i in range(432)}

out = Path("models/gujarati_labels.json")
out.parent.mkdir(exist_ok=True)
out.write_text(
    json.dumps(mapping, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("Saved:", out)
for i in [0, 1, 12, 13, 36, 60, 100, 200, 431]:
    print(f"box_{i}: {mapping[f'box_{i}']}")
