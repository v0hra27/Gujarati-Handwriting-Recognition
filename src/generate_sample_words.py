import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image

ROOT = Path("D:/Kaif/Sem7/ML/Gujarati-Handwriting-Recognition")
DATA_DIR = ROOT / "data" / "characters" / "source" / "preprocessed_images"
SAMPLE_DIR = ROOT / "data" / "sample_images"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

# Word definitions with verified clean handwriting samples:
WORDS = [
    {
        "name": "kamal",
        "gujarati": "કમળ",
        "boxes": [("box_12", "1.png"), ("box_276", "1.png"), ("box_384", "2.png")]
    },
    {
        "name": "kalam",
        "gujarati": "કલમ",
        "boxes": [("box_12", "2.png"), ("box_312", "2.png"), ("box_276", "2.png")]
    },
    {
        "name": "saral",
        "gujarati": "સરળ",
        "boxes": [("box_360", "2.png"), ("box_300", "2.png"), ("box_384", "2.png")]
    },
    {
        "name": "vatan",
        "gujarati": "વતન",
        "boxes": [("box_324", "2.png"), ("box_168", "5.png"), ("box_216", "1.png")]
    },
    {
        "name": "bharat",
        "gujarati": "ભરત",
        "boxes": [("box_264", "2.png"), ("box_300", "2.png"), ("box_168", "6.png")]
    },
    {
        "name": "gagan",
        "gujarati": "ગગન",
        "boxes": [("box_36", "1.png"), ("box_36", "1.png"), ("box_216", "2.png")]
    },
    {
        "name": "raman",
        "gujarati": "રમણ",
        "boxes": [("box_300", "1.png"), ("box_276", "1.png"), ("box_156", "1.png")]
    },
    {
        "name": "nagar",
        "gujarati": "નગર",
        "boxes": [("box_216", "1.png"), ("box_36", "1.png"), ("box_300", "2.png")]
    },
    {
        "name": "janak",
        "gujarati": "જનક",
        "boxes": [("box_84", "1.png"), ("box_216", "2.png"), ("box_12", "1.png")]
    },
    {
        "name": "madad",
        "gujarati": "મદદ",
        "boxes": [("box_276", "2.png"), ("box_192", "1.png"), ("box_192", "2.png")]
    },
    {
        "name": "das",
        "gujarati": "દસ",
        "boxes": [("box_192", "1.png"), ("box_360", "1.png")]
    },
    {
        "name": "pag",
        "gujarati": "પગ",
        "boxes": [("box_228", "1.png"), ("box_36", "1.png")]
    },
    {
        "name": "taras",
        "gujarati": "તરસ",
        "boxes": [("box_168", "5.png"), ("box_300", "1.png"), ("box_360", "2.png")]
    },
    {
        "name": "pavan",
        "gujarati": "પવન",
        "boxes": [("box_228", "2.png"), ("box_324", "1.png"), ("box_216", "1.png")]
    },
    {
        "name": "nayan",
        "gujarati": "નયન",
        "boxes": [("box_216", "1.png"), ("box_288", "1.png"), ("box_216", "2.png")]
    }
]

print("Regenerating 15 clean, verified sample words...")
for w_info in WORDS:
    loaded = []
    for box, filename in w_info["boxes"]:
        p = DATA_DIR / box / filename
        if not p.exists():
            files = sorted((DATA_DIR / box).glob("*.png"))
            if files:
                p = files[0]
        if p.exists():
            loaded.append(Image.open(p))

    if len(loaded) == len(w_info["boxes"]):
        spacing = 22
        total_w = sum(im.width for im in loaded) + spacing * (len(loaded) + 1)
        max_h = max(im.height for im in loaded) + 30

        canvas = Image.new("RGB", (total_w, max_h), "white")
        cur_x = spacing
        for im in loaded:
            canvas.paste(im, (cur_x, 15))
            cur_x += im.width + spacing

        out_file = SAMPLE_DIR / f"{w_info['name']}.png"
        canvas.save(out_file)
        print(f"Generated: {w_info['name']}.png ({w_info['gujarati']})")

print(f"\nAll 15 sample words ready in: {SAMPLE_DIR.resolve()}")
