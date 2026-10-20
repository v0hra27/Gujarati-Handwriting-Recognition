import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
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
    }
]

print("Regenerating sample words with clean glyphs...")
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
        spacing = 20
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

print(f"\nAll clean sample words ready in: {SAMPLE_DIR.resolve()}")
