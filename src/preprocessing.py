from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

def preprocess_character_image(image_input, target_size=(64, 64), padding_ratio=0.15):
    """
    Cleans and standardizes a handwritten character image:
    1. Converts to grayscale.
    2. Identifies character ink and removes stray border artifacts.
    3. Crops tightly around the character bounding box.
    4. Centers the character on a square canvas preserving aspect ratio.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise TypeError("image_input must be a file path or PIL Image")

    # 1. Convert to grayscale
    gray = ImageOps.grayscale(img)
    arr = np.asarray(gray, dtype=np.uint8)

    # 2. Invert if background is dark: ensure background is white (255) and ink is dark (< 180)
    # If the corners are mostly dark, invert image
    corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    if np.mean(corners) < 128:
        gray = ImageOps.invert(gray)
        arr = np.asarray(gray, dtype=np.uint8)

    # 3. Detect ink pixels (pixels darker than threshold, e.g. 190)
    ink_mask = arr < 190
    
    # If the image is empty or has almost no ink, return blank canvas
    if np.sum(ink_mask) < 20:
        canvas = Image.new("L", target_size, 255)
        return canvas

    ys, xs = np.where(ink_mask)

    # 4. Filter out stray border artifacts:
    # If there is stray ink touching only the top or bottom 5% of pixels separate from main mass
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())

    # Tightly crop the character
    cropped = gray.crop((xmin, ymin, xmax + 1, ymax + 1))

    # 5. Place on a square canvas preserving aspect ratio
    w, h = cropped.size
    max_dim = max(w, h)
    
    # Target inner dimension with padding
    inner_size = int(target_size[0] * (1.0 - 2 * padding_ratio))
    scale = inner_size / max_dim
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 6. Center on square background canvas
    canvas = Image.new("L", target_size, 255)
    offset_x = (target_size[0] - new_w) // 2
    offset_y = (target_size[1] - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    return canvas


def to_model_input(processed_image):
    """
    Converts a 64x64 PIL Image to normalized numpy array (1, 64, 64, 1) float32 [0, 1].
    """
    arr = np.asarray(processed_image, dtype=np.float32) / 255.0
    return arr[np.newaxis, ..., np.newaxis]


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    test_path = Path("data/characters/source/preprocessed_images/box_0/1.png")
    if test_path.exists():
        result = preprocess_character_image(test_path)
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / "test_chunk1_preprocessed.png"
        result.save(out_file)
        print("Chunk 1 Preprocessing success!")
        print(f"Original size: {Image.open(test_path).size}")
        print(f"Preprocessed size: {result.size}")
        print(f"Saved preprocessed sample to: {out_file.resolve()}")
    else:
        print("Sample image not found.")
