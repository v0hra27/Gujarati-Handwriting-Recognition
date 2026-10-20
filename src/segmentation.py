import cv2
import numpy as np
from PIL import Image

def segment_word_into_characters(image_input, min_area=40, max_gap_merge=8):
    """
    Segments a handwritten Gujarati word image into individual characters.
    Features:
    - Filters out horizontal/vertical ruling lines (scan artifacts)
    - Removes isolated tiny noise dots
    - Merges character components that belong to the same glyph
    - Returns ordered crops left-to-right
    """
    if isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
        img_np = np.array(pil_img)
    else:
        pil_img = Image.open(image_input).convert("RGB")
        img_np = np.array(pil_img)

    # 1. Grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 2. Ensure paper is black background, ink is white foreground
    corners = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
    if np.mean(corners) < 128:
        gray = 255 - gray

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Filter out horizontal ruling lines (e.g. from lined notebook paper or scan grid)
        aspect_ratio = w / max(1, h)
        if aspect_ratio > 3.5 and h <= 10:
            continue
        # Filter out vertical margin lines
        if (h / max(1, w)) > 5.0 and w <= 6:
            continue

        boxes.append((x, y, w, h))

    if not boxes:
        return [pil_img], [(0, 0, pil_img.width, pil_img.height)], pil_img

    # Sort left-to-right
    boxes = sorted(boxes, key=lambda b: b[0])

    # 4. Merge character pieces that belong to the same glyph
    merged_boxes = []
    skip_indices = set()

    for i in range(len(boxes)):
        if i in skip_indices:
            continue
        x1, y1, w1, h1 = boxes[i]

        for j in range(i + 1, len(boxes)):
            if j in skip_indices:
                continue
            x2, y2, w2, h2 = boxes[j]

            # Horizontal overlap check
            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            min_w = min(w1, w2)

            # If overlapping significantly horizontally or very close and overlapping in y
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            is_same_char = (
                (x_overlap > 0.35 * min_w) or
                (abs(x2 - (x1 + w1)) <= max_gap_merge and y_overlap > 0.4 * min(h1, h2))
            )

            if is_same_char:
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(x1 + w1, x2 + w2) - new_x
                new_h = max(y1 + h1, y2 + h2) - new_y
                x1, y1, w1, h1 = new_x, new_y, new_w, new_h
                skip_indices.add(j)

        merged_boxes.append((x1, y1, w1, h1))

    # Re-sort left-to-right
    merged_boxes = sorted(merged_boxes, key=lambda b: b[0])

    # 5. Extract crops and draw annotated bounding boxes
    annotated = img_np.copy()
    segmented_crops = []

    for idx, (x, y, w, h) in enumerate(merged_boxes, 1):
        pad_x = max(0, x - 3)
        pad_y = max(0, y - 3)
        pad_w = min(img_np.shape[1] - pad_x, w + 6)
        pad_h = min(img_np.shape[0] - pad_y, h + 6)

        crop_arr = gray[pad_y:pad_y + pad_h, pad_x:pad_x + pad_w]
        crop_pil = Image.fromarray(crop_arr)
        segmented_crops.append(crop_pil)

        # Draw green bounding box & index label
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (46, 204, 113), 2)
        cv2.putText(
            annotated,
            str(idx),
            (x + 2, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (41, 128, 185),
            2,
            cv2.LINE_AA
        )

    annotated_pil = Image.fromarray(annotated)
    return segmented_crops, merged_boxes, annotated_pil
