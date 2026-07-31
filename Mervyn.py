"""Image preprocessing for handwriting recognition.

Pipeline: grayscale -> denoise -> deskew -> adaptive threshold (binarize)
-> crop to text bounding box -> resize/pad to a fixed canvas.

Run directly to preprocess Data/img 3 .jpg and write the result next to it.
"""

import os

import cv2
import numpy as np

TARGET_SIZE = (512, 128)  # (width, height); kept large so the result stays human-readable


def to_grayscale(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(gray):
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def deskew(gray):
    """Estimate and correct small rotations using the angle of the ink mask."""
    inverted = cv2.bitwise_not(gray)
    _, mask = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(mask > 0))
    if coords.shape[0] < 20:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return gray

    h, w = gray.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def binarize(gray):
    # Large blockSize so the slowly-varying paper/watermark background washes
    # out while thin, blurred ink strokes survive (small blockSize + low C
    # shreds the strokes on this motion-blurred source image).
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=51,
        C=10,
    )


def crop_to_text(binary_img, padding=10):
    """Crop to the bounding box of dark ink pixels, with a small margin."""
    ink = cv2.bitwise_not(binary_img)
    coords = cv2.findNonZero(ink)
    if coords is None:
        return binary_img

    x, y, w, h = cv2.boundingRect(coords)
    x0 = max(x - padding, 0)
    y0 = max(y - padding, 0)
    x1 = min(x + w + padding, binary_img.shape[1])
    y1 = min(y + h + padding, binary_img.shape[0])
    return binary_img[y0:y1, x0:x1]


def resize_with_padding(img, target_size=TARGET_SIZE, pad_value=255):
    target_w, target_h = target_size
    h, w = img.shape
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_h, target_w), pad_value, dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def normalize(img):
    """Scale to float32 [0, 1] for model input."""
    return img.astype(np.float32) / 255.0


def preprocess_image(path, target_size=TARGET_SIZE):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    gray = to_grayscale(img)
    denoised = denoise(gray)
    straightened = deskew(denoised)
    binary = binarize(straightened)
    cropped = crop_to_text(binary)
    canvas = resize_with_padding(cropped, target_size)
    normalized = normalize(canvas)
    return canvas, normalized


def main():
    src_path = os.path.join(os.path.dirname(__file__), "Data", "img 3 .jpg")
    out_path = os.path.join(os.path.dirname(__file__), "Data", "img_3_preprocessed.png")

    canvas, normalized = preprocess_image(src_path)
    cv2.imwrite(out_path, canvas)

    print(f"Input:  {src_path}")
    print(f"Output: {out_path}")
    print(f"Shape:  {canvas.shape} (H, W)")
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")


if __name__ == "__main__":
    main()
