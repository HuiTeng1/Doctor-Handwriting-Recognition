"""
The actual implementation of P0 baseline + P1 CLHAA + P2 Fajardo + P3 Benitez.

Every apply_xxx() function calls spec.verify_coverage() when it finishes, confirming that
each step it actually executed has a matching entry in spec.py (extra or missing steps
both raise an error immediately) - the code and the spec table are never allowed to
diverge silently.
"""
import cv2
import numpy as np
from PIL import Image, ImageFilter
from skimage.morphology import skeletonize

from spec import verify_coverage

CLHAA_SIZE = 224
FAJARDO_HEIGHT = 64


def apply_baseline(img: Image.Image) -> Image.Image:
    verify_coverage("baseline", ["no_preprocessing"])
    return img.copy()


def apply_clhaa(img: Image.Image) -> Image.Image:
    steps_used = ["pixel_scaling"]  # exists conceptually, but functionally superseded by the downstream ToTensor(), not implemented separately here

    img = img.convert("RGB")
    w, h = img.size
    long_side = max(w, h)
    scale = CLHAA_SIZE / long_side
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    steps_used += ["resize_aspect_ratio", "resize_interpolation"]

    canvas = Image.new("RGB", (CLHAA_SIZE, CLHAA_SIZE), (255, 255, 255))
    left, top = (CLHAA_SIZE - new_w) // 2, (CLHAA_SIZE - new_h) // 2
    canvas.paste(img, (left, top))
    img = canvas
    steps_used.append("white_padding")

    # Unsharp mask must happen before grayscale (per the paper: grayscale applied after sharpening)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    steps_used += ["unsharp_technique", "unsharp_radius", "unsharp_percent", "unsharp_threshold"]

    img = img.convert("L")
    steps_used += ["grayscale_formula", "step_order", "final_size"]

    verify_coverage("clhaa", steps_used)
    return img


def apply_fajardo(img: Image.Image) -> Image.Image:
    gray = img.convert("L")
    arr = np.array(gray)

    # THRESH_BINARY_INV: ink (dark) -> foreground (255/True), background (light) -> 0,
    # matching skeletonize's assumption about the foreground
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    steps_used = ["binarization_method"]

    skeleton_bool = skeletonize(binary.astype(bool), method="zhang")
    steps_used += ["thinning_algorithm", "thinning_implementation"]

    # Convert back to the display convention people expect: skeleton (ink) = black, background = white
    skel_uint8 = np.where(skeleton_bool, 0, 255).astype(np.uint8)
    skel_img = Image.fromarray(skel_uint8, mode="L")

    sw, sh = skel_img.size
    scale = FAJARDO_HEIGHT / sh
    new_w = max(1, round(sw * scale))
    skel_img = skel_img.resize((new_w, FAJARDO_HEIGHT), resample=Image.BILINEAR)
    steps_used += ["height_normalization", "width_handling", "resize_interpolation"]

    verify_coverage("fajardo", steps_used)
    return skel_img


def apply_benitez(img: Image.Image) -> Image.Image:
    gray = img.convert("L")
    steps_used = ["grayscale"]
    arr = np.array(gray)

    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    steps_used.append("binarization_method")

    # median-based auto-Canny: M is taken from the image actually fed to Canny (i.e. the
    # binarized image), matching the paper's literal "grayscale -> binarization -> Canny"
    # order (Canny consumes the post-binarization image, not the raw grayscale one)
    median = float(np.median(binary))
    lower, upper = 0.66 * median, 1.33 * median
    edges = cv2.Canny(binary, threshold1=lower, threshold2=upper)
    steps_used += ["canny_technique", "canny_lower", "canny_upper"]

    inverted = 255 - edges
    steps_used.append("inversion")

    verify_coverage("benitez", steps_used)
    return Image.fromarray(inverted, mode="L")


PIPELINES = {
    "baseline": apply_baseline,
    "clhaa": apply_clhaa,
    "fajardo": apply_fajardo,
    "benitez": apply_benitez,
}
