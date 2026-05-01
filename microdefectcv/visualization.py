"""
visualization.py — MicroDefectCV
=================================
Optional visualization utilities for defect detection results.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

# Per defect-type BGR colour palette (matches original PerovskiteDefectFilter)
_CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "pbi2_bright": (0, 255, 255),   # cyan
    "pbi2_needle": (0, 200, 255),   # orange-yellow
    "pinhole_small": (0, 255, 0),   # green
    "pinhole_large": (100, 100, 255), # red-blue
}


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend a binary defect mask over the original image.

    Args:
        image: Grayscale or BGR image.
        mask: Binary mask (uint8, 0/255).
        color: BGR color for the defect overlay.
        alpha: Opacity of the overlay (0 = invisible, 1 = opaque).

    Returns:
        BGR image with color overlay.
    """
    if len(image.shape) == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()

    color_layer = np.zeros_like(out, dtype=np.uint8)
    color_layer[mask > 0] = color
    return cv2.addWeighted(out, 1.0, color_layer, alpha, 0)

def annotate_detections(
    image: np.ndarray,
    detections: List[Dict],
) -> np.ndarray:
    """
    Draw coloured bounding boxes and class labels on the image.

    Args:
        image:      BGR or grayscale image.
        detections: List of detection dicts (from ``detect_defects`` result["detections"]).

    Returns:
        Annotated BGR image.
    """
    if len(image.shape) == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()

    for det in detections:
        x, y, w, h = det["bbox"]
        dtype = det["defect_type"]
        color = _CLASS_COLORS.get(dtype, (255, 255, 255))
        thickness = 1 if w * h < 100 else 2
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        label = f"{dtype[:8]} #{det['class_id']}"
        font_scale = 0.28 if w * h < 100 else 0.38
        cv2.putText(out, label, (x, max(y - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
    return out


def save_yolo_annotations(
    detections: List[Dict],
    image_shape: Tuple[int, int],
    output_path: str,
) -> None:
    """
    Save detections to a YOLO-format .txt annotation file.

    Each line: ``class_id cx cy w h`` (all normalised to [0, 1]).

    Args:
        detections:   List of detection dicts from ``detect_defects``.
        image_shape:  (height, width) of the image the detections were made on.
        output_path:  Path to the output .txt file.
    """
    img_h, img_w = image_shape[:2]
    lines = []
    for det in detections:
        x, y, w, h = det["bbox"]
        cx = (x + w / 2.0) / img_w
        cy = (y + h / 2.0) / img_h
        wn = w / img_w
        hn = h / img_h
        lines.append(f"{det['class_id']} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def save_result_grid(
    image: np.ndarray,
    result: Dict,
    output_path: str,
    title_prefix: str = "MicroDefectCV",
) -> None:
    """
    Save a multi-panel matplotlib grid showing the detection pipeline stages.

    The function reads the optional ``intermediates`` dictionary from the
    result dict (populated when ``return_intermediate=True`` is passed to
    ``detect_defects``).

    Args:
        image:        Original input image (grayscale or BGR).
        result:       Dictionary returned by ``detect_defects``.
        output_path:  Path where the PNG grid will be saved.
        title_prefix: Suptitle prefix string.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    intermediates: Dict = result.get("intermediates", {})
    detections: List = result.get("detections", [])
    mode: str = result.get("mode", "?")

    # Build annotated detection panel with coloured bounding boxes
    annotated = annotate_detections(image, detections)

    # Compose display panels (only include stages that were captured)
    display: Dict[str, np.ndarray] = {}
    display["Original Image"] = image

    stage_labels = {
        "02_enhanced":           "CLAHE Enhanced",
        "03_grain_boundary_mask":"Grain Boundary\nSuppression",
        "04_bright_mask":        "Bright Particle\nMask (PbI₂)",
        "05_dark_mask":          "Dark Pit Mask\n(Pinholes)",
        "06_needle_mask":        "Needle Crystal\nMask",
    }
    for key, label in stage_labels.items():
        if key in intermediates:
            display[label] = intermediates[key]

    display[f"Detections ({result.get('defect_count', 0)})"] = annotated

    n = len(display)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (panel_title, img) in zip(axes, display.items()):
        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(panel_title, fontsize=9, fontweight="bold")
        ax.axis("off")

    stats_str = (
        f"mode={mode}  |  defects={result.get('defect_count', 0)}"
        f"  |  area_ratio={result.get('defect_area_ratio', 0.0):.3f}"
    )
    plt.suptitle(
        f"{title_prefix}  |  {stats_str}",
        fontsize=10,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
