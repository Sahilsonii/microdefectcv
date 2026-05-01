"""
postprocessing.py — MicroDefectCV
===================================
Mask cleaning, contour extraction, and contour drawing utilities.
"""

import cv2
import numpy as np
from typing import List, Tuple


def clean_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Clean a binary mask with morphological closing then opening.

    Closing fills small holes; opening removes isolated noise pixels.

    Args:
        mask:        Binary uint8 mask (0/255).
        kernel_size: Side length of the elliptical structuring element.

    Returns:
        Cleaned binary uint8 mask.
    """
    ksize = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned


def extract_contours(mask: np.ndarray, min_area: float = 20) -> List[np.ndarray]:
    """
    Find external contours in a binary mask, filtering by minimum area.

    Args:
        mask:     Binary uint8 mask.
        min_area: Minimum contour area in pixels.

    Returns:
        List of contours (each is an ndarray of shape (N, 1, 2)).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def draw_defect_contours(
    image: np.ndarray,
    contours: List[np.ndarray],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw contours on a copy of the input image.

    Args:
        image:     Grayscale or BGR image (will be converted to BGR if needed).
        contours:  List of contours to draw.
        color:     BGR colour tuple.
        thickness: Line thickness in pixels.

    Returns:
        BGR image with contours drawn.
    """
    if len(image.shape) == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()
    cv2.drawContours(out, contours, -1, color, thickness)
    return out
