"""
metrics.py — MicroDefectCV
==========================
Functions for computing defect statistics from masks and contours.
"""

import cv2
import numpy as np
from typing import List, Dict


def calculate_defect_area(contours: List[np.ndarray]) -> float:
    """
    Calculate the total pixel area covered by all detected defect contours.

    Args:
        contours: List of contours from ``extract_contours``.

    Returns:
        Total area in pixels (float).
    """
    return float(sum(cv2.contourArea(c) for c in contours))


def calculate_defect_area_ratio(mask: np.ndarray) -> float:
    """
    Calculate the fraction of the image covered by defect pixels.

    Args:
        mask: Binary mask (uint8, 0/255).

    Returns:
        Ratio in [0, 1].
    """
    total_pixels = mask.shape[0] * mask.shape[1]
    if total_pixels == 0:
        return 0.0
    defect_pixels = int(np.count_nonzero(mask))
    return defect_pixels / total_pixels


def summarize_defects(mask: np.ndarray, contours: List[np.ndarray]) -> Dict:
    """
    Produce a summary dictionary of defect statistics.

    Args:
        mask: Binary defect mask (uint8).
        contours: Filtered contour list.

    Returns:
        Dictionary with keys:
          - defect_count (int)
          - total_defect_area_px (float)
          - defect_area_ratio (float)
          - mean_defect_area_px (float)
          - min_defect_area_px (float)
          - max_defect_area_px (float)
    """
    areas = [cv2.contourArea(c) for c in contours]
    return {
        "defect_count": len(contours),
        "total_defect_area_px": float(sum(areas)),
        "defect_area_ratio": calculate_defect_area_ratio(mask),
        "mean_defect_area_px": float(np.mean(areas)) if areas else 0.0,
        "min_defect_area_px": float(np.min(areas)) if areas else 0.0,
        "max_defect_area_px": float(np.max(areas)) if areas else 0.0,
    }
