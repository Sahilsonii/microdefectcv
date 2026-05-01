"""
preprocessing.py — MicroDefectCV
=================================
Image preparation helpers: grayscale conversion, normalization, CLAHE, denoising.
"""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale if it is in colour (BGR).

    Args:
        image: Input image (BGR 3-channel or grayscale 2-channel).

    Returns:
        Grayscale uint8 image.
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Stretch image intensity to the full 0-255 range.

    Args:
        image: Grayscale input image.

    Returns:
        Normalized uint8 image.
    """
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Args:
        image:          Grayscale input image.
        clip_limit:     Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.

    Returns:
        Contrast-enhanced uint8 image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def denoise_image(
    image: np.ndarray,
    blur_kernel: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Apply Gaussian blur for denoising.

    Args:
        image:       Grayscale input image.
        blur_kernel: Kernel size (must be odd; even values are incremented by 1).
        sigma:       Standard deviation for the Gaussian kernel.

    Returns:
        Denoised uint8 image.
    """
    ksize = blur_kernel if blur_kernel % 2 != 0 else blur_kernel + 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)
