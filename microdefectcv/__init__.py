"""
MicroDefectCV
=============
Adaptive OpenCV-based defect enhancement and segmentation for SEM and
microstructure images.

Originally developed for perovskite solar-cell SEM pinhole and PbI2
bright-particle detection.

Quick start::

    from microdefectcv import detect_defects

    result = detect_defects(image, mode="auto")
    print(result["defect_count"])
"""

from .filters import detect_defects, adaptive_defect_filter

__version__ = "0.1.1"
__all__ = ["detect_defects", "adaptive_defect_filter"]
