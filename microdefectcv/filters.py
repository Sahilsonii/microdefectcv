"""
filters.py — MicroDefectCV
===========================
Main public API for defect detection in SEM and microstructure images.

Originally developed for perovskite solar-cell SEM pinhole and PbI2 detection.
Supports multiple morphology modes and defect targets.

Modes (morphology):
    "auto"   — Auto-detect image morphology from statistics
    "2d"     — 2D perovskite: flatter morphology, subtle defects
    "3d"     — 3D perovskite: grain boundary suppression + both defect paths
    "3d_2d"  — Mixed morphology: needle crystals + pinholes + grains
    "pbi2"   — Force bright particle / PbI2 detection only
    "pinhole"— Force dark pit / pinhole detection only
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from .preprocessing import to_grayscale, apply_clahe, denoise_image
from .postprocessing import clean_mask, extract_contours
from .metrics import calculate_defect_area_ratio, summarize_defects


# ── Mode profile dataclass ────────────────────────────────────────────────────

@dataclass
class _ModeProfile:
    clahe_clip: float = 3.0
    clahe_grid: int = 8
    gaussian_sigma: float = 1.0
    tophat_kernels: list = field(default_factory=lambda: [3, 5, 7])
    blackhat_kernels: list = field(default_factory=lambda: [5, 7, 11])
    bright_percentile_hi: float = 95.0
    bright_percentile_lo: float = 88.0
    dark_percentile: float = 12.0
    dark_percentile_strict: float = 6.0
    min_area_bright: int = 4
    max_area_bright: int = 2000
    min_area_dark: int = 15
    max_area_dark: int = 20000
    min_circularity_bright: float = 0.20
    min_circularity_dark: float = 0.30
    min_solidity: float = 0.40
    max_aspect_ratio: float = 5.0
    max_fill_ratio: float = 0.08
    conf_thresh: float = 0.30
    nms_iou: float = 0.40
    suppress_grain_boundaries: bool = False
    grain_boundary_canny_lo: int = 30
    grain_boundary_canny_hi: int = 80
    grain_boundary_dilate: int = 5
    detect_bright: bool = True
    detect_dark: bool = True
    detect_needles: bool = False
    needle_min_aspect: float = 2.5
    needle_min_area: int = 10


# ── Predefined profiles per mode ──────────────────────────────────────────────

_PROFILES: Dict[str, _ModeProfile] = {
    "pbi2": _ModeProfile(
        tophat_kernels=[3, 5, 7],
        detect_bright=True, detect_dark=False, detect_needles=True,
        min_area_bright=2, max_area_bright=3000,
        bright_percentile_hi=85.0, bright_percentile_lo=75.0,
        min_circularity_bright=0.05, suppress_grain_boundaries=False,
        conf_thresh=0.15,
    ),
    "pinhole": _ModeProfile(
        blackhat_kernels=[5, 7, 11, 15],
        detect_bright=False, detect_dark=True,
        min_area_dark=4, dark_percentile=12.0, dark_percentile_strict=6.0,
        min_circularity_dark=0.25, suppress_grain_boundaries=False,
    ),
    "3d": _ModeProfile(
        tophat_kernels=[3, 5], blackhat_kernels=[5, 7, 11],
        detect_bright=True, detect_dark=True, detect_needles=True,
        suppress_grain_boundaries=True, grain_boundary_dilate=7,
        min_area_bright=3, max_area_bright=1500, min_area_dark=10,
        bright_percentile_hi=95.0, bright_percentile_lo=88.0,
        dark_percentile=10.0, min_circularity_bright=0.15,
        min_circularity_dark=0.30, max_fill_ratio=0.06, conf_thresh=0.25,
    ),
    "3d_2d": _ModeProfile(
        tophat_kernels=[3, 5, 7], blackhat_kernels=[5, 7, 11],
        detect_bright=True, detect_dark=True, detect_needles=True,
        suppress_grain_boundaries=True, grain_boundary_dilate=5,
        min_area_bright=3, max_area_bright=2000, min_area_dark=8,
        bright_percentile_hi=94.0, bright_percentile_lo=87.0,
        dark_percentile=10.0, min_circularity_bright=0.12,
        min_circularity_dark=0.25, needle_min_aspect=2.0,
        needle_min_area=8, conf_thresh=0.25,
    ),
    "2d": _ModeProfile(
        tophat_kernels=[3, 5, 7, 9], blackhat_kernels=[7, 11, 15],
        detect_bright=True, detect_dark=True, detect_needles=False,
        suppress_grain_boundaries=False, min_area_bright=5,
        bright_percentile_hi=95.0, dark_percentile=15.0,
        min_circularity_bright=0.20, conf_thresh=0.30,
    ),
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _preprocess(gray: np.ndarray, p: _ModeProfile) -> np.ndarray:
    ksize = max(3, int(np.ceil(p.gaussian_sigma * 6)) | 1)
    denoised = cv2.GaussianBlur(gray, (ksize, ksize), p.gaussian_sigma)
    clahe = cv2.createCLAHE(clipLimit=p.clahe_clip,
                             tileGridSize=(p.clahe_grid, p.clahe_grid))
    return clahe.apply(denoised)


def _crop_sem_bar(gray: np.ndarray) -> Tuple[np.ndarray, int]:
    """Remove SEM metadata bar from the bottom of an image."""
    h, w = gray.shape
    scan = gray[int(h * 0.82):, :]
    row_medians = np.median(scan, axis=1)
    dark_rows = np.where(row_medians < 20)[0]
    if len(dark_rows) > 0:
        cut_y = max(int(h * 0.70), int(h * 0.82) + dark_rows[0] - 5)
    else:
        col_std = np.std(scan, axis=1)
        uniform_rows = np.where(col_std < 5)[0]
        cut_y = (max(int(h * 0.70), int(h * 0.82) + uniform_rows[0] - 3)
                 if len(uniform_rows) > 3 else h)
    return gray[:cut_y, :], cut_y


def _auto_detect_mode(gray: np.ndarray) -> str:
    """Select mode from image statistics."""
    h, w = gray.shape
    edges = cv2.Canny(gray, 30, 80)
    edge_density = float(np.sum(edges > 0)) / (h * w)
    gx = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
    grad_mean = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    bright_spot_density = float(np.sum(tophat > 40)) / (h * w)

    if edge_density > 0.08 and grad_mean > 20:
        return "3d"
    elif edge_density > 0.04:
        return "3d_2d"
    else:
        return "pbi2" if bright_spot_density > 0.008 else "2d"


def _build_grain_boundary_mask(enhanced: np.ndarray, p: _ModeProfile) -> np.ndarray:
    """
    Build a suppression mask that covers grain boundaries.

    Combines Canny edges (dilated) with high-gradient regions so that
    defect candidates overlapping grain boundaries are discarded.
    """
    edges = cv2.Canny(enhanced, p.grain_boundary_canny_lo, p.grain_boundary_canny_hi)
    k_dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (p.grain_boundary_dilate, p.grain_boundary_dilate))
    dilated = cv2.dilate(edges, k_dilate, iterations=1)

    gx = cv2.Sobel(enhanced.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(enhanced.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    high_grad = (grad_mag > np.percentile(grad_mag, 80)).astype(np.uint8) * 255
    high_grad = cv2.morphologyEx(high_grad, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    boundary_mask = cv2.bitwise_or(dilated, high_grad)
    return cv2.morphologyEx(boundary_mask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


def _contour_features(contour: np.ndarray, gray: np.ndarray) -> dict:
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    circ = 4.0 * np.pi * area / (perim ** 2) if perim > 1 else 0
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / (hull_area + 1e-6)
    x, y, w, h = cv2.boundingRect(contour)
    aspect = max(w, h) / (min(w, h) + 1e-6)

    mask_full = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask_full, [contour], -1, 255, -1)
    interior = gray[mask_full == 255]
    interior_mean = float(np.mean(interior)) if len(interior) > 0 else 0

    pad = max(w, h, 12)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
    roi = gray[y0:y1, x0:x1]
    roi_mask = mask_full[y0:y1, x0:x1]
    ext_pixels = roi[roi_mask == 0]
    ext_mean = float(np.mean(ext_pixels)) if len(ext_pixels) > 0 else 128
    contrast = float(np.clip(abs(ext_mean - interior_mean) / (ext_mean + 1e-6), 0, 1))

    return dict(area=area, circularity=circ, solidity=solidity,
                x=x, y=y, w=w, h=h, aspect=aspect,
                interior_mean=interior_mean, contrast=contrast)


def _detect_bright_mask(enhanced: np.ndarray, p: _ModeProfile) -> np.ndarray:
    combined = np.zeros_like(enhanced, dtype=np.float64)
    for ks in p.tophat_kernels:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
        combined = np.maximum(combined, tophat.astype(np.float64))
    combined_u8 = np.clip(combined, 0, 255).astype(np.uint8)

    # Smart Local Filtering: 
    # TopHat naturally subtracts the local background. If a pixel has a tophat response > 6, 
    # it is definitively a distinct "bump" or pebble compared to the immediate large cell it sits on.
    # No global percentiles needed!
    _, merged = cv2.threshold(combined_u8, 6, 255, cv2.THRESH_BINARY)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, k_open, iterations=1)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, k_close, iterations=1)

    return merged


def _detect_dark_mask(enhanced: np.ndarray, p: _ModeProfile) -> np.ndarray:
    dark_pct = float(np.percentile(enhanced, p.dark_percentile))
    _, binary = cv2.threshold(enhanced, int(max(dark_pct, 3)), 255, cv2.THRESH_BINARY_INV)

    if cv2.countNonZero(binary) / (binary.shape[0] * binary.shape[1]) > p.max_fill_ratio:
        dark_pct = float(np.percentile(enhanced, p.dark_percentile_strict))
        _, binary = cv2.threshold(enhanced, int(max(dark_pct, 3)), 255, cv2.THRESH_BINARY_INV)

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    _, micro = cv2.threshold(enhanced, int(max(np.percentile(enhanced, 3), 3)),
                              255, cv2.THRESH_BINARY_INV)
    micro = cv2.morphologyEx(micro, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return cv2.bitwise_or(binary, micro)


def _detect_needle_mask(enhanced: np.ndarray) -> np.ndarray:
    tophat_large = cv2.morphologyEx(
        enhanced, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
    bright_thresh = float(np.percentile(tophat_large, 92))
    _, mask = cv2.threshold(tophat_large, int(max(bright_thresh, 5)), 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)


def _filter_bright_contours(contours, gray, p):
    img_median = float(np.median(gray))
    kept = []
    for c in contours:
        f = _contour_features(c, gray)
        if not (p.min_area_bright <= f["area"] <= p.max_area_bright): continue
        # Relax circularity heavily for tiny irregular sand-like particles
        dyn_circ = p.min_circularity_bright
        if f["area"] <= 15:
            dyn_circ = 0.05
            
        if f["circularity"] < dyn_circ: continue
        if f["solidity"] < p.min_solidity: continue
        if f["aspect"] > p.max_aspect_ratio: continue
        
        # Only rely on LOCAL contrast. Ignore global image medians.
        if f["contrast"] < 0.03: continue
        conf = (0.40 * f["contrast"] + 0.25 * f["circularity"] +
                0.15 * f["solidity"] + 0.10 * float(np.clip(f["area"] / 500, 0, 1)) + 0.10)
        if float(np.clip(conf, 0, 1)) >= p.conf_thresh:
            kept.append(c)
    return kept


def _filter_dark_contours(contours, gray, p):
    img_median = float(np.median(gray))
    img_std = float(np.std(gray))
    dark_gate = img_median - 0.75 * img_std
    kept = []
    for c in contours:
        f = _contour_features(c, gray)
        if not (p.min_area_dark <= f["area"] <= p.max_area_dark): continue
        dyn_circ = p.min_circularity_dark
        if f["area"] > 500: dyn_circ = max(0.15, dyn_circ - 0.20)
        elif f["area"] > 200: dyn_circ = max(0.20, dyn_circ - 0.10)
        if f["circularity"] < dyn_circ: continue
        if f["solidity"] < p.min_solidity: continue
        if f["aspect"] > p.max_aspect_ratio: continue
        if f["interior_mean"] > dark_gate: continue
        if f["contrast"] < 0.10: continue
        conf = (0.40 * f["contrast"] + 0.25 * f["circularity"] +
                0.15 * f["solidity"] + 0.10 * float(np.clip(f["area"] / 2000, 0, 1)) + 0.10)
        if float(np.clip(conf, 0, 1)) >= p.conf_thresh:
            kept.append(c)
    return kept


def _filter_needle_contours(contours, gray, p):
    img_median = float(np.median(gray))
    kept = []
    for c in contours:
        f = _contour_features(c, gray)
        if f["area"] < p.needle_min_area or f["area"] > p.max_area_bright: continue
        if f["aspect"] < p.needle_min_aspect: continue
        if f["interior_mean"] < img_median: continue
        if f["contrast"] < 0.06: continue
        kept.append(c)
    return kept


def _nms_contours(contours: list, gray: np.ndarray, iou_thresh: float = 0.4) -> list:
    if not contours:
        return []
    scored = []
    for c in contours:
        f = _contour_features(c, gray)
        scored.append((f["contrast"], c, cv2.boundingRect(c)))
    scored.sort(key=lambda x: x[0], reverse=True)

    def _iou(a, b):
        xa, ya = max(a[0], b[0]), max(a[1], b[1])
        xb, yb = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        union = a[2]*a[3] + b[2]*b[3] - inter
        return inter / (union + 1e-6)

    keep = []
    for _, c, box in scored:
        if all(_iou(box, kb) < iou_thresh for _, _, kb in keep):
            keep.append((0, c, box))
    return [c for _, c, _ in keep]


# ── Public API ────────────────────────────────────────────────────────────────

def detect_defects(
    image: np.ndarray,
    mode: str = "auto",
    sensitivity: float = 1.5,
    min_area: float = 20,
    return_intermediate: bool = False,
) -> Dict:
    """
    Detect defects in a SEM or microstructure image using a classical
    adaptive OpenCV pipeline.

    Originally developed for perovskite solar-cell SEM pinhole and
    PbI2 bright-particle detection.

    Args:
        image:              Grayscale or BGR image (uint8).
        mode:               Detection mode:
                              - "auto"    — auto-select morphology from statistics
                              - "2d"      — 2D perovskite morphology
                              - "3d"      — 3D perovskite (grain boundary suppression)
                              - "3d_2d"   — Mixed 2D-3D morphology
                              - "pbi2"    — PbI2 bright particles only
                              - "pinhole" — Dark pit / pinhole detection only
        sensitivity:        Sensitivity multiplier (higher = more detections).
                            Currently used as a hint for future extensions.
        min_area:           Minimum contour area (pixels) to count as a defect.
        return_intermediate:If True, include intermediate pipeline images in
                            the result dict under the key "intermediates".

    Returns:
        dict with keys:
          - mask (np.ndarray):           Final binary defect mask.
          - enhanced (np.ndarray):       CLAHE-enhanced grayscale image.
          - defect_count (int):          Number of detected defects.
          - defect_area_ratio (float):   Fraction of image covered by defects.
          - contours (list):             Filtered contour list.
          - mode (str):                  Mode actually used.
          - intermediates (dict):        Present only when return_intermediate=True.

    Raises:
        ValueError: If image is None or mode is unrecognised.
    """
    if image is None:
        raise ValueError("image cannot be None.")

    valid_modes = {"auto", "2d", "3d", "3d_2d", "pbi2", "pinhole"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'.")

    # ── Grayscale conversion + SEM bar crop ──────────────────────────────────
    gray = to_grayscale(image)
    gray, _ = _crop_sem_bar(gray)

    # ── Mode resolution ──────────────────────────────────────────────────────
    resolved_mode = _auto_detect_mode(gray) if mode == "auto" else mode
    p = _PROFILES[resolved_mode]

    # ── Preprocessing ────────────────────────────────────────────────────────
    enhanced = _preprocess(gray, p)

    intermediates: Dict[str, np.ndarray] = {}
    if return_intermediate:
        intermediates["02_enhanced"] = enhanced.copy()

    # ── Grain boundary suppression mask (3D modes only) ──────────────────────
    grain_mask = None
    if p.suppress_grain_boundaries:
        grain_mask = _build_grain_boundary_mask(enhanced, p)
        if return_intermediate:
            intermediates["03_grain_boundary_mask"] = grain_mask.copy()

    def _apply_grain_suppression(m):
        if grain_mask is not None:
            return cv2.bitwise_and(m, cv2.bitwise_not(grain_mask))
        return m

    # Each entry: (contour, class_id, defect_type)
    tagged: List[tuple] = []
    final_mask = np.zeros_like(gray, dtype=np.uint8)

    # ── Bright particle detection (PbI2) ─────────────────────────────────────
    if p.detect_bright:
        bright_mask = _detect_bright_mask(enhanced, p)
        bright_mask = _apply_grain_suppression(bright_mask)
        if return_intermediate:
            intermediates["04_bright_mask"] = bright_mask.copy()
        raw_cnts, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bright_cnts = _filter_bright_contours(raw_cnts, gray, p)
        tagged.extend((c, 0, "pbi2_bright") for c in bright_cnts)
        cv2.drawContours(final_mask, bright_cnts, -1, 255, -1)

    # ── Dark pit detection (Pinholes) ─────────────────────────────────────────
    if p.detect_dark:
        dark_mask = _detect_dark_mask(enhanced, p)
        dark_mask = _apply_grain_suppression(dark_mask)
        if return_intermediate:
            intermediates["05_dark_mask"] = dark_mask.copy()
        raw_cnts, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dark_cnts = _filter_dark_contours(raw_cnts, gray, p)
        for c in dark_cnts:
            f = _contour_features(c, gray)
            cls = (1, "pinhole_small") if f["area"] <= 300 else (2, "pinhole_large")
            tagged.append((c, cls[0], cls[1]))
        cv2.drawContours(final_mask, dark_cnts, -1, 255, -1)

    # ── Needle crystal detection (PbI2 needles) ───────────────────────────────
    if p.detect_needles:
        needle_mask = _detect_needle_mask(enhanced)
        needle_mask = _apply_grain_suppression(needle_mask)
        if return_intermediate:
            intermediates["06_needle_mask"] = needle_mask.copy()
        raw_cnts, _ = cv2.findContours(needle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        needle_cnts = _filter_needle_contours(raw_cnts, gray, p)
        tagged.extend((c, 3, "pbi2_needle") for c in needle_cnts)
        cv2.drawContours(final_mask, needle_cnts, -1, 255, -1)

    # ── NMS + min_area filtering ──────────────────────────────────────────────
    # Sort by contrast so NMS keeps best candidate
    all_contours_only = [t[0] for t in tagged]
    kept_contours = _nms_contours(all_contours_only, gray, p.nms_iou)
    kept_set = {id(c) for c in kept_contours}

    # Re-filter tagged list preserving class info
    tagged = [(c, cid, dtype) for c, cid, dtype in tagged
              if id(c) in kept_set and cv2.contourArea(c) >= min_area]

    # Rebuild final mask from surviving contours
    final_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(final_mask, [t[0] for t in tagged], -1, 255, -1)

    # Build structured detections list for YOLO export and visualization
    detections = []
    for contour, class_id, defect_type in tagged:
        x, y, w, h = cv2.boundingRect(contour)
        detections.append({
            "contour": contour,
            "class_id": class_id,
            "defect_type": defect_type,
            "bbox": (x, y, w, h),
        })

    result = {
        "mask": final_mask,
        "enhanced": enhanced,
        "defect_count": len(detections),
        "defect_area_ratio": calculate_defect_area_ratio(final_mask),
        "contours": [t[0] for t in tagged],
        "detections": detections,   # full structured list with class info
        "mode": resolved_mode,
    }
    if return_intermediate:
        result["intermediates"] = intermediates

    return result


def adaptive_defect_filter(
    image: np.ndarray,
    mode: str = "auto",
    min_area: float = 20,
    return_intermediate: bool = False,
) -> Dict:
    """
    Convenience wrapper around ``detect_defects`` using default sensitivity.

    Suitable for use in pipelines where a single callable filter is needed
    without manually specifying every parameter.

    Args:
        image:              Input image (grayscale or BGR).
        mode:               Detection mode (see ``detect_defects``).
        min_area:           Minimum defect area in pixels.
        return_intermediate:Whether to include intermediate pipeline images.

    Returns:
        Same dict structure as ``detect_defects``.
    """
    return detect_defects(
        image,
        mode=mode,
        sensitivity=1.5,
        min_area=min_area,
        return_intermediate=return_intermediate,
    )
