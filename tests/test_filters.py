"""
test_filters.py — MicroDefectCV
================================
Basic pytest tests using synthetic images.
"""

import numpy as np
import pytest
import cv2

from microdefectcv import detect_defects, adaptive_defect_filter


def make_clean_image(h=128, w=128, intensity=180):
    """Create a uniform gray image with no defects."""
    return np.full((h, w), intensity, dtype=np.uint8)


def make_defect_image_dark(h=128, w=128):
    """Gray image with artificial dark circular pinholes."""
    img = np.full((h, w), 180, dtype=np.uint8)
    cv2.circle(img, (32, 32), 8, 30, -1)
    cv2.circle(img, (96, 96), 6, 20, -1)
    cv2.circle(img, (64, 32), 5, 25, -1)
    return img


def make_defect_image_bright(h=256, w=256):
    """Dark gray image with artificial bright circular PbI2 particles."""
    rng = np.random.default_rng(42)
    # Noisy background similar to real SEM grain texture
    img = rng.integers(60, 100, size=(h, w), dtype=np.uint8)
    # Large, high-contrast bright blobs to exceed contrast threshold
    cv2.circle(img, (64, 64), 14, 230, -1)
    cv2.circle(img, (180, 140), 10, 245, -1)
    cv2.circle(img, (100, 200), 12, 220, -1)
    return img


# ── detect_defects API tests ──────────────────────────────────────────────────

class TestDetectDefectsAPI:
    def test_returns_dict(self):
        img = make_defect_image_dark()
        result = detect_defects(img, mode="pinhole")
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        img = make_defect_image_dark()
        result = detect_defects(img, mode="pinhole")
        for key in ("mask", "enhanced", "defect_count", "defect_area_ratio", "contours", "mode"):
            assert key in result, f"Missing key: {key}"

    def test_mask_shape_matches_input(self):
        img = make_defect_image_dark()
        result = detect_defects(img, mode="pinhole")
        # The pipeline crops the SEM metadata bar, so mask height may be <= img height
        mh, mw = result["mask"].shape
        ih, iw = img.shape
        assert mw == iw, "mask width must match input width"
        assert mh <= ih, "mask height must be <= input height after SEM bar crop"

    def test_defect_count_type(self):
        img = make_defect_image_dark()
        result = detect_defects(img, mode="pinhole")
        assert isinstance(result["defect_count"], int)

    def test_defect_area_ratio_range(self):
        img = make_defect_image_dark()
        result = detect_defects(img, mode="pinhole")
        assert 0.0 <= result["defect_area_ratio"] <= 1.0

    def test_return_intermediate_keys(self):
        img = make_defect_image_dark()
        result = detect_defects(img, mode="pinhole", return_intermediate=True)
        assert "intermediates" in result
        assert isinstance(result["intermediates"], dict)


# ── Defect detection on synthetic data ───────────────────────────────────────

class TestDetectionSensitivity:
    def test_detects_dark_defects(self):
        img = make_defect_image_dark()
        result = detect_defects(img, mode="pinhole", min_area=5)
        assert result["defect_count"] > 0, "Expected at least one pinhole detected."

    def test_detects_bright_defects(self):
        img = make_defect_image_bright()
        result = detect_defects(img, mode="pbi2", min_area=5)
        assert result["defect_count"] > 0, "Expected at least one PbI2 particle detected."

    def test_clean_image_low_count(self):
        img = make_clean_image()
        result = detect_defects(img, mode="pinhole")
        # A fully uniform image should produce very few or zero detections
        assert result["defect_count"] < 5, "Unexpected high defect count on clean image."


# ── Mode support tests ────────────────────────────────────────────────────────

class TestModes:
    @pytest.mark.parametrize("mode", ["auto", "2d", "3d", "3d_2d", "pbi2", "pinhole"])
    def test_all_modes_run(self, mode):
        img = make_defect_image_dark()
        result = detect_defects(img, mode=mode)
        assert "defect_count" in result

    def test_invalid_mode_raises(self):
        img = make_defect_image_dark()
        with pytest.raises(ValueError):
            detect_defects(img, mode="invalid_mode")

    def test_none_image_raises(self):
        with pytest.raises(ValueError):
            detect_defects(None)


# ── adaptive_defect_filter wrapper ────────────────────────────────────────────

class TestAdaptiveFilter:
    def test_returns_same_keys(self):
        img = make_defect_image_dark()
        result = adaptive_defect_filter(img, mode="pinhole")
        for key in ("mask", "enhanced", "defect_count", "defect_area_ratio", "contours"):
            assert key in result

    def test_bgr_input_accepted(self):
        gray = make_defect_image_dark()
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result = adaptive_defect_filter(bgr, mode="pinhole")
        assert isinstance(result, dict)
