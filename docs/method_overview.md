# Method Overview — MicroDefectCV

This document describes the internal pipeline of the `detect_defects` function.

---

## 1. Grayscale Conversion

If the input image is colour (BGR), it is first converted to grayscale using `cv2.cvtColor`. All subsequent processing operates on a single-channel uint8 image.

## 2. SEM Metadata Bar Removal

SEM images frequently contain a bottom annotation bar (scale bar, metadata text). The pipeline scans the bottom 18 % of the image for rows whose median intensity is below 20 or whose column standard deviation is below 5. The first such row defines the crop point.

## 3. Mode / Profile Selection

The algorithm supports six named profiles:

| Mode | Target defects | Grain boundary suppression | Needle detection |
|---|---|---|---|
| `pbi2` | Bright PbI₂ particles | No | Yes |
| `pinhole` | Dark pinholes | No | No |
| `2d` | Both | No | No |
| `3d` | Both | **Yes** | Yes |
| `3d_2d` | Both | **Yes** | Yes |
| `auto` | Determined from image stats | Conditional | Conditional |

In `auto` mode the following image statistics are computed to select a profile:
- Canny edge density
- Mean Sobel gradient magnitude
- Top-hat bright spot density

## 4. Preprocessing

1. **Gaussian blur** — reduces high-frequency noise before CLAHE.
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** — boosts local contrast so that subtle defect boundaries become visible. Clip limit and tile size are tuned per mode.

## 5. Grain Boundary Suppression (3D / 3D-2D modes)

In 3D perovskite images, grain boundaries cast pseudo-defect shadows. A suppression mask is constructed by:
1. Canny edge detection on the enhanced image.
2. Dilation of edges to form thick boundary zones.
3. High-gradient pixel thresholding (top 20 % of Sobel magnitude).
4. Union of both masks with morphological closing.

Any candidate defect overlapping this mask is discarded.

## 6. Bright Particle Detection (PbI₂)

1. Multi-scale **Top-Hat transform** using elliptical kernels (sizes vary per mode). This isolates bright regions smaller than the kernel size.
2. Dual percentile thresholding (high and low) on the combined Top-Hat response.
3. A fill-ratio guard prevents false over-segmentation.
4. Morphological open + close to clean the mask.

## 7. Dark Pit Detection (Pinholes)

1. **Percentile thresholding** on the enhanced image: pixels below the N-th percentile (dark_percentile) are marked.
2. A fill-ratio guard triggers a stricter percentile if too much of the image is flagged.
3. A separate micro-threshold at the 3rd percentile captures very small pinholes.
4. Morphological close + open to smooth boundaries.

## 8. Needle Crystal Detection (PbI₂ Needles)

1. Large rectangular **Top-Hat transform** (11×11 rect kernel) isolates elongated bright structures.
2. Elongated morphological closing (3×1 kernel) bridges gap artefacts in long needles.
3. Contours are filtered by minimum **aspect ratio** (length/width ≥ 2.5 by default).

## 9. Contour Filtering

For each candidate contour, shape features are computed:
- **Area** — rejects tiny noise and oversized regions.
- **Circularity** — `4πA/P²` (dynamically relaxed for large dark pits).
- **Solidity** — `area / convex_hull_area` (rejects irregular shapes).
- **Aspect ratio** — rejects very elongated blobs in bright/dark paths.
- **Interior vs exterior mean intensity** — ensures correct polarity (bright/dark).
- **Contrast** — `|ext_mean − int_mean| / ext_mean`.

A confidence score is computed from these features and filtered against a threshold.

## 10. Non-Maximum Suppression (NMS)

Overlapping detections are deduplicated using IoU-based NMS (intersection-over-union threshold 0.4 by default), retaining the highest-confidence detection.

## 11. Output

The function returns a dictionary containing:
- `mask` — binary uint8 mask
- `enhanced` — CLAHE image
- `defect_count` — integer
- `defect_area_ratio` — float in [0, 1]
- `contours` — list of OpenCV contours
- `mode` — resolved mode string
- `intermediates` — per-stage images (when `return_intermediate=True`)
