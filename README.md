# MicroDefectCV

**Adaptive OpenCV-based defect enhancement and segmentation for SEM and microstructure images.**

MicroDefectCV is a lightweight classical computer vision library originally developed for perovskite solar-cell SEM pinhole and PbI₂ bright-particle detection. It provides a reusable, mode-aware pipeline that can be applied to a wide range of microstructure images without deep learning or labelled data.

> This package provides a lightweight classical computer vision **baseline** for defect enhancement and segmentation. It does not claim to replace deep learning methods on large annotated datasets.

---

## Features

- 🔬 **Six detection modes** covering different perovskite morphologies and defect types
- 🧠 **Auto mode** that classifies image morphology from statistics alone
- 🧩 **Grain boundary suppression** for 3D and mixed-morphology images
- 📐 **Needle crystal detection** for elongated PbI₂ excess structures
- 📊 **Defect statistics** (count, area, area ratio) in one call
- 🖼️ **Intermediate stage images** for debugging and research
- ✅ **Zero deep learning** — pure OpenCV + NumPy, runs on CPU
- 📦 **Pip-installable** clean package structure

---

## Installation

```bash
cd microdefectcv_release
pip install -e .
```

Or install from source after cloning the repository:

```bash
git clone https://github.com/yourusername/microdefectcv.git
cd microdefectcv
pip install -e .
```

---

## Quick Start

```python
import cv2
from microdefectcv import detect_defects

image = cv2.imread("sample_images/sem_image.png")

result = detect_defects(
    image,
    mode="auto",       # auto-selects morphology from image statistics
    min_area=20,
    return_intermediate=True
)

print(f"Defects found  : {result['defect_count']}")
print(f"Area ratio     : {result['defect_area_ratio']:.4f}")

mask     = result["mask"]         # binary defect mask
enhanced = result["enhanced"]     # CLAHE-enhanced image
contours = result["contours"]     # list of OpenCV contours
```

---

## Detection Modes

| Mode | Target Defects | Image Morphology |
|---|---|---|
| `auto` | All | Auto-detected from statistics |
| `pbi2` | PbI₂ bright particles + needles | Any |
| `pinhole` | Dark pinholes (small + large) | Any |
| `2d` | Both | 2D perovskite (flat morphology) |
| `3d` | Both + needles | 3D perovskite (grain suppression active) |
| `3d_2d` | Both + needles | Mixed 2D-3D morphology |

---

## Method Pipeline

```
Input Image
    │
    ├─ Grayscale conversion (if BGR)
    ├─ SEM metadata bar removal
    ├─ Mode selection (auto or user-specified)
    ├─ Gaussian denoising + CLAHE
    │
    ├─ [3D / 3D-2D only] Grain boundary suppression mask
    │
    ├─ Bright particle detection (Top-Hat + dual percentile threshold)
    ├─ Dark pit detection       (Percentile threshold + micro-threshold)
    ├─ Needle crystal detection (Rectangular Top-Hat + aspect ratio filter)
    │
    ├─ Shape feature filtering (area, circularity, solidity, contrast)
    ├─ Non-maximum suppression (IoU-based)
    │
    └─ Output: mask, enhanced, contours, defect_count, defect_area_ratio
```

See [`docs/method_overview.md`](docs/method_overview.md) for full technical details.

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `np.ndarray` | — | Grayscale or BGR uint8 image |
| `mode` | `str` | `"auto"` | Detection mode (see table above) |
| `sensitivity` | `float` | `1.5` | Sensitivity hint (reserved for tuning) |
| `min_area` | `float` | `20` | Minimum defect area in pixels |
| `return_intermediate` | `bool` | `False` | Include per-stage pipeline images |

---

## Quick Start Guide

### Method 1: Command Line (Single Image)
Process a single image and generate a pipeline grid + YOLO annotations in the `outputs/` folder.
```bash
# Auto-detect mode
python examples/demo_perovskite_sem.py path/to/image.jpg

# Force PbI2 mode and drop minimum area to catch tiny sand-like particles
python examples/demo_perovskite_sem.py path/to/image.jpg --mode pbi2 --min-area 3
```

### Method 2: Batch Processing (PowerShell)
Process an entire folder of images automatically:
```powershell
Get-ChildItem -Path "path\to\folder" -Filter *.jpg | ForEach-Object {
    python examples/demo_perovskite_sem.py $_.FullName --mode auto
}
```

### Method 3: Python API
Import and use the standalone pip package directly in your own scripts:
```python
import cv2
from microdefectcv import detect_defects
from microdefectcv.visualization import save_yolo_annotations

image = cv2.imread("path/to/image.jpg")
result = detect_defects(image, mode="auto", min_area=20)

print(f"Found {result['defect_count']} defects!")
save_yolo_annotations(result["detections"], image.shape, "outputs/labels.txt")
```

---

## Running Tests

```bash
cd microdefectcv_release
pytest
```

---

## Use Cases

- **Perovskite solar-cell SEM** — pinhole and PbI₂ crystal detection
- **Thin-film defect inspection** — dark voids and bright particle segmentation
- **Microstructure void detection** — general SEM / optical microscopy
- **Coating and surface QC** — surface dark defect segmentation
- **Classical CV baseline** — compare against DL models on annotated datasets

---

## Benchmark Plan

See [`docs/benchmark_plan.md`](docs/benchmark_plan.md) for a planned evaluation comparing MicroDefectCV against:
- Global threshold, Otsu, CLAHE+Otsu
- Canny edge detection, Watershed
- YOLOv8 / Faster R-CNN (when annotations are available)

Metrics: Precision, Recall, F1, IoU, Dice, processing time.

---

## Citation

If you use MicroDefectCV in academic work, please cite:

```
@software{microdefectcv2025,
  title  = {MicroDefectCV: Adaptive OpenCV-based Defect Segmentation for SEM Images},
  author = {[Sahil Soni]},
  year   = {2025},
  url    = {https://github.com/Sahilsonii/microdefectcv}
}
```

---

## Roadmap

- [ ] Annotated SEM benchmark dataset
- [ ] `scripts/evaluate.py` evaluation script
- [ ] Hyperparameter search / sensitivity analysis
- [ ] Optional integration with OpenCV-contrib

