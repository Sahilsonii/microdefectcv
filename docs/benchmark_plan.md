# Benchmark Plan — MicroDefectCV

This document outlines the planned evaluation of MicroDefectCV against
classical and deep-learning baseline methods.

> **Note**: MicroDefectCV is a classical computer vision baseline.
> It is not designed to outperform deep learning methods on large annotated datasets.
> The goal of this benchmark is honest characterisation of when and where it is useful.

---

## Methods to Compare

| # | Method | Category |
|---|---|---|
| 1 | **MicroDefectCV** (this package) | Classical adaptive |
| 2 | Global threshold (fixed intensity) | Classical |
| 3 | Otsu thresholding | Classical |
| 4 | CLAHE + Otsu | Classical |
| 5 | Canny edge detection | Classical |
| 6 | Watershed segmentation | Classical |
| 7 | YOLOv8 (if annotations available) | Deep learning |
| 8 | Faster R-CNN (if annotations available) | Deep learning |

---

## Suggested Metrics

### Pixel-level (requires ground truth mask)
| Metric | Description |
|---|---|
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1-score** | Harmonic mean of precision and recall |
| **IoU (Jaccard)** | Intersection / Union |
| **Dice coefficient** | 2·TP / (2·TP + FP + FN) |
| **Defect area error** | `|predicted_area − gt_area| / gt_area` |

### Object-level (requires bounding box annotations)
| Metric | Description |
|---|---|
| **Detection rate** | % of ground truth defects found |
| **False positive rate** | FP per image |
| **mAP@0.5** | Mean average precision at IoU 0.5 |

### Efficiency
| Metric | Description |
|---|---|
| **Processing time (ms)** | Per image, measured on CPU |
| **Memory usage (MB)** | Peak RAM during inference |

---

## Dataset Requirements

- Ground truth binary masks or bounding box annotations.
- At least 50 images per morphology category (2D, 3D, 3D-2D).
- Split: train (for DL baselines) / test (for all methods).

---

## Evaluation Protocol

1. Run each method on the test set with default parameters.
2. For classical methods, sweep key parameters (e.g. threshold values) and report the best result.
3. Compute all metrics above.
4. Report results per morphology category (2D / 3D / mixed) separately.

---

## Expected Outcomes

- MicroDefectCV is expected to perform well on **low-data** and **zero-shot** scenarios.
- Deep learning methods will likely outperform on large annotated datasets.
- Classical CLAHE + Otsu may be competitive but will struggle with:
  - Variable background intensities.
  - Grain boundary false positives in 3D morphologies.
  - Elongated needle-shaped particles.

---

## TODO

- [ ] Collect and annotate a labelled SEM dataset.
- [ ] Implement evaluation script `scripts/evaluate.py`.
- [ ] Add results table to README when benchmarks are complete.
