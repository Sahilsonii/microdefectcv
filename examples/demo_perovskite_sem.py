"""
demo_perovskite_sem.py — MicroDefectCV example
===============================================
Run defect detection on a SEM image and save outputs.

Usage:
    python examples/demo_perovskite_sem.py path/to/image.png
    python examples/demo_perovskite_sem.py path/to/image.png --mode 3d
    python examples/demo_perovskite_sem.py path/to/image.png --mode pbi2 --min-area 10
"""

import argparse
import os
import sys
import cv2

# Allow running from the repo root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from microdefectcv import detect_defects
from microdefectcv.postprocessing import draw_defect_contours
from microdefectcv.visualization import overlay_mask, save_result_grid, save_yolo_annotations


def parse_args():
    parser = argparse.ArgumentParser(
        description="MicroDefectCV — SEM defect detection demo"
    )
    parser.add_argument("image_path", help="Path to input SEM image (grayscale or BGR)")
    parser.add_argument(
        "--mode", default="auto",
        choices=["auto", "2d", "3d", "3d_2d", "pbi2", "pinhole"],
        help="Detection mode (default: auto)",
    )
    parser.add_argument("--min-area", type=float, default=20,
                        help="Minimum defect area in pixels (default: 20)")
    parser.add_argument("--output-dir", default="outputs",
                        help="Directory to save result images (default: outputs/)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load image ────────────────────────────────────────────────────────────
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"[ERROR] Could not load image: {args.image_path}")
        sys.exit(1)

    print(f"[INFO] Loaded image: {args.image_path}  shape={image.shape}")
    print(f"[INFO] Running detect_defects(mode='{args.mode}', min_area={args.min_area})")

    # ── Run filter ────────────────────────────────────────────────────────────
    result = detect_defects(
        image,
        mode=args.mode,
        min_area=args.min_area,
        return_intermediate=True,
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'-'*40}")
    print(f"  Mode resolved   : {result['mode']}")
    print(f"  Defect count    : {result['defect_count']}")
    print(f"  Defect area ratio: {result['defect_area_ratio']:.4f}  "
          f"({result['defect_area_ratio']*100:.2f}%)")
    print(f"{'-'*40}\n")

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image_path))[0]

    # 1. CLAHE-enhanced image
    enhanced_path = os.path.join(args.output_dir, f"{base}_enhanced.png")
    cv2.imwrite(enhanced_path, result["enhanced"])
    print(f"[SAVED] Enhanced image  -> {enhanced_path}")

    # 2. Binary defect mask
    mask_path = os.path.join(args.output_dir, f"{base}_mask.png")
    cv2.imwrite(mask_path, result["mask"])
    print(f"[SAVED] Defect mask     -> {mask_path}")

    # Crop original image to match the mask size (SEM bar removed)
    mh = result["mask"].shape[0]
    image_cropped = image[:mh, :]

    # 3. Colour overlay
    overlay = overlay_mask(image_cropped, result["mask"])
    overlay = draw_defect_contours(overlay, result["contours"])
    overlay_path = os.path.join(args.output_dir, f"{base}_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"[SAVED] Overlay image   -> {overlay_path}")

    # 4. Full pipeline grid
    grid_path = os.path.join(args.output_dir, f"{base}_pipeline_grid.png")
    save_result_grid(image_cropped, result, grid_path)
    print(f"[SAVED] Pipeline grid   -> {grid_path}")

    # 5. YOLO Annotations
    if "detections" in result:
        txt_path = os.path.join(args.output_dir, f"{base}.txt")
        save_yolo_annotations(result["detections"], image_cropped.shape, txt_path)
        print(f"[SAVED] YOLO labels     -> {txt_path}")


if __name__ == "__main__":
    main()
