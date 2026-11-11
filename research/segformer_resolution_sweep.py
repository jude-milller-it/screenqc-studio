# -*- coding: utf-8 -*-
"""Evaluate SegFormer input resolutions with the SAME C-group pipeline.

Place beside segformer_ablation_9groups.py / segformer_infer.py.
Run: D:\\AI\\miniForge1\\python.exe segformer_resolution_sweep.py

- Never modifies weights, training, or other scripts.
- Uses val.txt and pretrained SegFormer MiT-B2 checkpoint from the original project.
- Uses the SAME ROI + class confidence filtering as the tested C-group.
- No morphological/area postprocessing and no class voting.
- Writes results to resolution_sweep/summary.csv, plus confusion matrices.
- The values are VALIDATION metrics; keep an untouched split for final evaluation.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from segformer_ablation_9groups import (
    ROOT, DATASET, WEIGHTS, SegFormerInference, cvtColor,
    read_gt, make_mask, cm_update, scores,
)

# Shapes are [HEIGHT, WIDTH], and are divisible by 32 (MiT encoder requirement).
# 512x512 reproduces earlier results; 384x672 is the previous best.
VARIANTS = [
    ("baseline_512x512", 512, 512),
    ("previous_best_384x672", 384, 672),
    ("aspect_384x704", 384, 704),
    ("hires_448x800", 448, 800),
    ("hires_512x896", 512, 896),
    ("hires_576x1024", 576, 1024),
]
OUTPUT_DIR = ROOT / "resolution_sweep"


def main() -> None:
    split = DATASET / "ImageSets" / "Segmentation" / "val.txt"
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {WEIGHTS}")
    if not split.is_file():
        raise FileNotFoundError(f"Missing validation split: {split}")
    names = [s.strip() for s in split.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    if not names:
        raise RuntimeError("Validation split is empty")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device_cuda = bool(torch.cuda.is_available())
    print(f"GPU available: {device_cuda}; images: {len(names)}", flush=True)
    model = SegFormerInference(
        num_classes=4,
        variant="mit_b2",
        model_path=str(WEIGHTS),
        input_shape=[512, 512],
        cuda=device_cuda,
        fast_mode=True,
        use_roi_mask=True,
        use_postprocess=False,
    )
    # Evaluate each resolution end-to-end, so OOM can skip only that resolution.
    rows = []
    for variant, height, width in VARIANTS:
        print(f"\n=== {variant} [{height}, {width}] ===", flush=True)
        model.input_shape = [height, width]
        cm = np.zeros((4, 4), dtype=np.int64)
        total_model_sec = 0.0
        total_filter_sec = 0.0
        status, error = "ok", ""
        completed = 0
        try:
            for i, image_id in enumerate(names, 1):
                image_path = DATASET / "JPEGImages" / f"{image_id}.jpg"
                label_path = DATASET / "SegmentationClass" / f"{image_id}.png"
                if not image_path.is_file() or not label_path.is_file():
                    raise FileNotFoundError(f"Missing input or label for {image_id}")
                with Image.open(image_path) as im:
                    rgb = np.asarray(cvtColor(im), dtype=np.uint8)
                gt = read_gt(label_path, rgb.shape[:2])
                # ROI detection is the same for all sizes and happens at original resolution.
                roi = model._detect_screen_roi(rgb)
                if device_cuda:
                    torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    prob = model._fast_predict(rgb)
                if device_cuda:
                    torch.cuda.synchronize()
                total_model_sec += time.perf_counter() - start
                start = time.perf_counter()
                pred = make_mask(model, rgb, prob, (True, True, False, False), roi)
                total_filter_sec += time.perf_counter() - start
                cm_update(cm, gt, pred)
                completed += 1
                if i == 1 or i % 20 == 0 or i == len(names):
                    print(f"  {i}/{len(names)}  {image_id}", flush=True)
        except torch.cuda.OutOfMemoryError as ex:
            status, error = "CUDA_OOM", str(ex).splitlines()[0][:300]
            print(f"  Skipped {variant}: GPU out of memory; other variants continue.", flush=True)
            if device_cuda:
                torch.cuda.empty_cache()
        if status == "ok":
            iou, recall, precision = scores(cm)
            metrics = {
                "mIoU_pct": round(float(iou.mean() * 100), 3),
                "defect_mIoU_pct": round(float(iou[1:].mean() * 100), 3),
                "background_IoU_pct": round(float(iou[0] * 100), 3),
                "oil_IoU_pct": round(float(iou[1] * 100), 3),
                "stain_IoU_pct": round(float(iou[2] * 100), 3),
                "scratch_IoU_pct": round(float(iou[3] * 100), 3),
                "mean_inference_ms": round(total_model_sec / completed * 1000, 2),
                "mean_roi_conf_ms": round(total_filter_sec / completed * 1000, 2),
                "mean_precision_pct": round(float(precision.mean() * 100), 3),
                "mean_recall_pct": round(float(recall.mean() * 100), 3),
            }
            np.savetxt(OUTPUT_DIR / f"{variant}_confusion.csv", cm, fmt="%d", delimiter=",")
            print(f"  mIoU={metrics['mIoU_pct']:.3f}% "
                  f"oil/stain/scratch={metrics['oil_IoU_pct']}/"
                  f"{metrics['stain_IoU_pct']}/{metrics['scratch_IoU_pct']} "
                  f"inference={metrics['mean_inference_ms']} ms/image", flush=True)
        else:
            metrics = {key: "" for key in (
                "mIoU_pct", "defect_mIoU_pct", "background_IoU_pct", "oil_IoU_pct",
                "stain_IoU_pct", "scratch_IoU_pct", "mean_inference_ms",
                "mean_roi_conf_ms", "mean_precision_pct", "mean_recall_pct")}
        row = {"variant": variant, "input_height": height, "input_width": width,
               "status": status, "n_images": completed if status == "ok" else 0,
               **metrics, "error": error}
        rows.append(row)
        # Save after each resolution to preserve partial results if interrupted.
        with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    baseline = next((r for r in rows if r["variant"] == "baseline_512x512" and r["status"] == "ok"), None)
    previous = next((r for r in rows if r["variant"] == "previous_best_384x672" and r["status"] == "ok"), None)
    if baseline and abs(baseline["mIoU_pct"] - 72.167) > 0.3:
        print("WARNING: baseline differs from previous 72.167%; verify inputs and model.")
    if previous and abs(previous["mIoU_pct"] - 73.202) > 0.3:
        print("WARNING: 384x672 differs from previous 73.202%; verify inputs and model.")
    successful = [r for r in rows if r["status"] == "ok"]
    if successful:
        best = max(successful, key=lambda r: r["mIoU_pct"])
        print(f"\nBest VAL mIoU: {best['variant']} = {best['mIoU_pct']:.3f}%")
    (OUTPUT_DIR / "run_info.json").write_text(json.dumps({
        "dataset_split": str(split), "checkpoint": str(WEIGHTS),
        "samples": len(names), "settings": "fast_mode + ROI + confidence; no postprocess or vote",
        "warning": "These are validation results; test chosen configuration on independent held-out data.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results: {OUTPUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
