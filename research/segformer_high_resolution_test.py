# -*- coding: utf-8 -*-
r"""High-resolution SegFormer evaluation, same C-group pipeline.

Put this file beside segformer_infer.py in the DeepLabV3 project folder.
Run: D:\AI\miniForge1\python.exe segformer_high_resolution_test.py
No need for previous ablation/sweep scripts. Does not modify model or weights.
"""
from __future__ import annotations

import csv
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from segformer_infer import SegFormerInference
from utils.utils import cvtColor

ROOT = Path(__file__).resolve().parent
DATASET = Path(r"D:\AI\dataset\MobilePhoneScreenSurfaceDefectSegmentation\MSD")
WEIGHTS = ROOT / "logs_segformer_mit_b2" / "best_epoch_weights.pth"
OUTPUT = ROOT / "resolution_high_test"
# H x W; first 3 sizes only. Increase later if GPU headroom is sufficient.
VARIANTS = [("high_640x1152", 640, 1152),
            ("high_704x1248", 704, 1248),
            ("high_768x1344", 768, 1344)]


def read_gt(path, expected_hw):
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim != 2:
        raise ValueError(f"GT must be single channel: {path}, shape={arr.shape}")
    values = set(np.unique(arr).tolist())
    if values.issubset({0, 1, 2, 3, 255}):
        gt = arr.astype(np.uint8)
    elif values.issubset({0, 38, 75, 113, 255}):
        gt = np.zeros(arr.shape, dtype=np.uint8)
        for raw, cls in ((38, 1), (75, 2), (113, 3), (255, 255)):
            gt[arr == raw] = cls
    else:
        raise ValueError(f"Unexpected GT values: {sorted(values)} in {path}")
    if gt.shape != expected_hw:
        raise ValueError(f"GT dimensions {gt.shape} != image {expected_hw}: {path}")
    return gt


def apply_c_group(model, rgb, prob, roi):
    # Exactly the earlier C-group: ROI + adaptive confidence, no other postprocess.
    result = model._apply_roi_and_edge(prob.copy(), roi[0], roi[1])
    thresholds = (model.conf_thresholds_dark if roi[2] < model.screen_brightness_split
                  else model.conf_thresholds_light)
    for class_id, threshold in thresholds.items():
        if threshold > 0:
            result[:, :, class_id] = np.where(result[:, :, class_id] < threshold,
                                              0.0, result[:, :, class_id])
    return result.argmax(axis=-1).astype(np.uint8)


def update_cm(cm, gt, pred):
    valid = (gt < 4) & (pred < 4)
    cm += np.bincount(gt[valid].astype(np.int64) * 4 + pred[valid].astype(np.int64),
                      minlength=16).reshape(4, 4)


def compute_metrics(cm):
    m = cm.astype(np.float64)
    tp = np.diag(m)
    d = m.sum(0) + m.sum(1) - tp
    iou = np.divide(tp, d, out=np.zeros(4, dtype=np.float64), where=d > 0) * 100
    return dict(mIoU_pct=round(float(iou.mean()), 3),
                defect_mIoU_pct=round(float(iou[1:].mean()), 3),
                background_IoU_pct=round(float(iou[0]), 3),
                oil_IoU_pct=round(float(iou[1]), 3),
                stain_IoU_pct=round(float(iou[2]), 3),
                scratch_IoU_pct=round(float(iou[3]), 3))


def save_summary(rows):
    output = OUTPUT / "summary.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "input_height", "input_width",
            "status", "n_images", "mIoU_pct", "defect_mIoU_pct", "background_IoU_pct",
            "oil_IoU_pct", "stain_IoU_pct", "scratch_IoU_pct", "mean_inference_ms",
            "mean_roi_conf_ms", "gpu_peak_allocated_MiB", "gpu_peak_reserved_MiB", "error"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    split = DATASET / "ImageSets" / "Segmentation" / "val.txt"
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS}")
    if not split.is_file():
        raise FileNotFoundError(f"Validation list not found: {split}")
    ids = [x.strip() for x in split.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    if not ids:
        raise RuntimeError("val.txt contains no image names")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    print(f"Image count={len(ids)}, CUDA={cuda}; output={OUTPUT}", flush=True)
    model = SegFormerInference(num_classes=4, variant="mit_b2", model_path=str(WEIGHTS),
                               input_shape=[512, 512], cuda=cuda, fast_mode=True,
                               use_roi_mask=True, use_postprocess=False)
    rows = []
    for name, height, width in VARIANTS:
        print(f"\n=== {name}: {height} x {width} ===", flush=True)
        model.input_shape = [height, width]
        cm = np.zeros((4, 4), dtype=np.int64)
        model_secs = 0.0
        filter_secs = 0.0
        status, error = "ok", ""
        finished = 0
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            for idx, image_id in enumerate(ids, 1):
                image_path = DATASET / "JPEGImages" / f"{image_id}.jpg"
                label_path = DATASET / "SegmentationClass" / f"{image_id}.png"
                if not image_path.is_file() or not label_path.is_file():
                    raise FileNotFoundError(f"Missing image or label for {image_id}")
                with Image.open(image_path) as im:
                    rgb = np.asarray(cvtColor(im), dtype=np.uint8)
                gt = read_gt(label_path, rgb.shape[:2])
                roi = model._detect_screen_roi(rgb)
                if cuda:
                    torch.cuda.synchronize()
                t = time.perf_counter()
                with torch.inference_mode():
                    prob = model._fast_predict(rgb)
                if cuda:
                    torch.cuda.synchronize()
                model_secs += time.perf_counter() - t
                t = time.perf_counter()
                pred = apply_c_group(model, rgb, prob, roi)
                filter_secs += time.perf_counter() - t
                update_cm(cm, gt, pred)
                finished += 1
                if idx == 1 or idx % 20 == 0 or idx == len(ids):
                    print(f"  {idx}/{len(ids)}", flush=True)
        except torch.cuda.OutOfMemoryError as ex:
            status = "CUDA_OOM"
            error = str(ex).splitlines()[0][:350]
            print("  CUDA OOM: skipping this resolution; keeping completed results.", flush=True)
        finally:
            # Report the peak including model weight allocations. This is not system-wide GPU usage.
            peak_alloc = round(torch.cuda.max_memory_allocated() / 1048576, 1) if cuda else ""
            peak_reserved = round(torch.cuda.max_memory_reserved() / 1048576, 1) if cuda else ""
        metrics = compute_metrics(cm) if status == "ok" and finished == len(ids) else {}
        if status == "ok" and finished != len(ids):
            status = "incomplete"
        row = dict(variant=name, input_height=height, input_width=width,
                   status=status, n_images=finished if status == "ok" else 0,
                   mIoU_pct=metrics.get("mIoU_pct", ""),
                   defect_mIoU_pct=metrics.get("defect_mIoU_pct", ""),
                   background_IoU_pct=metrics.get("background_IoU_pct", ""),
                   oil_IoU_pct=metrics.get("oil_IoU_pct", ""),
                   stain_IoU_pct=metrics.get("stain_IoU_pct", ""),
                   scratch_IoU_pct=metrics.get("scratch_IoU_pct", ""),
                   mean_inference_ms=round(model_secs / finished * 1000, 2) if finished else "",
                   mean_roi_conf_ms=round(filter_secs / finished * 1000, 2) if finished else "",
                   gpu_peak_allocated_MiB=peak_alloc, gpu_peak_reserved_MiB=peak_reserved,
                   error=error)
        rows.append(row)
        save_summary(rows)
        if status == "ok":
            np.savetxt(OUTPUT / f"{name}_confusion.csv", cm, delimiter=",", fmt="%d")
            print(f"  mIoU={row['mIoU_pct']}%, oil={row['oil_IoU_pct']}%, "
                  f"stain={row['stain_IoU_pct']}%, scratch={row['scratch_IoU_pct']}%, "
                  f"peak VRAM allocated={peak_alloc} MiB", flush=True)
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
    (OUTPUT / "run_info.json").write_text(json.dumps(dict(
        checkpoint=str(WEIGHTS), validation_split=str(split), images=len(ids),
        variants=VARIANTS, pipeline="fast_mode + ROI + adaptive confidence", 
        caveat="Selected on validation; confirm final model on independent held-out test set"),
        ensure_ascii=False, indent=2), encoding="utf-8")
    successful = [r for r in rows if r["status"] == "ok"]
    if successful:
        best = max(successful, key=lambda x: x["mIoU_pct"])
        print(f"\nBest of NEW sizes: {best['variant']} = {best['mIoU_pct']}%", flush=True)
    print(f"Done. Results: {OUTPUT / 'summary.csv'}", flush=True)

if __name__ == "__main__":
    main()
