# -*- coding: utf-8 -*-
"""SegFormer inference ablation: 5 cumulative + 4 leave-one-out configurations.

This file does not alter the original project, network weights, or training setup.
Uses exact processing order of segformer_infer.py: model -> ROI -> confidence ->
industrial postprocess -> majority-by-connected-components vote.

Run from D:\\AI\\projects\\DeepLabV3 with the current base Python interpreter.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from segformer_infer import SegFormerInference
from utils.utils import cvtColor

ROOT = Path(__file__).resolve().parent
DATASET = Path(r"D:\AI\dataset\MobilePhoneScreenSurfaceDefectSegmentation\MSD")
WEIGHTS = ROOT / "logs_segformer_mit_b2" / "best_epoch_weights.pth"
RESULTS = ROOT / "ablation_segformer_9groups"
NAMES = ("background", "oil", "stain", "scratch")

# ROI / confidence / morphology and area filtering / connected-component voting
CONFIGS = [
    ("A_raw",           False, False, False, False),
    ("B_plus_roi",      True,  False, False, False),
    ("C_plus_conf",     True,  True,  False, False),
    ("D_plus_post",     True,  True,  True,  False),
    ("E_full",          True,  True,  True,  True),
    ("F_without_roi",   False, True,  True,  True),
    ("G_without_conf",  True,  False, True,  True),
    ("H_without_post",  True,  True,  False, True),
    ("I_without_vote",  True,  True,  True,  False),
]


def read_gt(path: Path, expected_hw: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim != 2:
        raise ValueError(f"Label image is not single channel: {path} (shape={arr.shape})")
    vals = set(np.unique(arr).tolist())
    if vals.issubset({0, 1, 2, 3, 255}):
        mapped = arr.astype(np.uint8)
    elif vals.issubset({0, 38, 75, 113, 255}):
        mapped = np.zeros(arr.shape, dtype=np.uint8)
        for raw, cls in ((38, 1), (75, 2), (113, 3), (255, 255)):
            mapped[arr == raw] = cls
    else:
        raise ValueError(f"Unexpected GT labels {sorted(vals)} in {path}; inspect before evaluation")
    if mapped.shape != expected_hw:
        raise ValueError(f"Ground truth {mapped.shape} != prediction {expected_hw} for {path}")
    return mapped


def majority_vote(mask: np.ndarray) -> np.ndarray:
    """Identical voting logic to SegFormerInference.detect_image."""
    result = mask.copy()
    classes = [c for c in (1, 2, 3) if np.any(result == c)]
    if len(classes) <= 1:
        return result
    components = {}
    for cls in classes:
        n, _, _, _ = cv2.connectedComponentsWithStats(
            (result == cls).astype(np.uint8), connectivity=8)
        components[cls] = n - 1
    max_n = max(components.values())
    candidates = [c for c in classes if components[c] == max_n]
    dominant = candidates[0] if len(candidates) == 1 else max(
        candidates, key=lambda c: int((result == c).sum()))
    result[(result > 0) & (result != dominant)] = 0
    return result


def make_mask(model, rgb, original_prob, enabled, roi_data=None):
    use_roi, use_conf, use_post, use_vote = enabled
    prob = original_prob.copy()
    edge = None
    mean_gray = 128.0  # detect_image uses this default when ROI is disabled
    if use_roi:
        roi, edge, mean_gray = roi_data
        prob = model._apply_roi_and_edge(prob, roi, edge)
    if use_conf:
        thresh = model.conf_thresholds_dark if mean_gray < model.screen_brightness_split \
            else model.conf_thresholds_light
        for class_id, threshold in thresh.items():
            if threshold > 0:
                prob[:, :, class_id] = np.where(prob[:, :, class_id] < threshold, 0.0,
                                                prob[:, :, class_id])
    mask = prob.argmax(axis=-1).astype(np.uint8)
    if use_post:
        mask = model.industrial_postprocess(mask, edge)
    if use_vote:
        mask = majority_vote(mask)
    return mask


def cm_update(matrix, gt, pred):
    good = (gt < 4) & (pred < 4)
    matrix += np.bincount(
        gt[good].astype(np.int64) * 4 + pred[good].astype(np.int64),
        minlength=16).reshape(4, 4)


def scores(matrix):
    m = matrix.astype(np.float64)
    tp = np.diag(m)
    true_count = m.sum(axis=1)
    pred_count = m.sum(axis=0)
    den = true_count + pred_count - tp
    iou = np.divide(tp, den, out=np.zeros_like(tp), where=den > 0)
    recall = np.divide(tp, true_count, out=np.zeros_like(tp), where=true_count > 0)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    return iou, recall, precision


def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Missing trained weight file: {WEIGHTS}")
    split_file = DATASET / "ImageSets" / "Segmentation" / "val.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing validation split: {split_file}")
    ids = [s.strip() for s in split_file.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    if not ids:
        raise ValueError("val.txt is empty")
    RESULTS.mkdir(parents=True, exist_ok=True)

    model = SegFormerInference(num_classes=4, variant="mit_b2", model_path=str(WEIGHTS),
                               input_shape=[512, 512], cuda=torch.cuda.is_available(),
                               fast_mode=True, use_roi_mask=True, use_postprocess=True)
    matrices = {name: np.zeros((4, 4), dtype=np.int64) for name, *_ in CONFIGS}
    preprocess_time = 0.0
    run_time = {name: 0.0 for name, *_ in CONFIGS}
    for count, image_id in enumerate(ids, 1):
        image_path = DATASET / "JPEGImages" / (image_id + ".jpg")
        gt_path = DATASET / "SegmentationClass" / (image_id + ".png")
        if not image_path.exists() or not gt_path.exists():
            raise FileNotFoundError(f"Dataset missing: {image_path} OR {gt_path}")
        with Image.open(image_path) as img:
            rgb = np.asarray(cvtColor(img)).astype(np.uint8)
        gt = read_gt(gt_path, rgb.shape[:2])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        prob = model._fast_predict(rgb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        preprocess_time += time.perf_counter() - t0

        roi_data = model._detect_screen_roi(rgb)
        for name, roi, conf, post, vote in CONFIGS:
            start = time.perf_counter()
            pred = make_mask(model, rgb, prob, (roi, conf, post, vote), roi_data)
            run_time[name] += time.perf_counter() - start
            cm_update(matrices[name], gt, pred)
        if count == 1 or count % 20 == 0 or count == len(ids):
            print(f"Done {count}/{len(ids)}: {image_id}", flush=True)

    out_csv = RESULTS / "summary.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "ROI", "confidence_filter", "morphology_area", "class_vote",
                         "mIoU_pct", "defect_mIoU_pct", "macro_precision_pct", "macro_recall_pct",
                         "IoU_background_pct", "IoU_oil_pct", "IoU_stain_pct", "IoU_scratch_pct",
                         "model_inference_ms_per_image", "postprocess_ms_per_image", "n_images"])
        for name, roi, conf, post, vote in CONFIGS:
            matrix = matrices[name]
            iou, rec, prec = scores(matrix)
            writer.writerow([name, int(roi), int(conf), int(post), int(vote),
                             *[round(float(v), 3) for v in (
                                 iou.mean() * 100, iou[1:].mean() * 100,
                                 prec.mean() * 100, rec.mean() * 100)],
                             *[round(float(v * 100), 3) for v in iou],
                             round(preprocess_time / len(ids) * 1000, 2),
                             round(run_time[name] / len(ids) * 1000, 2), len(ids)])
            np.savetxt(RESULTS / (name + "_confusion.csv"), matrix, delimiter=",", fmt="%d")
            print(f"{name:19} mIoU {iou.mean()*100:.2f}%  "
                  f"defect mIoU {iou[1:].mean()*100:.2f}%  "
                  f"Oil/Stain/Scratch {[round(float(x*100),2) for x in iou[1:]]}")

    metadata = {
        "evaluation_split": str(split_file),
        "weight_file": str(WEIGHTS),
        "samples": len(ids),
        "fast_mode": True,
        "description": "Same SegFormer weights, frozen network. Cumulative + leave-one-out postprocessing.",
        "important_cautions": [
            "Numbers are validation metrics, not held-out test metrics.",
            "F_without_roi uses default light thresholds (screen_mean=128), matching detect_image logic.",
            "ROI detection time and CSV computation are not included in postprocess_ms_per_image.",
            "The 9 configurations measure inference-stage effects only, not training-stage ablation.",
            "D_plus_post and I_without_vote intentionally represent the same configuration.",
            "With no ROI, edge-position rejection has no edge zone; this is a joint change within ROI feature.",
        ],
    }
    (RESULTS / "run_info.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nFinished ->", out_csv)
    print("Note: D_plus_post == I_without_vote by definition (repeated for both layouts).")


if __name__ == "__main__":
    main()
