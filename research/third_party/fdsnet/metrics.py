# -*- coding: utf-8 -*-
"""Pixel-level confusion matrix and objective metrics; class indices are canonical."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np

CLASSES = ("background", "oil", "stain", "scratch")


def update(cm: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> None:
    if pred.shape != gt.shape:
        raise ValueError(f"标注与预测尺寸不一致: gt={gt.shape}; pred={pred.shape}")
    valid = (gt < 4) & (pred < 4)
    cm += np.bincount(gt[valid].astype(np.int64) * 4 + pred[valid].astype(np.int64), minlength=16).reshape(4, 4)


def scores(cm: np.ndarray) -> dict:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    actual, predicted = cm.sum(axis=1), cm.sum(axis=0)
    union = actual + predicted - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    prec = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    return {
        "mIoU": round(float(iou.mean()*100), 4),
        "defect_mIoU": round(float(iou[1:].mean()*100), 4),
        "IoU": {n: round(float(iou[i]*100), 4) for i, n in enumerate(CLASSES)},
        "Precision": {n: round(float(prec[i]*100), 4) for i, n in enumerate(CLASSES)},
        "Recall": {n: round(float(recall[i]*100), 4) for i, n in enumerate(CLASSES)},
        "support": {n: int(actual[i]) for i,n in enumerate(CLASSES)},
    }


def save_cm(path: Path, cm: np.ndarray) -> None:
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['GT\\Pred', *CLASSES])
        for name, row in zip(CLASSES, cm):
            w.writerow([name, *map(int, row)])


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def save_rows(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows and not fields:
        return
    fields = fields or list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
