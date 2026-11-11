"""Consistent per-pixel segmentation metrics, ignoring GT label 255."""
from __future__ import annotations
import numpy as np

CLASSES = ('background', 'oil', 'stain', 'scratch')
CHINESE = ('背景', '油污', '污渍', '划痕')


def confusion_update(cm: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> None:
    if gt.shape != pred.shape:
        raise ValueError(f'预测与标签尺寸不一致: {pred.shape} vs {gt.shape}')
    if np.any(pred > 3):
        raise ValueError('预测标签只能为0-3')
    valid = gt != 255
    if np.any(gt[valid] > 3):
        raise ValueError('真实标签除了255忽略值外只能为0-3')
    cm += np.bincount(gt[valid].astype(np.int64)*4 + pred[valid].astype(np.int64),
                      minlength=16).reshape(4, 4)


def score(cm: np.ndarray) -> dict:
    a = cm.astype(np.float64)
    tp = a.diagonal()
    supports, preds = a.sum(1), a.sum(0)
    fp, fn = preds - tp, supports - tp
    tn = a.sum() - tp - fp - fn

    def safe(n, d):
        return np.divide(n, d, out=np.zeros_like(n, dtype=np.float64), where=d > 0)

    iou = safe(tp, supports + preds - tp)
    pr = safe(tp, preds)
    re = safe(tp, supports)
    f1 = safe(2*tp, 2*tp + fp + fn)
    spec = safe(tn, tn+fp)
    return {
        'mIoU_pct': float(iou.mean()*100),
        'defect_mIoU_pct': float(iou[1:].mean()*100),
        'mPrecision_pct': float(pr.mean()*100),
        'mRecall_pct': float(re.mean()*100),
        'mF1_pct': float(f1.mean()*100),
        'pixel_accuracy_pct': float(tp.sum()/max(1, a.sum())*100),
        'per_class': {name: {
            'IoU_pct': float(iou[i]*100),
            'Precision_pct': float(pr[i]*100),
            'Recall_pct': float(re[i]*100),
            'F1_Dice_pct': float(f1[i]*100),
            'Specificity_pct': float(spec[i]*100),
            'TP': int(tp[i]), 'FP': int(fp[i]), 'FN': int(fn[i]),
            'GT_pixels': int(supports[i]), 'pred_pixels': int(preds[i])
        } for i, name in enumerate(CLASSES)}
    }
