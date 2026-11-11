# -*- coding: utf-8 -*-
"""SegFormer oil-precision constrained tuning, without changing training weights.

Usage from existing DeepLabV3 root:
    python experiments/segformer_oil_optimization.py

Relies on the SAME model loader and label evaluator used in
experiments/three_models/evaluate_three_models.py, so the baseline should
reproduce the latest unified comparison. No new environment required.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SHARED = HERE / 'three_models'
if not (SHARED / 'evaluate_three_models.py').is_file():
    raise SystemExit('找不到 experiments/three_models/evaluate_three_models.py。请先合并之前的三模型整合包。')
sys.path.insert(0, str(SHARED))
from evaluate_three_models import (  # noqa: E402
    image_file, load_segformer, read_label, score_cm, update,
)

LABELS = ['background', 'oil', 'stain', 'scratch']
# 仅调整油污规则；其它类别继续使用原模型的明暗自适应阈值。
DELTAS = [-0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06,
          0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30]
MARGINS = [0.01, 0.03, 0.05, 0.08]
REFERENCE = {
    'mIoU_pct': 76.72733429562977,
    'oil_IoU_pct': 71.42181624178365,
    'oil_Precision_pct': 74.66267873184609,
    'oil_Recall_pct': 94.27068364316276,
    'stain_IoU_pct': 72.21593429283737,
    'scratch_IoU_pct': 63.59833229170482,
}


def candidates():
    result = [{'name': 'baseline', 'delta': 0.0, 'margin': 0.0}]
    for delta in DELTAS:
        if delta != 0:
            result.append({'name': 'threshold_%+0.2f' % delta,
                           'delta': delta, 'margin': 0.0})
    for margin in MARGINS:
        result.append({'name': 'margin_%0.2f' % margin,
                       'delta': 0.0, 'margin': margin})
    # Small, predeclared combinations; do NOT search arbitrary combinations until overfit.
    for delta in (0.04, 0.08, 0.12):
        for margin in (0.03, 0.06):
            result.append({'name': 'delta_%0.2f_margin_%0.2f' % (delta, margin),
                           'delta': delta, 'margin': margin})
    return result


def save_csv(path: Path, rows: list[dict], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def prepare_prob(prob: np.ndarray, model, brightness: float):
    """Apply unchanged non-oil thresholds once. Oil is tuned separately.

    Important: np.argmax chooses the lowest class ID on a tie. Class 1 wins
    against stain/scratch ties but loses against background ties.
    """
    base_thresholds = (model.conf_thresholds_dark if brightness < model.screen_brightness_split
                       else model.conf_thresholds_light)
    for cls in (2, 3):
        t = float(base_thresholds.get(cls, 0))
        if t > 0:
            prob[..., cls] = np.where(prob[..., cls] < t, 0.0, prob[..., cls])

    bg = prob[..., 0]
    oil = prob[..., 1]
    stain = prob[..., 2]
    scratch = prob[..., 3]
    other_choice = np.argmax(prob[..., [0, 2, 3]], axis=-1).astype(np.uint8)
    other = np.array([0, 2, 3], dtype=np.uint8)[other_choice]
    # Tie must be handled exactly like np.argmax on original class order.
    oil_possible = (oil > bg) & (oil >= stain) & (oil >= scratch)
    competitors = np.maximum(np.maximum(bg, stain), scratch)
    oil_base_t = float(base_thresholds.get(1, 0))
    return oil, oil_possible, competitors, other, oil_base_t


def make_prediction(oil, oil_possible, competitors, other, base_t,
                    delta: float, margin: float):
    t = float(np.clip(base_t + delta, 0.0, 1.0))
    allowed = oil_possible & (oil >= t)
    if margin > 0:
        allowed &= (oil - competitors >= margin)
    return np.where(allowed, np.uint8(1), other)


def decode_metrics(cm: np.ndarray, cfg: dict, split: str) -> dict:
    r = score_cm(cm)
    out = {
        'variant': cfg['name'], 'subset': split,
        'oil_delta': cfg['delta'], 'oil_margin': cfg['margin'],
        'mIoU_pct': r['mIoU_pct'],
        'oil_Precision_pct': r['per_class']['oil']['Precision_pct'],
        'oil_Recall_pct': r['per_class']['oil']['Recall_pct'],
        'oil_IoU_pct': r['per_class']['oil']['IoU_pct'],
        'stain_IoU_pct': r['per_class']['stain']['IoU_pct'],
        'scratch_IoU_pct': r['per_class']['scratch']['IoU_pct'],
        'oil_FP_pixels': r['per_class']['oil']['FP'],
        'oil_FN_pixels': r['per_class']['oil']['FN'],
        'oil_TP_pixels': r['per_class']['oil']['TP'],
        'scratch_Recall_pct': r['per_class']['scratch']['Recall_pct'],
        'stain_Recall_pct': r['per_class']['stain']['Recall_pct'],
    }
    return out


def checks(row, base, min_recall):
    # Direct user requirement: NO IoU decrease for oil / stain / scratch / overall,
    # and preserve at least 92% oil recall. Tiny epsilon for float arithmetic only.
    eps = 1e-9
    keys = ['mIoU_pct', 'oil_IoU_pct', 'stain_IoU_pct', 'scratch_IoU_pct']
    return (all(row[k] + eps >= base[k] for k in keys)
            and row['oil_Recall_pct'] + eps >= min_recall)


def main():
    ap = argparse.ArgumentParser(description='SegFormer 油污误检抑制：零IoU退步约束')
    ap.add_argument('--dataset', default=r'D:\AI\dataset\MobilePhoneScreenSurfaceDefectSegmentation\MSD')
    ap.add_argument('--split', default='val')
    ap.add_argument('--min-oil-recall', type=float, default=92.0)
    ap.add_argument('--output', default=str(HERE / 'oil_optimization_results'))
    ap.add_argument('--baseline-tolerance', type=float, default=0.05,
                    help='Baseline must match previously unified evaluation within this many percentage points.')
    args = ap.parse_args()
    dataset = Path(args.dataset)
    split_file = dataset / 'ImageSets' / 'Segmentation' / (args.split + '.txt')
    if not split_file.exists():
        raise FileNotFoundError(f'找不到验证集文件: {split_file}')
    names = [x.strip() for x in split_file.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError(f'图片列表为空或有重复ID：{split_file}')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    print(f'使用 {len(names)} 张图片；CUDA={cuda}；权重保持不变；输入640x1152', flush=True)
    model, weights, _ = load_segformer(PROJECT, cuda)
    cfgs = candidates()
    # Deterministic partition based on val.txt's order; not a truly independent test,
    # because original training/split may have seen these images.
    cm_dev = np.zeros((len(cfgs), 4, 4), dtype=np.int64)
    cm_hold = np.zeros_like(cm_dev)
    timings = []
    from utils.utils import cvtColor
    for idx, name in enumerate(names):
        with Image.open(image_file(dataset, name)) as im:
            img = np.asarray(cvtColor(im.convert('RGB')), dtype=np.uint8)
        gt = read_label(dataset / 'SegmentationClass' / (name + '.png'), img.shape[:2])
        roi = model._detect_screen_roi(img)
        if cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            prob = model._fast_predict(img)
        if cuda: torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000)
        prob = model._apply_roi_and_edge(prob, roi[0], roi[1])
        oil, oil_possible, competitors, other, base_t = prepare_prob(prob, model, roi[2])
        valid = (gt < 4) & (gt != 255)
        # Fixed split for exploratory robustness: alternate image IDs.
        target = cm_dev if idx % 2 == 0 else cm_hold
        gt_flat = gt[valid].astype(np.int64)
        for j, cfg in enumerate(cfgs):
            pred = make_prediction(oil, oil_possible, competitors, other, base_t,
                                   cfg['delta'], cfg['margin'])
            encoded = gt_flat * 4 + pred[valid].astype(np.int64)
            target[j] += np.bincount(encoded, minlength=16).reshape(4, 4)
        if (idx+1) % 20 == 0 or idx+1 == len(names):
            print(f'  {idx+1}/{len(names)}', flush=True)

    cm_all = cm_dev + cm_hold
    rows = []
    by_subset = {}
    for group, cm_array in (('dev', cm_dev), ('holdout', cm_hold), ('all', cm_all)):
        rr = [decode_metrics(cm_array[i], cfg, group) for i, cfg in enumerate(cfgs)]
        rows += rr
        by_subset[group] = rr
    for group in ('dev', 'holdout', 'all'):
        base = by_subset[group][0]
        # A 120-image subset can have lower recall than the full 240-image set.
        # Never make the unchanged baseline ineligible merely because the
        # subset's original recall was below the user-specified FULL-set floor.
        # On subsets require no recall regression from their own baseline;
        # on the full set enforce the user's >=92% absolute requirement.
        recall_floor = (args.min_oil_recall if group == 'all'
                        else min(args.min_oil_recall, base['oil_Recall_pct']))
        for r in by_subset[group]:
            r['eligible_no_iou_drop'] = checks(r, base, recall_floor)
            r['oil_precision_gain_pp'] = r['oil_Precision_pct'] - base['oil_Precision_pct']
            r['oil_iou_gain_pp'] = r['oil_IoU_pct'] - base['oil_IoU_pct']
            r['miou_gain_pp'] = r['mIoU_pct'] - base['mIoU_pct']

    # Selection ONLY on dev. Holdout is reported but NOT used to pick/adjust parameters.
    dev_eligible = [r for r in by_subset['dev'] if r['eligible_no_iou_drop']]
    # Safety fallback for empty feasible sets: always produce reports.
    # The baseline is considered the safe fallback, not a successful improvement.
    chosen_dev = (max(dev_eligible,
                      key=lambda r: (r['oil_Precision_pct'], r['mIoU_pct']))
                  if dev_eligible else by_subset['dev'][0])
    chosen_name = chosen_dev['variant']
    picked = {group: next(r for r in by_subset[group] if r['variant'] == chosen_name)
              for group in ('dev', 'holdout', 'all')}
    baseline = by_subset['all'][0]
    # Strict no-loss release criterion is checked on both dev & holdout; if the selected
    # dev candidate does not satisfy holdout/full-set, production recommendation = baseline.
    is_improvement = (chosen_name != 'baseline' and
                      picked['holdout']['eligible_no_iou_drop'] and
                      picked['all']['eligible_no_iou_drop'] and
                      picked['all']['oil_Precision_pct'] > baseline['oil_Precision_pct'] + 1e-9)
    recommended = chosen_name if is_improvement else 'baseline'

    cols = ['subset', 'variant', 'oil_delta', 'oil_margin', 'mIoU_pct', 'oil_IoU_pct',
            'oil_Precision_pct', 'oil_Recall_pct', 'stain_IoU_pct', 'scratch_IoU_pct',
            'scratch_Recall_pct', 'stain_Recall_pct', 'oil_TP_pixels', 'oil_FP_pixels',
            'oil_FN_pixels', 'eligible_no_iou_drop', 'oil_precision_gain_pp',
            'oil_iou_gain_pp', 'miou_gain_pp']
    save_csv(out / 'all_variants.csv', rows, cols)
    save_csv(out / 'precision_recall_curve.csv', [r for r in rows if r['subset'] == 'all'],
             ['variant', 'oil_delta', 'oil_margin', 'oil_Precision_pct', 'oil_Recall_pct',
              'oil_IoU_pct', 'stain_IoU_pct', 'scratch_IoU_pct', 'mIoU_pct',
              'eligible_no_iou_drop'])
    # Compare with actual prior unified result: do not silently accept a different baseline.
    diff = {key: baseline[key] - expected for key, expected in REFERENCE.items()}
    baseline_matches = all(abs(v) <= args.baseline_tolerance for v in diff.values())
    report = {
        'images': len(names), 'split': str(split_file), 'checkpoint': str(weights),
        'method': 'oil-specific threshold increments and probability-margin gating',
        'baseline_match_previous': baseline_matches,
        'baseline_deviation_pp': diff,
        'reference_unified_metrics': REFERENCE,
        'baseline_current': baseline,
        'dev_eligible_count': len(dev_eligible),
        'dev_baseline_oil_recall_pct': by_subset['dev'][0]['oil_Recall_pct'],
        'holdout_baseline_oil_recall_pct': by_subset['holdout'][0]['oil_Recall_pct'],
        'full_min_oil_recall_pct': args.min_oil_recall,
        'selected_on_dev_only': chosen_name,
        'selected_dev': picked['dev'], 'selected_holdout': picked['holdout'],
        'selected_all': picked['all'],
        'recommended': recommended if baseline_matches else 'baseline',
        'recommendation_verified': bool(baseline_matches and (is_improvement or recommended == 'baseline')),
        'average_model_inference_ms': round(float(np.mean(timings)), 2),
        'warnings': [
            '若baseline与旧统一评测不一致，禁止采信自动推荐；检查权重、图片列表和评测代码。',
            '油污阈值及概率差距筛选仅改变预测后的判定，不修改模型参数。',
            '该验证集已经用于多轮实验；dev/holdout只是内部交叉检查，不能宣称独立测试。',
            '仅根据dev选参数；holdout和全量结果用于保守验收，不根据它们继续调参。',
            '开发/保留子集若原版召回率低于92%，采用该子集原版召回率作为无退步底线；全量仍必须满足92%。',
            '若没有满足条件的改善方案，保持原始配置，不代表程序出错。'
        ]
    }
    (out / 'recommended.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n=========== 结果 ===========')
    for name in ('mIoU_pct','oil_IoU_pct','oil_Precision_pct','oil_Recall_pct',
                 'stain_IoU_pct','scratch_IoU_pct'):
        print(f'{name:>23}: {baseline[name]:.4f}% -> {picked["all"][name]:.4f}%')
    print('旧统一评测基准复现:', '成功' if baseline_matches else '失败（结果不可信）')
    print('开发集符合条件的配置数:', len(dev_eligible), '/', len(cfgs))
    if not dev_eligible:
        print('提示：开发集无合格候选，回退到baseline，结果仍完整保存。')
    print('开发集最佳:', chosen_name)
    print('保留集与全量均满足零退步:', is_improvement)
    print('最终推荐:', report['recommended'])
    print('完整结果:', out / 'all_variants.csv')
    print('结论文件:', out / 'recommended.json')
    if not baseline_matches:
        raise SystemExit('ERROR: 基准未复现，停止采用优化方案；请提供recommended.json与输出报错。')

if __name__ == '__main__':
    main()
