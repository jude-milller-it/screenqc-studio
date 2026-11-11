# -*- coding: utf-8 -*-
r"""统一评测 DeepLabV3+ / SegFormer / FDSNet（MSD 数据集）。

绝不修改模型、权重或标签。全部使用同一个 val.txt，原图尺寸累计混淆矩阵。
运行：D:\AI\miniForge1\python.exe evaluate_three_models.py
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]  # DeepLabV3 root
FDSNET_ROOT = PROJECT_ROOT / "third_party" / "fdsnet"
LABELS = ("background", "oil", "stain", "scratch")


def read_label(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.ndim != 2:
        raise ValueError(f"标签必须单通道/P模式: {path}, shape={a.shape}")
    vals = set(np.unique(a).tolist())
    if vals.issubset({0, 1, 2, 3, 255}):
        gt = a.astype(np.uint8)
    elif vals.issubset({0, 38, 75, 113, 255}):
        gt = np.zeros(a.shape, dtype=np.uint8)
        for src, dst in ((38, 1), (75, 2), (113, 3), (255, 255)):
            gt[a == src] = dst
    else:
        raise ValueError(f"标签值未知 {vals}: {path}；停止评测，避免错分")
    if gt.shape != shape:
        raise ValueError(f"标签大小 {gt.shape} != 图片 {shape}: {path}")
    return gt


def score_cm(cm: np.ndarray) -> dict:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    fp = predicted - tp
    fn = support - tp
    tn = cm.sum() - tp - fp - fn
    union = support + predicted - tp

    def div(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b > 0)

    iou = div(tp, union)
    precision = div(tp, predicted)
    recall = div(tp, support)
    f1 = div(2 * tp, 2 * tp + fp + fn)
    specificity = div(tn, tn + fp)
    bg = float(support[0])
    return {
        'mIoU_pct': float(iou.mean() * 100),
        'defect_mIoU_pct': float(iou[1:].mean() * 100),
        'mPrecision_pct': float(precision.mean() * 100),
        'mRecall_pct': float(recall.mean() * 100),
        'mF1_pct': float(f1.mean() * 100),
        'pixel_accuracy_pct': float(tp.sum() / max(1, cm.sum()) * 100),
        'background_false_positives_per_Mpx': float(cm[0, 1:].sum() / bg * 1e6) if bg else 0.0,
        'per_class': {name: {
            'IoU_pct': float(iou[k] * 100), 'Precision_pct': float(precision[k] * 100),
            'Recall_pct': float(recall[k] * 100), 'F1_Dice_pct': float(f1[k] * 100),
            'Specificity_pct': float(specificity[k] * 100),
            'TP': int(tp[k]), 'FP': int(fp[k]), 'FN': int(fn[k]),
            'GT_pixels': int(support[k]), 'pred_pixels': int(predicted[k]),
        } for k, name in enumerate(LABELS)}
    }


def update(cm, gt, pred):
    if pred.shape != gt.shape:
        raise ValueError(f"预测尺寸错: {pred.shape} / {gt.shape}")
    if not np.isin(pred, [0, 1, 2, 3]).all():
        raise ValueError(f"预测类别超出0-3: {np.unique(pred)}")
    valid = (gt != 255) & (gt < 4)
    cm += np.bincount(gt[valid].astype(np.int64) * 4 + pred[valid].astype(np.int64),
                      minlength=16).reshape(4, 4)


def save_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def image_file(dataset: Path, name: str) -> Path:
    for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
        p = dataset / 'JPEGImages' / (name + ext)
        if p.exists():
            return p
    raise FileNotFoundError(f'找不到图片: {name}')


def load_deeplab(project: Path, scheme: int, cuda: bool):
    sys.path.insert(0, str(project))
    from deeplab import DeeplabV3
    folder = {0: 'logs_baseline', 1: 'logs_loss_opt', 2: 'logs_cbam', 3: 'logs_full_all'}[scheme]
    weights = project / folder / 'best_epoch_weights.pth'
    if not weights.is_file():
        raise FileNotFoundError(weights)
    model = DeeplabV3(num_classes=4, model_path=str(weights), use_cbam=(scheme >= 2),
                      use_postprocess=False, cuda=cuda)
    return model, weights, f'DeepLabV3_scheme{scheme}'


def load_segformer(project: Path, cuda: bool):
    sys.path.insert(0, str(project))
    from segformer_infer import SegFormerInference
    weights = project / 'logs_segformer_mit_b2' / 'best_epoch_weights.pth'
    if not weights.is_file():
        raise FileNotFoundError(weights)
    model = SegFormerInference(num_classes=4, variant='mit_b2', model_path=str(weights),
                               cuda=cuda, input_shape=[640, 1152], fast_mode=True,
                               use_roi_mask=True, use_postprocess=False)
    return model, weights, 'SegFormer_640x1152_C'


def load_fdsnet(cuda: bool):
    if not FDSNET_ROOT.is_dir():
        raise FileNotFoundError(f'FDSNet源码目录不存在：{FDSNET_ROOT}')
    sys.path.insert(0, str(FDSNET_ROOT))
    from fdsnet_core.fdsnet import FDSNet
    weights = FDSNET_ROOT / 'fdsnet__phone_voc_best_model.pth'
    if not weights.is_file():
        raise FileNotFoundError(weights)
    model = FDSNet(num_classes=4, aux=True)
    state = torch.load(str(weights), map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=True)
    model.to('cuda' if cuda else 'cpu').eval()
    return model, weights, 'FDSNet_original'


def predict_deeplab(model, rgb: Image.Image):
    return np.asarray(model.get_miou_png(rgb), dtype=np.uint8)


def predict_segformer(model, rgb: Image.Image):
    from utils.utils import cvtColor
    image = np.asarray(cvtColor(rgb), dtype=np.uint8)
    roi = model._detect_screen_roi(image)
    with torch.inference_mode():
        prob = model._fast_predict(image)
    prob = model._apply_roi_and_edge(prob.copy(), roi[0], roi[1])
    thresholds = (model.conf_thresholds_dark if roi[2] < model.screen_brightness_split
                  else model.conf_thresholds_light)
    for cls, threshold in thresholds.items():
        if threshold > 0:
            prob[..., cls] = np.where(prob[..., cls] < threshold, 0.0, prob[..., cls])
    return prob.argmax(-1).astype(np.uint8)


def predict_fdsnet(model, rgb: Image.Image, cuda: bool):
    # 与已经复现81.4263%的修正版一致：model 原始类别顺序与GT相同，不能交换2、3。
    w = max(32, int(rgb.width * 0.75))
    h = max(32, int(rgb.height * 0.75))
    w = (w + 31) // 32 * 32
    h = (h + 31) // 32 * 32
    arr = np.asarray(rgb.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.
    arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    ten = torch.from_numpy(arr.transpose(2, 0, 1).copy()).float().unsqueeze(0)
    if cuda:
        ten = ten.cuda()
    with torch.inference_mode():
        raw = model(ten)[0]
        raw = F.interpolate(raw, size=(rgb.height, rgb.width), mode='bilinear', align_corners=True)
        pred = raw.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


def run_one(which, project, dataset, names, output, cuda, scheme):
    print(f'\n======== 评测 {which} ========', flush=True)
    if which == 'deeplab':
        model, weights, label = load_deeplab(project, scheme, cuda)
        predictor = lambda im: predict_deeplab(model, im)
        settings = {'scheme': scheme, 'use_cbam': scheme >= 2, 'use_postprocess': False}
    elif which == 'segformer':
        model, weights, label = load_segformer(project, cuda)
        predictor = lambda im: predict_segformer(model, im)
        settings = {'input_hw': [640, 1152], 'fast_mode': True, 'ROI': True,
                    'confidence': 'adaptive', 'morphology': False, 'vote': False}
    else:
        model, weights, label = load_fdsnet(cuda)
        predictor = lambda im: predict_fdsnet(model, im, cuda)
        settings = {'scale': 0.75, 'alignment': 32, 'class_map': 'identity', 'postprocess': False}
    cm = np.zeros((4, 4), dtype=np.int64)
    elapsed = []
    per_image = []
    for idx, stem in enumerate(names, 1):
        with Image.open(image_file(dataset, stem)) as im:
            rgb = im.convert('RGB')
        gt = read_label(dataset / 'SegmentationClass' / (stem + '.png'), (rgb.height, rgb.width))
        if cuda: torch.cuda.synchronize()
        start = time.perf_counter()
        pred = predictor(rgb)
        if cuda: torch.cuda.synchronize()
        sec = time.perf_counter() - start
        elapsed.append(sec * 1000)
        update(cm, gt, pred)
        local_cm = np.zeros((4, 4), dtype=np.int64)
        update(local_cm, gt, pred)
        local = score_cm(local_cm)
        per_image.append({'model': label, 'image': stem, 'time_ms': round(sec*1000, 2),
                          'oil_IoU_pct': round(local['per_class']['oil']['IoU_pct'],4),
                          'stain_IoU_pct': round(local['per_class']['stain']['IoU_pct'],4),
                          'scratch_IoU_pct': round(local['per_class']['scratch']['IoU_pct'],4),
                          'GT_scratch_pixels': local['per_class']['scratch']['GT_pixels']})
        if idx == 1 or idx % 20 == 0 or idx == len(names):
            print(f'  {idx}/{len(names)}', flush=True)
    result = score_cm(cm)
    result.update({'model': label, 'weights': str(weights), 'count': len(names),
                   'mean_pipeline_ms': float(np.mean(elapsed)),
                   'p95_pipeline_ms': float(np.percentile(elapsed, 95)),
                   'fps_sequential': float(1000 / np.mean(elapsed)),
                   'settings': settings})
    out = output / label
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'confusion_matrix.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['GT\\Pred', *LABELS])
        for name, line in zip(LABELS, cm): w.writerow([name, *line.tolist()])
    (out / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    save_csv(out / 'per_image.csv', per_image, ['model', 'image', 'time_ms', 'oil_IoU_pct',
                                               'stain_IoU_pct', 'scratch_IoU_pct', 'GT_scratch_pixels'])
    del model
    gc.collect()
    if cuda: torch.cuda.empty_cache()
    print(f"{label}: mIoU={result['mIoU_pct']:.3f}%，Oil={result['per_class']['oil']['IoU_pct']:.2f}%，"
          f"Stain={result['per_class']['stain']['IoU_pct']:.2f}%，Scratch={result['per_class']['scratch']['IoU_pct']:.2f}%", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description='MSD 三模型统一评测')
    parser.add_argument('--project', default=str(PROJECT_ROOT))
    parser.add_argument('--dataset', default=r'D:\AI\dataset\MobilePhoneScreenSurfaceDefectSegmentation\MSD')
    parser.add_argument('--split', default='val')
    parser.add_argument('--models', nargs='+', choices=['deeplab','segformer','fdsnet'],
                        default=['deeplab','segformer','fdsnet'])
    parser.add_argument('--deeplab-scheme', type=int, default=3, choices=[0,1,2,3])
    args = parser.parse_args()
    project, dataset = Path(args.project), Path(args.dataset)
    if any(x in args.models for x in ('deeplab','segformer')) and not project.is_dir():
        raise FileNotFoundError(project)
    split = dataset / 'ImageSets' / 'Segmentation' / (args.split + '.txt')
    if not split.is_file():
        raise FileNotFoundError(split)
    names = [line.strip() for line in split.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError(f'{split} 为空或存在重复图片ID')
    output = HERE / ('unified_results_' + args.split)
    output.mkdir(exist_ok=True)
    cuda = torch.cuda.is_available()
    print(f'统一测试：{len(names)}张，{split}，GPU={cuda}；类别顺序={LABELS}', flush=True)
    print('注意：FDSNet公开权重可能与当前验证图片存在训练集重叠，结果仅为探索性对比。', flush=True)
    records = []
    for name in args.models:
        try:
            result = run_one(name, project, dataset, names, output, cuda, args.deeplab_scheme)
            records.append(result)
            write_outputs(output, records, split)
        except Exception:
            write_outputs(output, records, split)
            raise
    print('\n完成：', output / 'model_comparison.csv', flush=True)


def write_outputs(output, results, split):
    summaries = []
    classes = []
    for r in results:
        row = {'model': r['model'], 'images': r['count'], 'mIoU_pct': r['mIoU_pct'],
               'defect_mIoU_pct': r['defect_mIoU_pct'], 'mPrecision_pct': r['mPrecision_pct'],
               'mRecall_pct': r['mRecall_pct'], 'mF1_pct': r['mF1_pct'],
               'pixel_accuracy_pct': r['pixel_accuracy_pct'],
               'background_FP_per_Mpx': r['background_false_positives_per_Mpx'],
               'mean_pipeline_ms': r['mean_pipeline_ms'], 'p95_pipeline_ms': r['p95_pipeline_ms'],
               'fps': r['fps_sequential']}
        for cls in LABELS:
            for metric, key in (('IoU','IoU_pct'),('P','Precision_pct'),('R','Recall_pct'),('F1','F1_Dice_pct')):
                row[f'{cls}_{metric}'] = r['per_class'][cls][key]
            classes.append({'model': r['model'], 'class': cls, **r['per_class'][cls]})
        summaries.append(row)
    basic = ['model','images','mIoU_pct','defect_mIoU_pct','mPrecision_pct','mRecall_pct','mF1_pct',
             'pixel_accuracy_pct','background_FP_per_Mpx','mean_pipeline_ms','p95_pipeline_ms','fps']
    wide = basic + [f'{cls}_{k}' for cls in LABELS for k in ('IoU','P','R','F1')]
    save_csv(output / 'model_comparison.csv', summaries, wide)
    save_csv(output / 'per_class_metrics.csv', classes,
             ['model','class','IoU_pct','Precision_pct','Recall_pct','F1_Dice_pct',
              'Specificity_pct','TP','FP','FN','GT_pixels','pred_pixels'])
    (output / 'experiment_info.json').write_text(json.dumps({
        'split_path': str(split), 'n_images': results[0]['count'] if results else 0,
        'class_order': LABELS, 'models': [{'name':r['model'],'weights':r['weights'],'settings':r['settings']} for r in results],
        'notes': ['背景类参与mIoU、mPrecision、mRecall、mF1的四类宏平均',
                  'Dice与F1在此像素级任务数值相同',
                  '单张无该类时IoU显示0，不可把单张指标直接平均当作整体IoU',
                  'FDSNet可能存在原作者训练集与当前val重叠；对比结果不能声称独立泛化',
                  '各模型推理输入尺寸不同，符合各自已测最优配置；耗时是单图预测完整流程，不含读图及读标签'],
    }, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
