# -*- coding: utf-8 -*-
"""Auditable, deterministic evidence summaries used by *both* report variants.
No AI in measurements, decisions or ranking.
"""
from __future__ import annotations
from collections import Counter
from statistics import mean,median
from pathlib import Path
from datetime import datetime
import math

def safe_float(value,default=0.0):
    try:
        v=float(value)
        return v if math.isfinite(v) else default
    except (ValueError,TypeError):return default

def inspection_evidence(images,regions):
    models=sorted({str(x.get('model','')).strip() for x in images if x.get('model')})
    by_image=Counter(str(r.get('image','')) for r in regions)
    found=sum(bool(safe_float(r.get('regions',by_image.get(str(r.get('source','')),0)))) for r in images)
    times=[safe_float(x.get('time_ms') or x.get('inference_ms'),None) for x in images]
    times=sorted(x for x in times if x is not None)
    p95=times[max(0,math.ceil(.95*len(times))-1)] if times else None
    counts=Counter(str(r.get('class_name') or r.get('class') or '').lower() for r in regions)
    pixels={c:sum(int(safe_float(x.get(c+'_pixels'))) for x in images) for c in ('oil','stain','scratch')}
    img_rank=sorted(images,key=lambda r:(-safe_float(r.get('regions',0)),-sum(safe_float(r.get(c+'_pixels')) for c in pixels),str(r.get('source',''))))
    review=[{'image':Path(x.get('source') or x.get('image') or '?').name,
             'regions':int(safe_float(x.get('regions'))),'predicted_pixels':sum(int(safe_float(x.get(c+'_pixels'))) for c in pixels),
             'oil_pixels':int(safe_float(x.get('oil_pixels'))),'stain_pixels':int(safe_float(x.get('stain_pixels'))),
             'scratch_pixels':int(safe_float(x.get('scratch_pixels'))),
             'reason':'候选区域较多（人工复核排序，不代表真实严重程度）'} for x in img_rank[:8]]
    all_px=sum(pixels.values())
    return {
        'schema':'screenqc.evidence.inspection.v1','images':len(images),'images_with_candidates':found,
        'images_without_candidates':len(images)-found,
        'candidate_image_rate_pct':round(100*found/len(images),2) if images else None,
        'regions':len(regions),'region_counts':{c:counts[c] for c in pixels},
        'predicted_pixels':pixels,
        'predicted_pixel_share_pct':{c:round(100*pixels[c]/all_px,2) if all_px else 0 for c in pixels},
        'regions_per_image_median':median([safe_float(x.get('regions')) for x in images]) if images else 0,
        'mean_processing_ms':round(mean(times),2) if times else None,
        'p95_processing_ms':round(p95,2) if p95 is not None else None,
        'total_recorded_processing_s':round(sum(times)/1000,2),
        'model_ids':models,'review_priority':review,
        'review_method':'按候选区域数量与预测像素数排序，不使用尚不存在的缺陷严重度标签',
        'limitations':['模型候选区域不等于经人工确认的缺陷实例',
                       '未提供人工复核及产品判定，不计算良率、误放行率、真实缺陷率',
                       'MSD图片批次不代表生产线真实产品批次',
                       '像素量反映预测面积，不反映真实物理面积或缺陷严重程度']}

def research_evidence(data):
    models={m['id']:m for m in data.get('models',[])}
    ordered=['deeplab_base','deeplab','segformer_raw','segformer_scale','segformer_roi','segformer_conf','segformer','segformer_oil','segformer_post_bad','fdsnet_tuned']
    stages=[]
    baseline=models.get('segformer_raw')
    for mid in ordered:
        m=models.get(mid)
        if not m:continue
        current=m['summary'].get('mIoU_pct')
        stages.append({'id':mid,'mIoU_pct':current,
          'delta_vs_raw_pp':round(current-baseline['summary']['mIoU_pct'],3) if baseline and current is not None and baseline['summary'].get('mIoU_pct') is not None and mid.startswith('segformer') else None})
    pairs=[]
    for a,b,label in [('deeplab_base','deeplab','DeepLab独立训练权重对照'),
                      ('segformer_raw','segformer_scale','同权重输入尺度'),
                      ('segformer','segformer_oil','同权重油污阈值'),
                      ('segformer_raw','segformer_oil','同权重完整推理历程')]:
        if a not in models or b not in models:continue
        old,new=models[a],models[b]
        def d(x,y):return round(x-y,3) if x is not None and y is not None else None
        pairs.append({'id':a+'__'+b,'label':label,'baseline':a,'candidate':b,
                      'mIoU_delta_pp':d(new['summary'].get('mIoU_pct'),old['summary'].get('mIoU_pct')),
                      'classes':{c:{k:d(new['classes'][c].get(k),old['classes'][c].get(k)) for k in ('IoU_pct','Precision_pct','Recall_pct','F1_Dice_pct')} for c in ('oil','stain','scratch')}})
    main=[models[k] for k in ('deeplab_base','deeplab','segformer_raw','segformer_oil','fdsnet_tuned') if k in models]
    rankings={}
    if main:
        for cls in ('oil','stain','scratch'):
            rankings[cls]={}
            for metric in ('IoU_pct','Precision_pct','Recall_pct','F1_Dice_pct'):
                valid=[m for m in main if m['classes'][cls].get(metric) is not None]
                if valid:
                    best=max(valid,key=lambda m:m['classes'][cls][metric]);rankings[cls][metric]={'model':best['id'],'value':best['classes'][cls][metric]}
    return {'schema':'screenqc.evidence.research.v1',
           'experiment':data.get('experiment',{}),'stage_metrics':stages,'controlled_pairs':pairs,
           'primary_rankings':rankings,
           'selected':{'five_main':[m['id'] for m in main],'ablation':[x['id'] for x in stages if x['id'].startswith('segformer')]},
           'limits':data.get('caveats',[])}
