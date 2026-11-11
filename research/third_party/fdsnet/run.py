# -*- coding: utf-8 -*-
"""FDSNet 高精度检测与后处理消融项目：Windows PyCharm/PowerShell 直接运行。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import time
from pathlib import Path
from dataclasses import fields

import numpy as np
import torch
from PIL import Image

from engine import (PROJECT, DEFAULT_DATASET, WEIGHTS, dataset_root, list_split,
                    image_for, read_label, load_model, predict_probs, save_artifacts)
from metrics import CLASSES, update, scores, save_cm, save_json, save_rows
from postprocess import Config, variants, apply, detect_safe_roi

RESULT_ROOT = PROJECT / 'results'
SELECTED = PROJECT / 'configs' / 'selected.json'


def opts():
    p = argparse.ArgumentParser(description='FDSNet 适配/推理后处理/消融评测（无需重新训练）')
    p.add_argument('command', nargs='?', default='tune', choices=['tune','evaluate','predict','check'],
                   help='tune=自动消融; evaluate=评测指定方案; predict=图片预测; check=检查权重')
    p.add_argument('--dataset', default=str(DEFAULT_DATASET))
    p.add_argument('--weights', default=str(WEIGHTS))
    p.add_argument('--split', default='val', choices=['val','test','train','trainval'])
    p.add_argument('--scale', type=float, default=.75)
    p.add_argument('--image', default='', help='predict: 图片路径或图片文件夹')
    p.add_argument('--variant', default='auto', help='baseline | auto | 预置方案名称')
    p.add_argument('--limit', type=int, default=0, help='仅调试前N张，0代表全部')
    p.add_argument('--extended', action='store_true', help='tune: 额外4个探索方案，更慢')
    p.add_argument('--flip-tta', action='store_true', help='使用水平翻转增强；运行开销增加（单独实验）')
    p.add_argument('--preview', type=int, default=3, help='生成多少张预测预览；默认3')
    p.add_argument('--activate', action='store_true', help='tune后把dev选出的配置写入configs/selected.json')
    p.add_argument('--out', default='', help='输出目录（默认项目下results/子目录）')
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    return p.parse_args()


def device_from(s: str) -> torch.device:
    if s=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('指定cuda但PyTorch未发现GPU。请确认你现有base环境的CUDA版PyTorch。')
    return torch.device('cuda' if ((s=='auto' and torch.cuda.is_available()) or s=='cuda') else 'cpu')


def partition(name: str) -> str:
    # A fixed, preannounced deterministic development/holdout split, 80/20.
    v=int(hashlib.sha256(('FDSNET_R_2026_v1/'+name).encode('utf-8')).hexdigest()[:8],16)
    return 'holdout' if v % 5 == 0 else 'dev'


def choose_config(name: str) -> Config:
    if name == 'auto':
        if not SELECTED.is_file():
            print('尚未激活优化配置，自动使用baseline（原版81.43%方案）',flush=True)
            return Config('baseline')
        raw=json.loads(SELECTED.read_text(encoding='utf-8'))
        allowed={x.name for x in fields(Config)}
        args={k:v for k,v in raw['config'].items() if k in allowed}
        return Config(**args)
    for c in variants(True):
        if name == c.name:
            return c
    names=[c.name for c in variants(True)]
    raise ValueError('未知variant：'+name+'；可选 '+', '.join(names))


def get_img_and_gt(root: Path, stem: str, need_gt: bool = True):
    with Image.open(image_for(root,stem)) as im:
        rgb=im.convert('RGB')
    gt=read_label(root/'SegmentationClass'/f'{stem}.png',(rgb.height,rgb.width)) if need_gt else None
    return rgb,gt


def gallery(out: Path, title: str, images: list[tuple[str,str]]) -> None:
    # Offline local HTML file – no network, JS libraries, or server required.
    cells='\n'.join(f'<figure><img src="{html.escape(rel)}" loading="lazy"><figcaption>{html.escape(name)}</figcaption></figure>'
                    for name,rel in images)
    page=f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1200px;margin:36px auto;padding:0 16px;background:#f6f7fb;color:#1b2735}}
h1{{font-size:26px}}p{{color:#5a6576}}.g{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}}
figure{{margin:0;padding:12px;background:white;border-radius:12px;box-shadow:0 2px 12px #d6dce6}}
img{{width:100%;object-fit:contain}}figcaption{{padding:8px 0}}</style></head><body>
<h1>{html.escape(title)}</h1><p>橙色：油污 · 蓝色：污渍 · 红色：划痕；可在图片同级目录查看原始mask.png。</p><section class="g">{cells}</section></body></html>'''
    (out/'gallery.html').write_text(page, encoding='utf-8')


def show_result(name: str, result: dict, n: int, cost: float) -> None:
    c=result['IoU']
    print(f'{name:23s} mIoU={result["mIoU"]:7.3f}%  Oil={c["oil"]:6.2f}  '
          f'Stain={c["stain"]:6.2f}  Scratch={c["scratch"]:6.2f}  '
          f'Model~{cost/max(n,1):.0f}ms',flush=True)


def run_tune(args, model, dev):
    root=dataset_root(Path(args.dataset))
    names=list_split(root,args.split)
    if args.limit:
        names=names[:args.limit]
    cfgs=variants(args.extended)
    out=Path(args.out) if args.out else RESULT_ROOT/f'tune_{args.split}'
    out.mkdir(parents=True,exist_ok=True)
    cms={c.name:{s:np.zeros((4,4),dtype=np.int64) for s in ('dev','holdout','all')} for c in cfgs}
    cost_ms=0.; engine_times=[]; per_image=[]; gallery_imgs=[]
    print(f'数据集：{root}  划分：{args.split}  图片：{len(names)}  候选：{len(cfgs)}',flush=True)
    print('油污模型IoU已达90.23%；本项目不使用任何油污最大面积删除规则。',flush=True)
    for idx,stem in enumerate(names,1):
        rgb,gt=get_img_and_gt(root,stem)
        probs,engine_ms=predict_probs(model,rgb,args.scale,dev,args.flip_tta)
        engine_times.append(engine_ms); cost_ms+=engine_ms
        rgb_np = np.asarray(rgb) if any(c.roi_safe for c in cfgs) else None
        roi=None
        if rgb_np is not None:
            roi,_=detect_safe_roi(rgb_np)
        bucket=partition(stem)
        row={'image':stem,'partition':bucket,'model_inference_ms':round(engine_ms,2)}
        for c in cfgs:
            t0=time.perf_counter()
            pred=apply(probs,c,rgb_np,roi)
            elapsed=(time.perf_counter()-t0)*1000
            update(cms[c.name]['all'],gt,pred)
            update(cms[c.name][bucket],gt,pred)
            row[f'{c.name}_post_ms']=round(elapsed,2)
            if idx<=args.preview and c.name in ('baseline','scratch_conf_cc','scratch_bridge_cc'):
                save_artifacts(rgb,pred,f'{stem}_{c.name}',out/'previews')
                gallery_imgs.append((f'{stem} / {c.name}',f'previews/{stem}_{c.name}_overlay.jpg'))
        per_image.append(row)
        del probs
        if idx==1 or idx%20==0 or idx==len(names):
            print(f'{idx}/{len(names)}  model={engine_ms:.0f}ms',flush=True)
    results={c.name:{k:scores(cms[c.name][k]) for k in ('dev','holdout','all')} for c in cfgs}
    counts={s:sum(partition(x)==s for x in names) for s in ('dev','holdout')}
    print('\n==== 开发子集：只用此子集选择方案 ====',flush=True)
    rows=[]
    base=results['baseline']['dev']
    for c in cfgs:
        cur=results[c.name]['dev']; held=results[c.name]['holdout']; all_=results[c.name]['all']
        row={'variant':c.name,'dev_mIoU':cur['mIoU'],'dev_delta_pp':round(cur['mIoU']-base['mIoU'],4),
             'holdout_mIoU':held['mIoU'],'holdout_delta_pp':round(held['mIoU']-results['baseline']['holdout']['mIoU'],4),
             'all_mIoU':all_['mIoU'],'all_defect_mIoU':all_['defect_mIoU'],
             'all_oil_IoU':all_['IoU']['oil'],'all_stain_IoU':all_['IoU']['stain'],
             'all_scratch_IoU':all_['IoU']['scratch'],
             'dev_oil_IoU':cur['IoU']['oil'],'dev_stain_IoU':cur['IoU']['stain'],
             'dev_scratch_IoU':cur['IoU']['scratch'],
             'holdout_oil_IoU':held['IoU']['oil'],'holdout_stain_IoU':held['IoU']['stain'],
             'holdout_scratch_IoU':held['IoU']['scratch']}
        rows.append(row)
        print(f'{c.name:24s} dev={cur["mIoU"]:.3f}% ({row["dev_delta_pp"]:+.3f}pp)'
              f'  holdout={held["mIoU"]:.3f}% ({row["holdout_delta_pp"]:+.3f}pp)',flush=True)
    # A conservative rule fixed BEFORE reviewing the user's holdout results.
    eligible=[]
    for c in cfgs:
        cur=results[c.name]['dev']
        if c.name=='baseline':continue
        if cur['mIoU']-base['mIoU']<.20:continue
        if any(cur['IoU'][n] < base['IoU'][n]-1.0 for n in ('oil','stain','scratch')):
            continue
        eligible.append(c)
    winner=max(eligible,key=lambda c:results[c.name]['dev']['mIoU']) if eligible else cfgs[0]
    print(f'\n按开发集选中的方案：{winner.name}（dev提升≥0.20pp，且各缺陷IoU退步<1pp）',flush=True)
    print(f'冻结方案之后再查看holdout：{results[winner.name]["holdout"]["mIoU"]:.3f}%'
          f' vs baseline {results["baseline"]["holdout"]["mIoU"]:.3f}%',flush=True)
    print('注意：原作者权重可能看过当前val图片；holdout仅防止本次后处理调参过拟合，不能排除原模型训练数据重叠。')
    save_rows(out/'summary.csv',rows)
    save_rows(out/'per_image_runtime.csv',per_image)
    for c in cfgs:
        save_cm(out/f'cm_{c.name}.csv',cms[c.name]['all'])
    save_json(out/'details.json',{
        'dataset':str(root),'split':args.split,'images':len(names),
        'dev_images':counts['dev'],'holdout_images':counts['holdout'],
        'model_times_ms':{'mean':float(np.mean(engine_times)),'median':float(np.median(engine_times))},
        'scale':args.scale,'flip_tta':args.flip_tta,
        'variants':{c.name:c.to_dict() for c in cfgs},'results':results,
        'selected_on_development_only':winner.name,
        'limitations':'Pretrained FDSNet may overlap this val split; not independent generalization.',
    })
    selected={'config':winner.to_dict(),'selection':'development-only','split':args.split,'scale':args.scale,
              'dev_mIoU':results[winner.name]['dev']['mIoU'],
              'holdout_mIoU':results[winner.name]['holdout']['mIoU'],
              'warning':'This config is NOT proven on a truly independent test set.'}
    save_json(out/'candidate_selected.json',selected)
    if args.activate:
        SELECTED.parent.mkdir(parents=True,exist_ok=True)
        save_json(SELECTED,selected)
        print(f'已写入启用配置: {SELECTED}',flush=True)
    if gallery_imgs:
        gallery(out,'FDSNet 原版与后处理候选对比',gallery_imgs)
    if len(names)==240 and args.scale==.75 and args.split=='val' and not args.flip_tta and not args.limit:
        base_miou=results['baseline']['all']['mIoU']
        if abs(base_miou-81.4263)>.10:
            print(f'【注意】原版应约81.4263%，本次{base_miou:.4f}%，超出容差！需核对数据/模型/标注处理。',flush=True)
        else:
            print(f'基准复现通过: {base_miou:.4f}%≈81.4263%',flush=True)
    print(f'\n结果：{out / "summary.csv"}',flush=True)


def run_eval(args,model,dev):
    root=dataset_root(Path(args.dataset))
    names=list_split(root,args.split)
    if args.limit:names=names[:args.limit]
    cfg=choose_config(args.variant)
    out=Path(args.out) if args.out else RESULT_ROOT/f'eval_{args.split}_{cfg.name}'
    out.mkdir(parents=True,exist_ok=True)
    cm=np.zeros((4,4),dtype=np.int64);cm_baseline=np.zeros((4,4),dtype=np.int64)
    cost=0.;post_cost=0.; rows=[];gallery_imgs=[]
    for idx,stem in enumerate(names,1):
        rgb,gt=get_img_and_gt(root,stem)
        prob,ms=predict_probs(model,rgb,args.scale,dev,args.flip_tta)
        base=prob.argmax(0).astype(np.uint8)
        t0=time.perf_counter()
        pred=apply(prob,cfg,np.asarray(rgb) if cfg.roi_safe else None)
        post_ms=(time.perf_counter()-t0)*1000
        update(cm,gt,pred);update(cm_baseline,gt,base)
        cost+=ms;post_cost+=post_ms
        # Image-level class errors to locate actual pain points, not just aggregate numbers.
        row={'image':stem,'bucket':partition(stem),'model_ms':round(ms,2),'post_ms':round(post_ms,2)}
        for cls,name in enumerate(CLASSES[1:],1):
            truth=gt==cls;guess=pred==cls
            tp=int(np.count_nonzero(truth&guess))
            fn=int(np.count_nonzero(truth&(~guess)))
            fp=int(np.count_nonzero((~truth)&guess))
            row[f'{name}_TP']=tp;row[f'{name}_FN']=fn;row[f'{name}_FP']=fp
            row[f'{name}_IoU']=round(100*tp/max(tp+fn+fp,1),3) if tp+fn+fp>0 else ''
        rows.append(row)
        if idx<=args.preview:
            save_artifacts(rgb,base,f'{stem}_baseline',out/'preview')
            save_artifacts(rgb,pred,f'{stem}_{cfg.name}',out/'preview')
            gallery_imgs.extend([(f'{stem}/baseline',f'preview/{stem}_baseline_overlay.jpg'),
                                 (f'{stem}/{cfg.name}',f'preview/{stem}_{cfg.name}_overlay.jpg')])
        if idx==1 or idx%20==0 or idx==len(names):
            print(f'{idx}/{len(names)}',flush=True)
    stats=scores(cm);baseline=scores(cm_baseline)
    s={'variant':cfg.name,'images':len(names),'split':args.split,'scale':args.scale,
       'model_ms_mean':round(cost/len(names),2),'post_ms_mean':round(post_cost/len(names),2),
       'baseline':baseline,'current':stats,'delta_mIoU_pp':round(stats['mIoU']-baseline['mIoU'],4),
       'class_order':CLASSES,
       'warning':'FDSNet checkpoint training data could overlap this evaluation split.'}
    save_json(out/'summary.json',s)
    save_rows(out/'summary.csv',[{'variant':cfg.name,'images':len(names),
        'mIoU':stats['mIoU'],'baseline_mIoU':baseline['mIoU'],'delta_pp':s['delta_mIoU_pp'],
        'oil_IoU':stats['IoU']['oil'],'stain_IoU':stats['IoU']['stain'],
        'scratch_IoU':stats['IoU']['scratch'],'model_ms':s['model_ms_mean'],
        'postprocess_ms':s['post_ms_mean']}])
    save_rows(out/'per_image.csv',rows)
    save_cm(out/'confusion_matrix.csv',cm)
    save_cm(out/'baseline_confusion_matrix.csv',cm_baseline)
    if gallery_imgs:gallery(out,'FDSNet 检测结果：原版 vs 改进',gallery_imgs)
    # Highest-error images can be sorted without changing metrics.
    for cls in ('oil','stain','scratch'):
        worst=sorted(rows,key=lambda r:r[f'{cls}_FN']+r[f'{cls}_FP'],reverse=True)[:20]
        save_rows(out/f'worst_{cls}_top20.csv',worst)
    show_result('原始FDSNet',baseline,len(names),cost)
    show_result(cfg.name,stats,len(names),cost)
    print(f'变化：{s["delta_mIoU_pp"]:+.4f}个百分点；结果：{out}',flush=True)


def run_predict(args,model,dev):
    if not args.image:
        raise ValueError('predict 模式必须提供 --image "图片路径或文件夹"')
    path=Path(args.image)
    if path.is_dir():
        files=sorted(p for p in path.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))
    elif path.is_file():
        files=[path]
    else:
        raise FileNotFoundError(path)
    if not files:
        raise RuntimeError('没有找到图片')
    cfg=choose_config(args.variant)
    out=Path(args.out) if args.out else RESULT_ROOT/f'predict_{cfg.name}'
    out.mkdir(parents=True,exist_ok=True)
    images=[];rows=[]
    for idx,path in enumerate(files,1):
        with Image.open(path) as im: rgb=im.convert('RGB')
        probs,ms=predict_probs(model,rgb,args.scale,dev,args.flip_tta)
        t0=time.perf_counter()
        mask=apply(probs,cfg,np.asarray(rgb) if cfg.roi_safe else None)
        post_ms=(time.perf_counter()-t0)*1000
        # Prevent collisions when files with same stem in different folders.
        stem=f'{idx:04d}_{path.stem}'
        save_artifacts(rgb,mask,stem,out)
        images.append((path.name,f'{stem}_overlay.jpg'))
        rows.append({'image':str(path),'variant':cfg.name,'oil_px':int((mask==1).sum()),
                     'stain_px':int((mask==2).sum()),'scratch_px':int((mask==3).sum()),
                     'model_ms':round(ms,2),'post_ms':round(post_ms,2)})
        print(f'{idx}/{len(files)} {path.name}: oil/stain/scratch='
              f'{rows[-1]["oil_px"]}/{rows[-1]["stain_px"]}/{rows[-1]["scratch_px"]}',flush=True)
    save_rows(out/'predictions.csv',rows)
    gallery(out,'FDSNet 手机屏幕缺陷检测结果',images)
    print(f'输出位置：{out}；双击 gallery.html 浏览所有检测图',flush=True)


def main():
    args=opts()
    if args.limit<0 or args.preview<0:
        raise ValueError('--limit和--preview不能小于0')
    dev=device_from(args.device)
    model=load_model(dev,Path(args.weights))
    if args.command=='check':
        print('模型严格匹配成功；本脚本不改变原训练权重或原项目文件。')
    elif args.command=='tune':
        run_tune(args,model,dev)
    elif args.command=='evaluate':
        run_eval(args,model,dev)
    elif args.command=='predict':
        run_predict(args,model,dev)


if __name__=='__main__':
    main()
