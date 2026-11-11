"""Dataset GT defect-size distribution. Does NOT claim real-world millimeter sizes."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from core.dataset import image_by_stem, read_gt, read_ids
from core.metrics import CLASSES
from core.artifacts import write_csv

BINS=[(0,32,'1-32 pixels'),(33,256,'33-256 pixels'),(257,2048,'257-2048 pixels'),(2049,10**20,'2049+ pixels')]


def analyze(dataset:Path,split:str,output:Path,max_images:int=0):
    import cv2
    ids=read_ids(dataset,split)
    if max_images>0:ids=ids[:max_images]
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    for idx,stem in enumerate(ids,1):
        with Image.open(image_by_stem(dataset,stem)) as im: shape=(im.height,im.width)
        gt=read_gt(dataset/'SegmentationClass'/f'{stem}.png',shape)
        for cls in range(1,4):
            binary=(gt==cls).astype(np.uint8)
            n,labels,stats,center=cv2.connectedComponentsWithStats(binary,8)
            for i in range(1,n):
                x,y,w,h,area=map(int,stats[i])
                label=next(name for lo,hi,name in BINS if lo<=area<=hi)
                records.append({'image':stem,'class':CLASSES[cls],'area_px':area,
                    'bbox_x':x,'bbox_y':y,'bbox_w':w,'bbox_h':h,
                    'image_width':shape[1],'image_height':shape[0],
                    'area_pct':round(area/(shape[0]*shape[1])*100,7),
                    'size_bin':label})
        if idx%20==0 or idx==len(ids):print(f'[PROGRESS] {idx}/{len(ids)} GT尺寸分析',flush=True)
    summary=[]
    for cls in CLASSES[1:]:
        for _,_,label in BINS:
            selected=[r for r in records if r['class']==cls and r['size_bin']==label]
            summary.append({'class':cls,'size_bin':label,'regions':len(selected),
                            'median_area_px':float(np.median([r['area_px'] for r in selected])) if selected else 0})
    write_csv(output/'gt_defect_regions.csv',records,
              ['image','class','area_px','bbox_x','bbox_y','bbox_w','bbox_h','image_width','image_height','area_pct','size_bin'])
    write_csv(output/'size_summary.csv',summary,['class','size_bin','regions','median_area_px'])
    (output/'analysis_info.json').write_text(json.dumps({
      'dataset':str(dataset),'split':split,'images':len(ids),
      'components':len(records),'bins':BINS,
      'important':'area counts original-image GT pixels; bins are descriptive exploratory thresholds, not calibrated physical defect sizes; connected components need not equal physical defect instances'
    },ensure_ascii=False,indent=2),encoding='utf-8')
    from html import escape
    rows=''.join(f'<tr><td>{escape(r["class"])}</td><td>{escape(r["size_bin"])}</td><td>{r["regions"]}</td><td>{r["median_area_px"]:.0f}</td></tr>' for r in summary)
    html=f'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>MSD缺陷面积分析</title><style>body{{font:16px "Microsoft YaHei",sans-serif;background:#f3f6fb;color:#23354b;padding:40px}}main{{max-width:860px;margin:auto;background:white;padding:32px;border-radius:16px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:14px;border-bottom:1px solid #e5ebf3;text-align:left}}p{{color:#677891}}</style></head><body><main><h1>MSD 缺陷面积分布</h1><p>{len(ids)}张图片 · {len(records)}个GT连通域 · 面积以原始标注像素计（不是毫米）。</p><table><tr><th>类别</th><th>面积分组</th><th>连通域数量</th><th>组内面积中位数</th></tr>{rows}</table><p>仅为缺陷大小分布描述；连通域不一定对应一个完整物理缺陷。不能仅凭该表断言真实工厂的微米级检测能力。</p></main></body></html>'''
    (output/'size_report.html').write_text(html,encoding='utf-8')
    print(f'[OUTPUT] {output}\n[REPORT] {output/"size_report.html"}',flush=True)
    return output
