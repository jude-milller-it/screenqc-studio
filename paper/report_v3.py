# -*- coding: utf-8 -*-
"""V3 evidence-first, narrative dissertation report. All measurements are computed locally.

Sections share one structured source, figures have axis provenance and captions,
tables are native Word/HTML/text rather than raster screenshots.
"""
from __future__ import annotations
import html
import json
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from paper.workflow import make_structured, _safe
from paper.evidence import research_evidence

MODEL_LABELS = {
 'deeplab_base':'DeepLabV3+ / 基础训练',
 'deeplab':'DeepLabV3+ / 完整优化','segformer_raw':'SegFormer / 原始512',
 'segformer_scale':'SegFormer / 高分辨率',
 'segformer':'SegFormer / 基础规则',
 'segformer_oil':'SegFormer / 完整推理优化','segformer_edge_guard':'SegFormer / 全优化+显示ROI实验',
 'fdsnet':'FDSNet / 作者原始',
 'fdsnet_tuned':'FDSNet / 推理校正候选',
 'segformer_roi':'SegFormer / 尺度+ROI',
 'segformer_conf':'SegFormer / 尺度+置信度',
 'segformer_post_bad':'SegFormer / 旧后处理(负优化)',
 'DeepLabV3_scheme3':'DeepLabV3+',
 'SegFormer_640x1152_C':'SegFormer / 基础规则',
 'FDSNet_original':'FDSNet',
}
CLASS_CN={'background':'背景','oil':'油污','stain':'污渍','scratch':'划痕'}
CLASS_EN={'background':'Background','oil':'Oil','stain':'Stain','scratch':'Scratch'}
COLORS={'deeplab_base':'#9aa6b7','fdsnet_tuned':'#aa6b42','segformer_roi':'#4e86be',
        'segformer_conf':'#77a0c6','segformer_post_bad':'#bd6e65','deeplab':'#64748b','segformer_raw':'#9cb4cc','segformer_scale':'#85a3d1',
        'segformer':'#315caa','segformer_oil':'#176e89','fdsnet':'#a06b45',
        'DeepLabV3_scheme3':'#64748b','SegFormer_640x1152_C':'#315caa','FDSNet_original':'#a06b45'}


def _f(v, decimals=2):
    return '—' if v is None else f'{v:.{decimals}f}'


def _key(data,key):
    return next((m for m in data['models'] if m['id']==key),None)


def _name(m):return MODEL_LABELS.get(m['id'],m['id'])


def _delta(a,b,k):
    x=a['summary'].get(k);y=b['summary'].get(k)
    return x-y if x is not None and y is not None else None


def _stage_note(data):
    ids={m['id'] for m in data['models']}
    if {'segformer','segformer_oil'}<=ids:
        return '两者使用同一SegFormer最终训练权重；完整推理优化方案=高分辨率+ROI+类别阈值+油污决策调整。相对基础规则版的增量仅来自最后一项。'
    return '若缺少基础与优化两组同权重结果，当前报告不能计算可控的油污优化增益。'


MAIN_IDS=('deeplab_base','deeplab','segformer_raw','segformer_oil','fdsnet_tuned',
          'DeepLabV3_scheme3','SegFormer_640x1152_C','FDSNet_original')
def _primary_models(data):
    models=[m for m in data['models'] if m['id'] in MAIN_IDS]
    return models or data['models']

def _narrative(data):
    ms=_primary_models(data)
    ranked=[m for m in ms if m['summary'].get('mIoU_pct') is not None]
    ranked.sort(key=lambda m:m['summary']['mIoU_pct'],reverse=True)
    top=ranked[0] if ranked else None
    baseline=_key(data,'segformer') or _key(data,'SegFormer_640x1152_C')
    oil=_key(data,'segformer_oil')
    lines={}
    lines['protocol']='本研究采用MSD多类别像素分割，统一读取相同图片与标注。背景、油污、污渍、划痕的类别映射分别为0、1、2、3；计算采用全数据集累积混淆矩阵而非逐图指标的简单平均。'
    if top:
        lines['overall']=f'在本次评测的{len(ms)}项配置中，{_name(top)}的总体mIoU最高（{_f(top["summary"]["mIoU_pct"])}%）。但mIoU只是总体重叠程度，不能取代分缺陷类别的误检与漏检分析。'
    else:lines['overall']='总体mIoU缺失，不能进行排名。'
    if baseline and oil:
        dp=_delta(oil,baseline,'mIoU_pct')
        oi=oil['classes']['oil'];bi=baseline['classes']['oil']
        p=(oi['Precision_pct'] or 0)-(bi['Precision_pct'] or 0)
        r=(oi['Recall_pct'] or 0)-(bi['Recall_pct'] or 0)
        d=(oi['IoU_pct'] or 0)-(bi['IoU_pct'] or 0)
        lines['ablation']=(f'在同一SegFormer权重与相同输入分辨率的条件下，油污专属决策调整使油污Precision变化{p:+.2f}个百分点、油污IoU变化{d:+.2f}个百分点，'
                         f'对应Recall变化{r:+.2f}个百分点；总mIoU变化{dp:+.2f}个百分点。'
                         '这是推理规则增益，不是训练或网络架构贡献。')
    else:lines['ablation']='缺少同权重基础版或油污优化版的完整数据，暂不推断优化收益。'
    for cls in ('oil','stain','scratch'):
        relevant=[m for m in ms if m['classes'][cls]['IoU_pct'] is not None]
        if relevant:
            winner=max(relevant,key=lambda m:m['classes'][cls]['IoU_pct'])
            rec=max(relevant,key=lambda m:m['classes'][cls]['Recall_pct'] or 0)
            prec=max(relevant,key=lambda m:m['classes'][cls]['Precision_pct'] or 0)
            lines[cls]=(f'{CLASS_CN[cls]}IoU最优的是{_name(winner)}（{_f(winner["classes"][cls]["IoU_pct"])}%）；'
                       f'像素召回率最高的是{_name(rec)}（{_f(rec["classes"][cls]["Recall_pct"])}%），'
                       f'精确率最高的是{_name(prec)}（{_f(prec["classes"][cls]["Precision_pct"])}%）。'
                       '这些指标反映不同误差类型，不能仅凭单项得分评定模型。')
        else:lines[cls]='该类别缺少有效数据。'
    ranked_time=[m for m in ms if m['summary'].get('mean_pipeline_ms') is not None]
    if ranked_time:
        quick=min(ranked_time,key=lambda m:m['summary']['mean_pipeline_ms'])
        lines['runtime']=f'本次运行中耗时较低的是{_name(quick)}（{_f(quick["summary"]["mean_pipeline_ms"])} ms/图）。速度依赖GPU、预处理和分辨率，不能视为跨硬件基准。'
    else:lines['runtime']='缺少运行时数据。'
    lines['limits']='当前val图片曾用于调参；FDSNet公开权重可能与验证图片存在训练重叠；FDSNet污渍校正系数1.10为探索候选且其开发集选择结果为原版；不同架构的训练与推理配置不完全一致。本文报告的是像素级性能，而非产品级漏检率或产线验收结论。'
    raw=_key(data,'segformer_raw')
    if raw and oil:
        gain=_delta(oil,raw,'mIoU_pct')
        lines['conclusion']=(f'同权重SegFormer从原始推理到最终推理方案的mIoU变化为{gain:+.2f}个百分点；'
          '这是多项推理配置的合并效果。油污专属阈值的单独增量应以基础规则版为对照，不能把全部改善都归功于油污阈值。'
          '跨架构表现与训练策略同时变化，尚不能证明Transformer本身导致优势。')
    else:
        lines['conclusion']='不同方案在总体重叠精度与像素召回之间存在取舍；缺失的受控配置不做推断。'
    return lines


def _analysis_sections(data, ai):
    default=_narrative(data)
    if isinstance(ai,dict):
        out={}
        for key,txt in default.items():
            content=ai.get(key)
            out[key]=str(content).strip() if isinstance(content,str) and content.strip() else txt
        return out
    return default


def _tables(data):
    models=data['models'];major=_primary_models(data);tab={}
    tab['performance']=[['方案','mIoU %','缺陷mIoU %','Precision %','Recall %','F1 %','ms/图']]
    for m in major:
        t=m['summary'];tab['performance'].append([_name(m)]+[_f(t.get(k)) for k in ('mIoU_pct','defect_mIoU_pct','mPrecision_pct','mRecall_pct','mF1_pct','mean_pipeline_ms')])
    for c in ('background','oil','stain','scratch'):
        tab[c]=[['方案','IoU %','Precision %','Recall %','F1 / Dice %','FP像素','FN像素']]
        for m in major:
            t=m['classes'][c]
            tab[c].append([_name(m)]+[_f(t.get(k)) for k in ('IoU_pct','Precision_pct','Recall_pct','F1_Dice_pct')]
                          +[str(int(t[k])) if t.get(k) is not None else '—' for k in ('FP','FN')])
    # Every benchmark run can export a native editable ablation table,
    # including beneficial and harmful configurations (no post-hoc filtering).
    seg=[m for m in models if m['id'].startswith('segformer')]
    if seg:
        tab['ablation']=[['SegFormer推理配置','mIoU %','油污IoU %','污渍IoU %','划痕IoU %','油污Precision %','划痕Recall %']]
        for m in seg:
            tab['ablation'].append([_name(m),_f(m['summary'].get('mIoU_pct')),
                *[_f(m['classes'][c].get('IoU_pct')) for c in ('oil','stain','scratch')],
                _f(m['classes']['oil'].get('Precision_pct')),
                _f(m['classes']['scratch'].get('Recall_pct'))])
    tab['stage']=[['阶段','输入及策略','权重关系','用途']]
    for m in models:
        descriptions={
         'deeplab_base':('原始DeepLabV3+方案0','独立基础训练权重','传统CNN基线'),
         'deeplab':('完整DeepLabV3+方案3','独立完整优化权重','卷积优化对照'),
         'segformer_raw':('512×512，原始Argmax','同一最终SegFormer权重','原始推理参考'),
         'segformer_scale':('640×1152，原始Argmax','同一最终SegFormer权重','仅输入尺度变化'),
         'segformer':('640×1152，ROI+类别阈值','同一最终SegFormer权重','基础推理策略'),
         'segformer_oil':('尺度+ROI+类别阈值+油污+0.30','同一最终SegFormer权重','在基础推理组合上再增油污决策；完整方案由多项规则构成'),
         'fdsnet':('0.75缩放、原作者原始推理','原作者权重','未优化外部参考'),
         'fdsnet_tuned':('0.75缩放+污渍概率系数1.10','与fdsnet共用作者权重','验证集探索参数，开发子集未选中此配置'),
         'segformer_roi':('640×1152，ROI，不加类别阈值','同一最终SegFormer权重','分解ROI单项效果'),
         'segformer_conf':('640×1152，类别阈值，无ROI','同一最终SegFormer权重','分解置信度单项效果'),
         'segformer_edge_guard':('最终推理+中心显示区域ROI内缩16px','同一最终SegFormer权重','屏幕边缘误检专项实验，未证明无损召回'),
         'segformer_edge_guard':('最终推理+中心显示区域ROI内缩16px','同一最终SegFormer权重','屏幕边缘误检专项实验，未证明无损召回'),
         'segformer_post_bad':('640×1152，旧形态学+面积过滤','同一最终SegFormer权重','负优化参照，不作为推荐'),
        }
        v=descriptions.get(m['id'],('配置未知','未记录','历史导入'))
        tab['stage'].append([_name(m),*v])
    return tab


def _plot_charts(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from visualization.fonts import configure_chinese_matplotlib
    configure_chinese_matplotlib()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':9,'axes.edgecolor':'#d3dbe4',
        'axes.labelcolor':'#283649','text.color':'#233147','xtick.color':'#64748b','ytick.color':'#64748b',
        'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
    ms=_primary_models(data);labels=[_name(m).replace(' / ','\n') for m in ms]
    def decorate(ax,ylabel,ylim=(0,100),grid=True):
        ax.set_ylabel(ylabel);ax.set_ylim(*ylim)
        if grid:ax.grid(axis='y',color='#e9eef4',linewidth=.7);ax.set_axisbelow(True)
        for spine in ('top','right'):ax.spines[spine].set_visible(False)
    def save(fig,name):
        fig.savefig(out/name,dpi=170,bbox_inches='tight',facecolor='white');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.7))
    vs=[m['summary']['mIoU_pct'] or 0 for m in ms]
    bars=ax.bar(range(len(ms)),vs,color=[COLORS.get(m['id'],'#6c819a') for m in ms],width=.62)
    ax.set_xticks(range(len(ms)),labels);decorate(ax,'mIoU (%)')
    for b,v in zip(bars,vs):ax.text(b.get_x()+b.get_width()/2,min(98,v+1.5),f'{v:.2f}',ha='center',fontsize=9)
    fig.tight_layout();save(fig,'01_overall_miou.png')
    defects=['oil','stain','scratch']
    for metric,label,fil in [('IoU_pct','IoU (%)','02_defect_iou.png'),('Recall_pct','像素召回率 (%)','03_defect_recall.png')]:
        fig,ax=plt.subplots(figsize=(9,3.65));x=np.arange(3);w=.83/len(ms)
        for i,m in enumerate(ms):
            vals=[m['classes'][c][metric] or 0 for c in defects]
            ax.bar(x+(i-(len(ms)-1)/2)*w,vals,width=w*.92,label=_name(m),color=COLORS.get(m['id'],'#6c819a'))
        ax.set_xticks(x,[CLASS_CN[c] for c in defects]);decorate(ax,label);ax.legend(ncol=min(3,len(ms)),frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.22))
        fig.tight_layout();save(fig,fil)
    for c in defects:
        fig,ax=plt.subplots(figsize=(9,3.45));x=np.arange(len(ms));w=.19
        for k,metric,color in [(0,'IoU_pct','#1f477a'),(1,'Precision_pct','#6094c3'),(2,'Recall_pct','#bb8357'),(3,'F1_Dice_pct','#52a6a0')]:
            ax.bar(x+(k-1.5)*w,[m['classes'][c][metric] or 0 for m in ms],w*.93,color=color,label={'IoU_pct':'IoU','Precision_pct':'Precision','Recall_pct':'Recall','F1_Dice_pct':'F1'}[metric])
        ax.set_xticks(x,labels);decorate(ax,f'{CLASS_CN[c]} (%)');ax.legend(frameon=False,ncol=4,loc='upper center',bbox_to_anchor=(.5,1.17));fig.tight_layout();save(fig,f'04_{c}_metrics.png')
    fig,ax=plt.subplots(figsize=(8.5,3.5))
    for m in ms:
        recall=m['classes']['oil']['Recall_pct'];precision=m['classes']['oil']['Precision_pct']
        if recall is not None and precision is not None:
            ax.scatter([recall],[precision],s=130,color=COLORS.get(m['id'],'#6c819a'),label=_name(m),edgecolor='white',linewidth=1.3,zorder=3)
    ax.set_xlim(0,100);decorate(ax,'油污Precision (%)');ax.set_xlabel('油污Recall (%)');ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),frameon=False,fontsize=8)
    fig.tight_layout();save(fig,'05_oil_operating_points.png')
    fig,ax=plt.subplots(figsize=(9,3.4));lat=[m['summary']['mean_pipeline_ms'] or 0 for m in ms];mx=max(lat+[1]);bars=ax.bar(range(len(ms)),lat,color=[COLORS.get(m['id'],'#6c819a') for m in ms],width=.55)
    ax.set_xticks(range(len(ms)),labels);decorate(ax,'毫秒/图 (ms)',(0,mx*1.2))
    for bar,v in zip(bars,lat):ax.text(bar.get_x()+bar.get_width()/2,v+mx*.025,f'{v:.0f}',ha='center')
    fig.tight_layout();save(fig,'06_latency.png')
    base=_key(data,'segformer');improved=_key(data,'segformer_oil')
    if base and improved:
        fig,ax=plt.subplots(figsize=(7,3.15))
        keys=['mIoU_pct','oil_IoU','oil_Precision','oil_Recall','stain_IoU','scratch_IoU']
        before=[base['summary']['mIoU_pct']]+[base['classes'][x]['IoU_pct'] for x in ('oil',)] + [base['classes']['oil']['Precision_pct'],base['classes']['oil']['Recall_pct'],base['classes']['stain']['IoU_pct'],base['classes']['scratch']['IoU_pct']]
        after=[improved['summary']['mIoU_pct'],improved['classes']['oil']['IoU_pct'],improved['classes']['oil']['Precision_pct'],improved['classes']['oil']['Recall_pct'],improved['classes']['stain']['IoU_pct'],improved['classes']['scratch']['IoU_pct']]
        diffs=[(b or 0)-(a or 0) for a,b in zip(before,after)]
        y=np.arange(len(keys));ax.barh(y,diffs,color=['#17866d' if v>=0 else '#bc6d59' for v in diffs])
        ax.set_yticks(y,['总体mIoU','油污IoU','油污Precision','油污Recall','污渍IoU','划痕IoU']);ax.axvline(0,color='#334155',lw=1)
        for i,v in enumerate(diffs):ax.text(v+(.04 if v>=0 else -.04),i,f'{v:+.2f}',ha='left' if v>=0 else 'right',va='center')
        ax.invert_yaxis();ax.set_xlabel('变化（百分点；非相对增长率）');ax.margins(x=.17)
        for sp in ('top','right','left'):ax.spines[sp].set_visible(False)
        fig.tight_layout();save(fig,'07_segformer_delta.png')
    ablation_order=('segformer_raw','segformer_scale','segformer_roi','segformer_conf','segformer','segformer_oil','segformer_post_bad')
    seg=sorted((m for m in data['models'] if m['id'].startswith('segformer') and m['id'] in ablation_order),key=lambda m:ablation_order.index(m['id']))
    if len(seg)>1:
        fig,ax=plt.subplots(figsize=(9,3.6));x=np.arange(len(seg))
        for cls,color in (('oil','#3b82a0'),('stain','#d39452'),('scratch','#6785b1')):
            ax.plot(x,[m['classes'][cls]['IoU_pct'] or 0 for m in seg],
                    marker='o',lw=1.6,markersize=5,label=CLASS_CN[cls],color=color)
        ax.set_xticks(x,[_name(m).replace('SegFormer / ','') for m in seg],rotation=18,ha='right')
        decorate(ax,'类别IoU (%)',(0,100));ax.legend(frameon=False,ncol=3)
        fig.tight_layout();save(fig,'10_segformer_ablation.png')
    # Normalized confusion matrices, actual GT row/pred column.
    import numpy as np
    import csv
    src=Path(data['source_dir'])
    for m in data['models']:
        f=src/m['id']/'confusion_matrix.csv'
        if not f.is_file():continue
        with f.open(encoding='utf-8-sig',newline='') as r:raw=list(csv.DictReader(r))
        if len(raw)!=4:continue
        mat=np.asarray([[int(row[c]) for c in ('background','oil','stain','scratch')] for row in raw],dtype=float)
        sums=mat.sum(axis=1,keepdims=True);norm=np.divide(mat,sums,out=np.zeros_like(mat),where=sums>0)*100
        fig,ax=plt.subplots(figsize=(5.4,4.4));im=ax.imshow(norm,cmap='Blues',vmin=0,vmax=100)
        ax.set_xticks(range(4),[CLASS_CN[c] for c in ('background','oil','stain','scratch')]);ax.set_yticks(range(4),[CLASS_CN[c] for c in ('background','oil','stain','scratch')]);ax.set_xlabel('预测');ax.set_ylabel('真实标注')
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{norm[i,j]:.1f}',ha='center',va='center',fontsize=9,color='white' if norm[i,j]>55 else '#172a45')
        fig.colorbar(im,ax=ax,label='按真实类别归一化 (%)',fraction=.046)
        fig.tight_layout();save(fig,'cm_'+m['id']+'.png')



def _archive_research_figures(out):
    """Label archived training and oil-tuning figures separately from current runs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from visualization.fonts import configure_chinese_matplotlib
    configure_chinese_matplotlib()
    from core.settings import ROOT
    logs=ROOT/'research'/'logs_segformer_mit_b2'
    found=sorted(logs.glob('loss_*/epoch_miou.txt'))
    if found:
        file=found[-1]
        try: values=[float(x.strip()) for x in file.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
        except ValueError:values=[]
        if len(values)>=3:
            fig,ax=plt.subplots(figsize=(9,3.4))
            ax.plot(range(1,len(values)+1),values,color='#285b99',lw=1.65)
            ax.set_xlabel('验证回调记录序号（不一定等于epoch）');ax.set_ylabel('日志记录的mIoU (%)')
            ax.set_ylim(0,100);ax.grid(alpha=.16);ax.spines['right'].set_visible(False);ax.spines['top'].set_visible(False)
            fig.tight_layout();fig.savefig(out/'08_segformer_training_history.png',dpi=170);plt.close(fig)
    curve=ROOT/'research'/'experiments'/'oil_optimization_results'/'precision_recall_curve.csv'
    if curve.is_file():
        with curve.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        entries={}
        for row in rows:
            try:
                if abs(float(row.get('oil_margin') or 0))>1e-9:continue
                delta=float(row.get('oil_delta') or 0)
                entries[delta]=tuple(float(row[k]) for k in ('oil_Precision_pct','oil_Recall_pct','oil_IoU_pct'))
            except (ValueError,TypeError,KeyError):continue
        if entries:
            x=sorted(entries);fig,ax=plt.subplots(figsize=(9,3.5))
            for i,name,color in ((0,'Precision','#285b99'),(1,'Recall','#bc8556'),(2,'IoU','#299c90')):
                ax.plot(x,[entries[z][i] for z in x],marker='o',markersize=3.4,lw=1.65,label=name,color=color)
            ax.axvline(.3,ls='--',lw=1,color='#94a3b8',label='已测试+0.30')
            ax.set_ylim(60,100);ax.set_xlabel('油污阈值相对增量（同一最终权重）');ax.set_ylabel('像素指标 (%) — 纵轴60%起')
            ax.grid(alpha=.16);ax.legend(frameon=False,ncol=4,loc='upper center',bbox_to_anchor=(.5,1.17))
            ax.spines['right'].set_visible(False);ax.spines['top'].set_visible(False)
            fig.tight_layout();fig.savefig(out/'09_oil_threshold_sweep.png',dpi=170);plt.close(fig)


def _collect_case_panels(data,out):
    """One side-by-side original/GT/model prediction for each class if evaluation saved examples."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    from core.dataset import image_by_stem,read_gt
    exp=data.get('experiment') or {}
    ds=Path(exp.get('dataset',''))
    if not ds.is_dir():return []
    root=Path(data['source_dir'])
    outputs=[]
    for cl in ('oil','stain','scratch'):
        folders=[(m,root/m['id']/'examples'/cl/'masks') for m in data['models']]
        active=[(m,folder) for m,folder in folders if folder.is_dir()]
        if not active:continue
        candidates=[x for x in active[0][1].glob('*.png')]
        for j,mask in enumerate(candidates[:2],1):
            # Filename uses truncated stem + _sha1; robustly look up original via per_image stems.
            stem=mask.stem.rsplit('_',1)[0]
            try:
                image=image_by_stem(ds,stem)
                with Image.open(image) as im:rgb=im.convert('RGB')
                gt=read_gt(ds/'SegmentationClass'/f'{stem}.png',(rgb.height,rgb.width))
            except (FileNotFoundError,ValueError):continue
            from core.artifacts import RGB_COLORS
            variants=[('原始图',rgb),('人工标签',_overlay(rgb,gt,RGB_COLORS))]
            for m,folder in active:
                candidate=folder/mask.name
                if candidate.is_file():
                    with Image.open(candidate) as im:label=np.asarray(im,dtype=np.uint8)
                    variants.append((_name(m),_overlay(rgb,label,RGB_COLORS)))
            from visualization.fonts import configure_chinese_matplotlib
            configure_chinese_matplotlib()
            # At most three columns, so 6-7 models flow into two/three readable rows.
            from math import ceil
            columns=min(3,len(variants));rows=ceil(len(variants)/columns)
            fig,ax=plt.subplots(rows,columns,figsize=(columns*4.4,rows*2.9),squeeze=False)
            axes=ax.flatten()
            for a,(label,picture) in zip(axes,variants):
                a.imshow(picture);a.set_title(label,fontsize=10);a.axis('off')
            for a in axes[len(variants):]:a.axis('off')
            fig.tight_layout(pad=.8,h_pad=1.1)
            path=out/f'case_{cl}_{j:02d}.png';fig.savefig(path,dpi=150,bbox_inches='tight');plt.close(fig);outputs.append(path.name)
    return outputs


def _overlay(rgb,mask,colors):
    from PIL import Image
    import numpy as np
    a=np.asarray(rgb,dtype=np.uint8).copy();active=mask>0
    color=np.asarray(colors,dtype=np.uint8)[np.clip(mask,0,3)]
    a[active]=(a[active].astype(np.float32)*.5+color[active].astype(np.float32)*.5).astype(np.uint8)
    return Image.fromarray(a)


def _parts(data,ai_text,figures):
    """Normalized sequence for Markdown, DOCX, HTML, PDF. Paragraphs and evidence interleave."""
    lines=_analysis_sections(data,ai_text)
    tables=_tables(data)
    ms=data['models'];exp=data.get('experiment') or {}
    parts=[]
    evidence=research_evidence(data)
    def header(n,title):parts.append(('h2',f'{n}. {title}'))
    def para(txt):parts.append(('p',txt))
    def table(name,caption):parts.append(('table',name,caption))
    def fig(name,caption):
        if name in figures:parts.append(('figure',name,caption))
    header('01','执行摘要｜先看最重要的发现')
    para(lines['overall'])
    para(lines['conclusion'])
    raw=_key(data,'segformer_raw');final=_key(data,'segformer_oil')
    if raw and final:
        para(f"采用同一SegFormer训练权重：原始512推理mIoU {_f(raw['summary'].get('mIoU_pct'))}% → 完整推理优化{_f(final['summary'].get('mIoU_pct'))}%。具体贡献需以不同配置的消融结果判断。")
    header('02','实验协议与五项核心对照')
    para(lines['protocol'])
    para(f'数据划分：{exp.get("split","未记录")}；样本数：{exp.get("images",ms[0]["summary"].get("images") if ms else "未记录")}；数据源：{exp.get("dataset","未记录")}。')
    table('performance','表1　五项核心对照：总体像素分割指标与运行耗时')
    fig('01_overall_miou.png','图1　五项主模型总体mIoU；零基线柱状图')
    para('注：主表用于比较总体表现；各项SegFormer消融另表展示。不同架构之间权重与输入不一致，不作为纯架构因果实验。')
    header('03','分类别误检和漏检分析')
    para('每类均联合考察IoU、Precision、Recall和F1。仅由像素指标得到的结论不能解释整机是否漏检。')
    for num,cls in [('3.1','oil'),('3.2','stain'),('3.3','scratch')]:
        parts.append(('h3',f'{num}　{CLASS_CN[cls]}'))
        para(lines[cls]);table(cls,f'表　{CLASS_CN[cls]}分割及错误像素（五项主对照）')
        fig(f'04_{cls}_metrics.png',f'图{ {"oil":3,"stain":5,"scratch":7}[cls]}　{CLASS_CN[cls]}四项指标并列比较；统一0—100%纵轴')
        for j in (1,2):fig(f'case_{cls}_{j:02d}.png',f'图　{CLASS_CN[cls]}定性示例{j}：原图、真实标注及同一图片上的不同方案')
    header('04','SegFormer逐项消融｜哪些方法真正有效')
    para('本部分对同一SegFormer最终训练权重逐项改变推理环节，正向改进和曾导致退步的规则均保留完整结果；不允许仅展示提升的方案。DeepLab基础/完整是不同训练权重，其差值不等于SegFormer训练增益。')
    para('消融图表依次展示仅尺度适配、加入ROI、加入类别置信度、组合规则、加入油污阈值以及曾产生反作用的形态学/面积过滤。增删实验使用相同权重，标明实际变化而非仅按名称推断。')
    para(lines['ablation']);para(_stage_note(data))
    if 'ablation' in tables:table('ablation','表　同权重SegFormer正向、剔除与历史负优化对照')
    fig('10_segformer_ablation.png','图9　SegFormer消融阶段：三类缺陷IoU，同一训练权重；纵轴从0开始')
    fig('07_segformer_delta.png','附图　油污规则对各项指标的增减；正负号表示百分点')
    header('04.1','既有训练日志与阈值搜索（历史研究记录）')
    para('以下图片读取先前保留的实验记录，不是当前四模型统一评测产生的新消融。训练曲线的记录序号不自动等同于训练epoch；阈值曲线来自多次使用的val数据，不能作为独立验证。')
    fig('08_segformer_training_history.png','附图　历史SegFormer验证mIoU记录；回调序号并非严格epoch编号')
    fig('09_oil_threshold_sweep.png','附图　油污阈值搜索的Precision、Recall和IoU；为观察趋势，纵轴从60%开始，非绝对基线')
    header('05','工业检测指标与性能代价')
    para(lines['runtime']);fig('06_latency.png','图10　单图完整推理耗时；柱状图从0开始，依赖硬件和推理配置')
    fig('05_oil_operating_points.png','图11　油污Precision–Recall操作点；不是连续PR曲线，坐标均为0—100%')
    header('06','可靠性边界与结论')
    para(lines['limits']);para(lines['conclusion'])
    if isinstance(ai_text,dict):
        para('分析段落由外部大语言模型辅助生成，指标、图表和表格仍由本地算法计算。AI文字需在论文提交前人工核对。')
    header('附录A','完整配置、背景指标与可复核原始数据')
    table('stage','附表　模型配置、权重关系与控制变量说明')
    table('background','附表　背景类别IoU、Precision、Recall、F1及错误像素')
    for m in ms:
        fig('cm_'+m['id']+'.png',f'附图　{_name(m)}：行是真实类别、列是预测类别，按真实类别归一化')
    for caveat in data.get('caveats',[])+data.get('issues',[]):para('注意：'+caveat)
    para('原始可机读指标：structured_metrics.json、model_comparison.csv、per_class_metrics.csv、各模型confusion_matrix.csv；图表与Word表格均从这些数据生成。')
    return parts,tables


def _build_md(parts,tables,out,title="手机屏幕缺陷分割：实验分析报告"):
    lines=['# '+title,'',
     ('自动生成的工业候选检测报告；需质检员复核。' if '工业批量' in title else '自动生成的可复核研究报告；不是学校官方毕业论文模板。'),'']
    for part in parts:
        kind=part[0]
        if kind=='h2':lines.extend(['','## '+part[1],''])
        elif kind=='h3':lines.extend(['','### '+part[1],''])
        elif kind=='p':lines.extend([part[1],''])
        elif kind=='figure':lines.extend([f'![{part[2]}]({part[1]})',f'*{part[2]}*',''])
        elif kind=='table':
            rows=tables[part[1]]
            lines.extend([f'**{part[2]}**','', '| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |'])
            lines.extend(['| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in rows[1:]])
            lines.append('')
    (out/'论文实验分析.md').write_text('\n'.join(lines),encoding='utf-8')


def _build_html(parts,tables,out,title="手机屏幕表面缺陷分割：实验评测与方法分析"):
    blocks=[]
    for part in parts:
        kind=part[0]
        if kind=='h2':blocks.append('<h2>'+html.escape(part[1])+'</h2>')
        elif kind=='h3':blocks.append('<h3>'+html.escape(part[1])+'</h3>')
        elif kind=='p':blocks.append('<p>'+html.escape(part[1])+'</p>')
        elif kind=='figure':
            # Embed figures so the HTML remains complete when opened through /api/result,
            # copied to another computer or previewed outside its original directory.
            import base64
            figure_file=out/part[1]
            src=('data:image/png;base64,'+base64.b64encode(figure_file.read_bytes()).decode('ascii')) if figure_file.is_file() else ''
            blocks.append(f'<figure><img loading="lazy" src="{src}" alt="{html.escape(part[2])}"><figcaption>{html.escape(part[2])}</figcaption></figure>')
        elif kind=='table':
            rows=tables[part[1]]
            head=''.join('<th>'+html.escape(str(v))+'</th>' for v in rows[0])
            body=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows[1:])
            blocks.append(f'<div class="tablebox"><div class="tablehead"><b>{html.escape(part[2])}</b><button onclick="copyTable(this)">复制表格</button></div><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
    css='''@page{size:A4;margin:20mm 18mm}*{box-sizing:border-box}body{background:#eef1f4;margin:0;color:#1c2735;font-family:"Noto Sans CJK SC","Microsoft YaHei",Arial,sans-serif;line-height:1.78}main{max-width:940px;margin:38px auto;background:white;padding:56px 64px;box-shadow:0 16px 60px #cdd4df66}.eyebrow{font-size:11px;letter-spacing:2px;color:#637184}h1{font-size:29px;line-height:1.35;margin:10px 0 8px;font-weight:700;letter-spacing:-.5px}header{border-bottom:3px solid #263647;padding-bottom:25px;margin-bottom:24px}header p{color:#687487}h2{margin:40px 0 12px;padding-bottom:7px;border-bottom:1px solid #dce2e8;font-size:19px;color:#1e334a;break-after:avoid}h3{font-size:15px;margin:23px 0 7px;color:#31516e}p{margin:9px 0 15px;font-size:12px;color:#3b4758}figure{margin:18px 0 26px;break-inside:avoid}figure img{width:100%;max-height:370px;object-fit:contain;border:1px solid #eef1f5}figcaption{font-size:11px;color:#617085;margin-top:5px;text-align:center}.tablebox{margin:18px 0 26px;break-inside:avoid}.tablehead{display:flex;justify-content:space-between;align-items:center;font-size:12px;margin-bottom:8px}.tablehead button,.actions button{background:white;border:1px solid #cbd5df;border-radius:6px;padding:6px 11px;cursor:pointer;font:inherit;color:#41516a}table{width:100%;border-collapse:collapse;font-size:10.5px}th{text-align:left;color:#f9fafb;background:#25364a;padding:8px 7px;font-weight:500}td{padding:8px 7px;border-bottom:1px solid #e5eaf0}tr:nth-child(even) td{background:#f7f9fb}.actions{display:flex;gap:8px;justify-content:flex-end;margin-top:15px}.stamp{color:#5f7387;font-size:11px}footer{margin-top:35px;border-top:1px solid #dce3eb;padding-top:12px;color:#6a7686;font-size:11px}@media print{body{background:white}main{margin:0;max-width:none;padding:0;box-shadow:none}.tablehead button,.actions{display:none}h2{break-after:avoid}}'''
    js='''function copyTable(btn){let t=btn.closest('.tablebox').querySelector('table');let html=t.outerHTML;let rows=[...t.rows].map(r=>[...r.cells].map(c=>c.innerText).join('\\t')).join('\\n');if(navigator.clipboard&&window.ClipboardItem){navigator.clipboard.write([new ClipboardItem({'text/html':new Blob([html],{type:'text/html'}),'text/plain':new Blob([rows],{type:'text/plain'})})]).then(()=>{btn.innerText='已复制';setTimeout(()=>btn.innerText='复制表格',1200)}).catch(()=>navigator.clipboard.writeText(rows));}else{let sel=window.getSelection(),range=document.createRange();range.selectNode(t);sel.removeAllRanges();sel.addRange(range);document.execCommand('copy');sel.removeAllRanges();btn.innerText='已复制';setTimeout(()=>btn.innerText='复制表格',1200)}}'''
    content='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+css+'</style></head><body><main><header><div class="eyebrow">SCREENQC · RESEARCH EVIDENCE REPORT</div><h1>'+html.escape(title).replace('：','<br>')+'</h1><p>'+(('批次汇总 · 位置数据 · 人工复核 · 可编辑文档' if '工业批量' in title else '本地确定性指标 · 分类别证据 · 研究限制 · 可复制的Word表格'))+'</p><div class="actions"><button onclick="window.print()">打印 / 保存PDF</button></div></header>'+''.join(blocks)+'<footer>ScreenQC Studio · 自动生成｜模型检测输出需经人工审阅｜by Dong</footer></main><script>'+js+'</script></body></html>'
    (out/'论文实验分析.html').write_text(content,encoding='utf-8')


def _build_docx(parts,tables,out,title="手机屏幕表面缺陷分割：实验评测与方法分析"):
    from docx import Document
    from docx.shared import Cm,Pt,RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    doc=Document();sec=doc.sections[0]
    sec.top_margin=Cm(2.1);sec.bottom_margin=Cm(1.9);sec.left_margin=Cm(2.1);sec.right_margin=Cm(2.1)
    normal=doc.styles['Normal'];normal.font.name='Microsoft YaHei';normal.font.size=Pt(10)
    normal.paragraph_format.space_after=Pt(7);normal.paragraph_format.line_spacing=1.35
    for level in (1,2):
        h=doc.styles[f'Heading {level}'];h.font.name='Microsoft YaHei';h.font.color.rgb=RGBColor(29,51,71)
        h.font.size=Pt(15 if level==1 else 12)
    report_is_industrial='工业批量' in title
    doc.add_paragraph('SCREENQC / '+('INDUSTRIAL REPORT' if report_is_industrial else 'RESEARCH REPORT'),style='Subtitle')
    doc.add_heading(title,0)
    doc.add_paragraph('批次统计 · 区域定位 · 人工复核 · 数据追溯' if '工业批量' in title else '本地指标计算 · 分类别分析 · 实验阶段 · 工业权衡 · 可复核证据')
    for part in parts:
        kind=part[0]
        if kind in ('h2','h3'):doc.add_heading(part[1],level=1 if kind=='h2' else 2)
        elif kind=='p':doc.add_paragraph(part[1])
        elif kind=='figure':
            img=out/part[1]
            if img.exists():
                paragraph=doc.add_paragraph();paragraph.alignment=WD_ALIGN_PARAGRAPH.CENTER
                from PIL import Image
                with Image.open(img) as im:ratio=im.height/max(1,im.width)
                paragraph.add_run().add_picture(str(img),width=Cm(15.5),height=Cm(min(9,15.5*ratio)))
                cap=doc.add_paragraph(part[2]);cap.alignment=WD_ALIGN_PARAGRAPH.CENTER
                for run in cap.runs:run.italic=True;run.font.size=Pt(8);run.font.color.rgb=RGBColor(87,104,124)
        elif kind=='table':
            caption_paragraph=doc.add_paragraph(part[2]);caption_paragraph.style='Caption'
            rows=tables[part[1]];t=doc.add_table(rows=1,cols=len(rows[0]));t.style='Table Grid';t.alignment=WD_TABLE_ALIGNMENT.CENTER
            for j,v in enumerate(rows[0]):t.rows[0].cells[j].text=str(v)
            for row in rows[1:]:
                cells=t.add_row().cells
                for j,v in enumerate(row):cells[j].text=str(v)
            hdr=t.rows[0]
            for cell in hdr.cells:
                tcPr=cell._tc.get_or_add_tcPr();shd=OxmlElement('w:shd');shd.set(qn('w:fill'),'24384E');tcPr.append(shd)
                for paragraph in cell.paragraphs:
                    for r in paragraph.runs:r.font.color.rgb=RGBColor(255,255,255);r.font.bold=True;r.font.size=Pt(8)
            for row in t.rows[1:]:
                for cell in row.cells:
                    cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    for paragraph in cell.paragraphs:
                        for r in paragraph.runs:r.font.size=Pt(8)
            doc.add_paragraph('')
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    footer.text=('ScreenQC Studio｜工业候选记录，复核后方可判定' if report_is_industrial else 'ScreenQC Studio｜可编辑研究资料，非学校论文模板')
    path=out/'论文实验分析_可编辑.docx';doc.save(path)


def _build_pdf(parts,tables,out,title="手机屏幕表面缺陷分割：实验评测与方法分析"):
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image as RLImage,PageBreak,KeepTogether
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import mm
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    ink=colors.HexColor('#24384e');muted=colors.HexColor('#5c6d80')
    styles={
      'title':ParagraphStyle('qt',fontName='STSong-Light',fontSize=23,leading=32,textColor=ink,spaceAfter=18),
      'section':ParagraphStyle('qs',fontName='STSong-Light',fontSize=14,leading=21,textColor=ink,spaceBefore=14,spaceAfter=9,keepWithNext=True),
      'sub':ParagraphStyle('qss',fontName='STSong-Light',fontSize=11,leading=17,textColor=ink,spaceBefore=9,spaceAfter=6,keepWithNext=True),
      'body':ParagraphStyle('qb',fontName='STSong-Light',fontSize=9.5,leading=16,textColor=colors.HexColor('#354356'),spaceAfter=9),
      'small':ParagraphStyle('qsm',fontName='STSong-Light',fontSize=7.5,leading=12,textColor=muted,spaceAfter=7),
      'table':ParagraphStyle('qtb',fontName='STSong-Light',fontSize=7,leading=11,textColor=ink),
      'table_header':ParagraphStyle('qth',fontName='STSong-Light',fontSize=7,leading=11,textColor=colors.white),
    }
    path=out/'论文实验分析_黑玉兰.pdf'
    pdf=SimpleDocTemplate(str(path),pagesize=A4,topMargin=19*mm,bottomMargin=19*mm,leftMargin=18*mm,rightMargin=18*mm,title='ScreenQC 实验研究报告')
    width=A4[0]-36*mm
    story=[Spacer(1,13*mm),Paragraph('SCREENQC  /  RESEARCH EVIDENCE',styles['small']),
           Paragraph(html.escape(title).replace('：','<br/>'),styles['title']),
           Paragraph('批次统计 · 缺陷候选 · 坐标与复核' if '工业批量' in title else '各阶段参数 · 三类缺陷 · 误检与漏检 · 论文可复核数据',styles['body']),
           Spacer(1,10*mm),Paragraph(('数据来源：本批次模型推理记录；质量结论须人工复核。' if '工业批量' in title else '数据来源：统一评测CSV与可追溯实验配置，LLM不参与指标计算。'),styles['small']),PageBreak()]
    for part in parts:
        kind=part[0]
        if kind in ('h2','h3'):story.append(Paragraph(html.escape(part[1]),styles['section' if kind=='h2' else 'sub']))
        elif kind=='p':story.append(Paragraph(html.escape(part[1]).replace('\n','<br/>'),styles['body']))
        elif kind=='figure':
            path_img=out/part[1]
            if path_img.is_file():
                from PIL import Image
                with Image.open(path_img) as im:
                    iw,ih=im.size
                w=width;h=w*ih/max(1,iw)
                if h>106*mm:w*=106*mm/h;h=106*mm
                story.append(KeepTogether([RLImage(str(path_img),width=w,height=h),Spacer(1,2*mm),Paragraph(html.escape(part[2]),styles['small']),Spacer(1,3*mm)]))
        elif kind=='table':
            rows=tables[part[1]]
            story.append(Paragraph(html.escape(part[2]),styles['sub']))
            cells=[[Paragraph(html.escape(str(v)),styles['table_header' if i==0 else 'table']) for v in r] for i,r in enumerate(rows)]
            n=len(rows[0]);col=[width/n]*n
            if part[1]=='stage':col=[width*.18,width*.34,width*.25,width*.23]
            tbl=Table(cells,colWidths=col,repeatRows=1,hAlign='LEFT')
            tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),ink),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7fa')]),
                                    ('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#dde4eb')),
                                    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                                    ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
            story.append(tbl);story.append(Spacer(1,3*mm))
    def decoration(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#cbd5df'));canvas.line(18*mm,16*mm,A4[0]-18*mm,16*mm)
        canvas.setFont('STSong-Light',7.8);canvas.setFillColor(muted)
        canvas.drawString(18*mm,11*mm,'ScreenQC Studio | by Dong')
        canvas.drawRightString(A4[0]-18*mm,11*mm,str(doc.page));canvas.restoreState()
    pdf.build(story,onFirstPage=decoration,onLaterPages=decoration)
    return path


def build_package(run:Path,output:Path|None=None,ai_text:dict|str|None=None)->Path:
    """Always generate DOCX+PDF+HTML+MD+JSON, with figures embedded alongside their claims."""
    run=Path(run).expanduser().resolve()
    output=Path(output or run/'paper_package').resolve()
    output.mkdir(parents=True,exist_ok=True)
    data=make_structured(run)
    data['report_schema']='screenqc.report.v5'
    data['evidence']=research_evidence(data)
    data['narrative']=_analysis_sections(data,ai_text)
    (output/'structured_metrics.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'evidence_manifest.json').write_text(json.dumps(data['evidence'],ensure_ascii=False,indent=2),encoding='utf-8')
    _plot_charts(data,output)
    _archive_research_figures(output)
    cases=_collect_case_panels(data,output)
    figures={p.name for p in output.glob('*.png')}
    parts,tables=_parts(data,ai_text,figures)
    _build_md(parts,tables,output)
    _build_html(parts,tables,output)
    _build_docx(parts,tables,output)
    _build_pdf(parts,tables,output)
    for name in ('model_comparison.csv','per_class_metrics.csv','experiment_info.json'):
        if (run/name).is_file():shutil.copy2(run/name,output/name)
    (output/'report_manifest.json').write_text(json.dumps({'version':'4.0','source':str(run),'models':[m['id'] for m in data['models']],
      'tables':list(tables),'figures':sorted(figures),'qualitative_examples':cases,'ai_assisted':bool(ai_text),
      'reports':['论文实验分析_黑玉兰.pdf','论文实验分析_可编辑.docx','论文实验分析.html','论文实验分析.md'],
      'evidence_manifest':'evidence_manifest.json','provenance':data['source_sha256'],'limitations':data['caveats']},ensure_ascii=False,indent=2),encoding='utf-8')
    return output
