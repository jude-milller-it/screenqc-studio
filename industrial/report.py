# -*- coding: utf-8 -*-
"""Industry inspection batch summary: no fabricated pass/fail, no GT-derived metrics."""
from __future__ import annotations
import csv,json,html
from pathlib import Path
from collections import Counter
from datetime import datetime
from paper.evidence import inspection_evidence
from paper.report_v3 import _build_pdf,_build_docx,_build_html,_build_md


def _csv(file):
    if not file.is_file():return []
    with file.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def build_inspection(run:Path,ai_text:dict|None=None)->Path:
    """Create a deterministic, editable industrial handoff report for existing predictions."""
    run=Path(run).resolve();images=_csv(run/'images.csv');regions=_csv(run/'regions.csv')
    if not images:raise FileNotFoundError(f'未找到工业批量检测结果 images.csv：{run}')
    output=run/('inspection_report_ai' if ai_text else 'inspection_report');output.mkdir(parents=True,exist_ok=True)
    evidence=inspection_evidence(images,regions)
    counts=Counter(str(row.get('class_name') or row.get('class') or '') for row in regions)
    all_times=[]
    for row in images:
        try:all_times.append(float(row.get('time_ms') or row.get('inference_ms') or 0))
        except (ValueError,TypeError):pass
    pixel_totals={k:sum(int(float(r.get(k+'_pixels') or 0)) for r in images) for k in ('oil','stain','scratch')}
    # Document useful distributional evidence for AI without shipping any raw images.
    import statistics
    regions_per_image=[int(float(r.get('regions') or 0)) for r in images]
    model_names=sorted({str(r.get('model') or '') for r in images if str(r.get('model') or '').strip()})
    edge_candidates=0
    dims={str(r.get('source','')):(int(float(r.get('width') or 0)),int(float(r.get('height') or 0))) for r in images}
    for r in regions:
        w,h=dims.get(str(r.get('image','')),(0,0))
        if w and h:
            try:
                x,y,bw,bh=[int(float(r.get(k) or 0)) for k in ('x','y','width','height')]
                if x<10 or y<10 or x+bw>w-10 or y+bh>h-10:edge_candidates+=1
            except ValueError:pass
    distribution={'images_with_candidate':sum(x>0 for x in regions_per_image),
                  'images_without_candidate':sum(x==0 for x in regions_per_image),
                  'regions_per_image_median':statistics.median(regions_per_image) if regions_per_image else 0,
                  'regions_per_image_max':max(regions_per_image,default=0),
                  'candidates_touching_outer_10px':edge_candidates}
    meta={'model_names':model_names,'distribution':distribution,
       'predicted_pixels':pixel_totals,'schema':'screenqc.inspection.v3','generated_at':datetime.now().astimezone().isoformat(timespec='seconds'),
       'source':str(run),'images':len(images),'regions':len(regions),'region_counts':{k:int(counts.get(k,0)) for k in ('oil','stain','scratch')},
       'mean_inference_ms':sum(all_times)/len(all_times) if all_times else None,
       'note':'自动预测结果仅为缺陷候选，不构成生产批次放行决定；无GT标注，不能计算真实Precision、Recall或mIoU。'}
    meta['evidence']=evidence
    (output/'inspection_data.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'evidence_manifest.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    from paper.report_v3 import _f
    tables={'summary':[['项目','数值'],['检测图片',str(len(images))],['检出区域',str(len(regions))],
            ['油污连通区域',str(counts.get('oil',0))],['污渍连通区域',str(counts.get('stain',0))],['划痕连通区域',str(counts.get('scratch',0))],
            ['油污预测像素',str(pixel_totals['oil'])],['污渍预测像素',str(pixel_totals['stain'])],['划痕预测像素',str(pixel_totals['scratch'])],
            ['平均处理耗时(ms/图)',_f(meta['mean_inference_ms'])],['P95处理耗时(ms/图)',_f(evidence['p95_processing_ms'])],['存在候选图片比例(%)',_f(evidence['candidate_image_rate_pct'])],['至少一处候选的图片',str(distribution['images_with_candidate'])],['未发现候选的图片',str(distribution['images_without_candidate'])]],
            'image_list':[['图片','检测模型','候选区域数','处理时间 (ms)']]}
    for item in sorted(images, key=lambda r:float(r.get('regions') or 0), reverse=True)[:8]:
        tables['image_list'].append([Path(item.get('source') or item.get('image') or '?').name,
            item.get('model') or '—',str(item.get('regions','—')),str(item.get('time_ms','—'))])
    tables['regions']=[['图片','缺陷类别','X','Y','宽','高','面积(px)']]
    for region in sorted(regions,key=lambda r:float(r.get('area_px') or 0),reverse=True)[:8]:
        tables['regions'].append([Path(region.get('image','')).name, region.get('class','—'),
            str(region.get('x','—')),str(region.get('y','—')),str(region.get('width','—')),str(region.get('height','—')),str(region.get('area_px','—'))])
    # Decision-ready report: do not confuse predicted candidates with confirmed failures.
    notes={'overview':(
       f"本次离线图像批次共处理{len(images)}张，其中{evidence['images_with_candidates']}张含至少一个模型候选区域"
       f"（{evidence['candidate_image_rate_pct']:.2f}%）；候选连通区域共{len(regions)}个。"
       f"平均耗时{_f(meta['mean_inference_ms'])} ms/张，P95为{_f(evidence['p95_processing_ms'])} ms/张。"
       '上述结果是算法筛查输出，不是人工确认的缺陷数量、产品不良率或放行依据。'),
       'locations':('按候选区域数量优先安排人工复核；下表仅列候选较集中的图片。'
         '优先级是透明的排序规则，不是缺陷严重度判断。'
         '原图坐标及多边形等全部保存在regions.csv和逐图JSON，PDF正文只保留代表性样本。'),
       'safety':('质检复核时应结合实际光照、手机结构和标准确认候选；尤其关注屏幕边缘反光、孔位、非显示区域误检。'
           '系统没有人工复核真值，不会自动计算良品率、误放行率，也不能自动签署放行结论。')}
    if isinstance(ai_text,dict):
        for key in notes:
            v=ai_text.get(key)
            if isinstance(v,str) and v.strip():notes[key]=v.strip()
    parts=[('h2','01. 批次速览｜需要关注什么'),('p',notes['overview']),
           ('table','summary','表1　批次主要事实（预测区域与预测像素分开统计）'),
           ('h2','02. 缺陷候选分布｜各类别如何分布')]
    # The chart is inserted below next to its supporting section.
    # Defect-type composition is not a quality metric; do not label as a pass rate.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from visualization.fonts import configure_chinese_matplotlib
    configure_chinese_matplotlib()
    fig,ax=plt.subplots(figsize=(8,3.5))
    names=['油污','污渍','划痕'];values=[counts.get(c,0) for c in ('oil','stain','scratch')]
    bars=ax.bar(names,values,color=['#ce9963','#5e90bd','#cb7382'],width=.53)
    ax.set_ylim(0,max(values+[1])*1.22);ax.set_ylabel('候选连通区域数（不是实际缺陷实例数）')
    ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
    for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+max(values+[1])*.03,str(v),ha='center')
    for edge in ('top','right'):ax.spines[edge].set_visible(False)
    fig.tight_layout();fig.savefig(output/'figure_defect_candidates.png',dpi=170);plt.close(fig)
    parts.append(('figure','figure_defect_candidates.png','图1　按油污、污渍、划痕统计的候选连通区域（非人工真值）'))
    # Separate predicted pixel composition from region counts; these are different units.
    px=[pixel_totals[c] for c in ('oil','stain','scratch')]
    fig2,ax2=plt.subplots(figsize=(7.8,3.2))
    total_px=sum(px) or 1
    bars2=ax2.bar(names,[100*v/total_px for v in px],color=['#ce9963','#5e90bd','#cb7382'],width=.52)
    ax2.set_ylim(0,100);ax2.set_ylabel('预测缺陷像素占比 (%)')
    ax2.grid(axis='y',alpha=.14);ax2.set_axisbelow(True)
    for bar,v in zip(bars2,px):ax2.text(bar.get_x()+bar.get_width()/2,100*v/total_px+1,f'{100*v/total_px:.1f}%',ha='center')
    for edge in ('top','right'):ax2.spines[edge].set_visible(False)
    fig2.tight_layout();fig2.savefig(output/'figure_predicted_pixel_share.png',dpi=170);plt.close(fig2)
    parts.append(('figure','figure_predicted_pixel_share.png','图2　按类别统计的预测像素占比；不得视为真实缺陷发生率'))
    parts.extend([('h2','03. 重点复核｜先看哪些图片'),('p',notes['locations']),('table','image_list','表2　按候选数量排序的8张优先复核图片'),('table','regions','表3　候选面积最大的8处区域（原图像素坐标）')])
    # Deliberately choose visually diverse class examples, one per defect class.
    used=set()
    examples=[]
    import shutil
    for label in ('oil','stain','scratch'):
        candidates=sorted(images,key=lambda r:float(r.get(label+'_pixels') or 0),reverse=True)
        for item in candidates:
            src=item.get('source')
            if src in used or float(item.get(label+'_pixels') or 0)<=0:continue
            box=Path(item.get('boxes') or '')
            if not box.is_file():
                # Historical CSV may still contain the previous Studio's absolute path.
                # Images copied into this run are the authoritative local artifacts.
                box=run/'boxes'/box.name
            if box.is_file():
                dest=output/f'example_{label}.png';shutil.copy2(box,dest)
                examples.append(dest.name);used.add(src)
                parts.append(('figure',dest.name,'缺陷区域示例（'+label+'）：'+box.name))
                break
    parts += [('h2','04. 复核与数据边界'),('p',notes['safety']),
              ('p','记录交付：images.csv（逐图候选及耗时）、regions.csv（逐区域原图坐标）、逐图JSON与可视化图片。已确认缺陷、责任人、处置状态需要人工录入；未录入时统一保留待复核。')] 
    report_title='手机屏幕工业批量检测｜候选与复核报告'
    _build_md(parts,tables,output,title=report_title);_build_html(parts,tables,output,title=report_title)
    _build_docx(parts,tables,output,title=report_title);_build_pdf(parts,tables,output,title=report_title)
    for original,renamed in [('论文实验分析.md','工业批量检测报告.md'),('论文实验分析.html','工业批量检测报告.html'),
                            ('论文实验分析_可编辑.docx','工业批量检测报告.docx'),('论文实验分析_黑玉兰.pdf','工业批量检测报告.pdf')]:
        (output/original).replace(output/renamed)
    return output
