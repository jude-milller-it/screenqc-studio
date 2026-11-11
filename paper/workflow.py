# -*- coding: utf-8 -*-
"""Deterministic paper artifacts from an existing evaluation run.
Numbers ONLY come from CSVs; no model-generated numbers.
"""
from __future__ import annotations
import csv, json, math, hashlib, html
from pathlib import Path
from datetime import datetime

CLASSES = [('background','背景'),('oil','油污'),('stain','污渍'),('scratch','划痕')]
MODELS = {'deeplab_base':'DeepLabV3+（基础训练）',
          'segformer_raw':'SegFormer（最终权重，原始512推理）',
          'segformer_scale':'SegFormer（仅尺度适配）',
          'segformer_roi':'SegFormer（尺度+ROI）',
          'segformer_conf':'SegFormer（尺度+置信度）',
          'segformer_post_bad':'SegFormer（历史负优化：面积/形态学）',
          'fdsnet_tuned':'FDSNet（验证集污渍概率校正候选）',
          'deeplab':'DeepLabV3+（改进CNN基线）','segformer':'SegFormer（基础推理配置）',
          'segformer_oil':'SegFormer（油污阈值优化）','fdsnet':'FDSNet（作者公开权重）',
          'DeepLabV3_scheme3':'DeepLabV3+（方案3）',
          'SegFormer_640x1152_C':'SegFormer（640×1152基础配置）',
          'FDSNet_original':'FDSNet（作者公开权重）'}
METRICS = [('IoU_pct','IoU (%)'),('Precision_pct','Precision (%)'),('Recall_pct','Recall (%)'),('F1_Dice_pct','F1/Dice (%)')]


def _csv(path):
    if not path.is_file():return []
    with path.open('r',newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def _number(v):
    try:
        n = float(v)
        return n if math.isfinite(n) else None
    except (TypeError,ValueError):return None


def _safe(v):
    if v is None:return '—'
    return f'{v:.2f}' if isinstance(v,(int,float)) else str(v)


def _filehash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def make_structured(run:Path) -> dict:
    run=Path(run)
    overall=_csv(run/'model_comparison.csv')
    classes=_csv(run/'per_class_metrics.csv')
    if not overall or not classes:
        raise FileNotFoundError('需要有效的 model_comparison.csv 和 per_class_metrics.csv')
    meta={}
    info=run/'experiment_info.json'
    if info.is_file():
        try:meta=json.loads(info.read_text(encoding='utf-8-sig'))
        except (ValueError,UnicodeError):meta={'metadata_error':'实验元数据无效'}
    overall_keys=['mIoU_pct','defect_mIoU_pct','mPrecision_pct','mRecall_pct','mF1_pct','pixel_accuracy_pct',
                  'mean_pipeline_ms','p95_pipeline_ms','fps','images']
    models=[]
    seen=set()
    for entry in overall:
        key=entry.get('model','').strip()
        if not key or key in seen:raise ValueError('结果中模型名缺失或重复')
        seen.add(key)
        per={}
        for code,_ in CLASSES:
            found=[r for r in classes if r.get('model')==key and r.get('class')==code]
            if len(found)!=1:raise ValueError(f'{key}/{code} 找不到唯一的类别指标')
            per[code]={k:_number(found[0].get(k)) for k,_ in METRICS}
            for k in ('TP','FP','FN','GT_pixels','pred_pixels'):
                per[code][k]=_number(found[0].get(k))
        models.append({'id':key,'name':MODELS.get(key,key),
                       'summary':{k:_number(entry.get(k)) for k in overall_keys},'classes':per})
    caveats=['结果由现有评测CSV确定性生成，LLM不参与数值计算。',
             '本项目当前val数据已被用于多次调参，不能把它称为独立测试。',
             'FDSNet作者权重可能存在验证集训练重叠，不能当作独立公平泛化排名。',
             '该评测包含三种网络架构和多组训练/推理配置；所有SegFormer推理消融共用最终权重，不能将推理增益归因于Transformer结构。',
             '召回率是像素级召回率，不代表整条划痕、整机产品的检出率。',
             '不同模型的输入尺寸、训练方式和权重不同，架构因果关系无法直接识别。']
    issues=[]
    if not meta:issues.append('缺少experiment_info.json，无法核实数据划分及推理条件。')
    if meta.get('split')=='val':issues.append('数据划分为val，结果用于验证而非独立测试。')
    if meta.get('max_images',0):issues.append('本次只评测部分图像。')
    doc={'schema':'screenqc.paper.v2','created_at':datetime.now().astimezone().isoformat(timespec='seconds'),
         'source_dir':str(run.resolve()),'source_sha256':{f:_filehash(run/f) for f in ('model_comparison.csv','per_class_metrics.csv')},
         'experiment':meta,'models':models,'caveats':caveats,'issues':issues,
         'paper_theme':'黑玉兰 · 学术白底黑金（非学校官方模版）'}
    return doc


def comparison_tables(data):
    models=data['models']
    top=[['指标']+[m['name'] for m in models]]
    for key,title in [('mIoU_pct','mIoU (%)'),('defect_mIoU_pct','缺陷mIoU (%)'),
                      ('mPrecision_pct','平均精确率 (%)'),('mRecall_pct','平均召回率 (%)'),
                      ('mF1_pct','平均F1 (%)'),('mean_pipeline_ms','平均耗时 (ms)'),('fps','FPS')]:
        top.append([title]+[_safe(m['summary'][key]) for m in models])
    groups={'总体对比':top}
    for code,zh in CLASSES:
        rows=[['指标']+[m['name'] for m in models]]
        for key,title in METRICS:
            rows.append([title]+[_safe(m['classes'][code][key]) for m in models])
        groups[zh+'分割对比']=rows
    return groups


def _chart_png(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from visualization.fonts import configure_chinese_matplotlib
    configure_chinese_matplotlib()
    names=[m['id'] for m in data['models']]
    values=[m['summary']['mIoU_pct'] or 0 for m in data['models']]
    fig,ax=plt.subplots(figsize=(8,3.6))
    ax.bar(names,values,color=['#313941','#d8a748','#8a6552','#579d91'][:len(names)])
    ax.set_ylim(0,100);ax.set_ylabel('mIoU (%)');ax.set_title('Overall mIoU (same image list)')
    for idx,v in enumerate(values):ax.text(idx,v+1,f'{v:.2f}',ha='center',fontsize=9)
    fig.tight_layout();fig.savefig(out/'figure_miou.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.6))
    cls=['oil','stain','scratch'];labs=['Oil','Stain','Scratch']
    import numpy as np
    x=np.arange(3);w=.75/max(1,len(names))
    for i,m in enumerate(data['models']):
        vals=[m['classes'][c]['Recall_pct'] or 0 for c in cls]
        ax.bar(x+(i-(len(names)-1)/2)*w,vals,w,label=m['id'])
    ax.set_xticks(x,labs);ax.set_ylabel('Pixel Recall (%)');ax.set_ylim(0,105)
    ax.legend(fontsize=8,ncol=2);ax.set_title('Defect recall by class')
    fig.tight_layout();fig.savefig(out/'figure_recall.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.6))
    for i,m in enumerate(data['models']):
        vals=[m['classes'][c]['IoU_pct'] or 0 for c in cls]
        ax.bar(x+(i-(len(names)-1)/2)*w,vals,w,label=m['id'])
    ax.set_xticks(x,labs);ax.set_ylabel('IoU (%)');ax.set_ylim(0,105)
    ax.legend(fontsize=8,ncol=2);ax.set_title('Per-class IoU')
    fig.tight_layout();fig.savefig(out/'figure_class_iou.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.6))
    for m in data['models']:
        pr=m['classes']['oil']['Precision_pct'];re=m['classes']['oil']['Recall_pct']
        if pr is not None and re is not None:ax.scatter(re,pr,s=90,label=m['id'])
    ax.set_xlabel('Oil recall (%)');ax.set_ylabel('Oil precision (%)');ax.set_xlim(0,105);ax.set_ylim(0,105)
    ax.legend(fontsize=8,ncol=2);ax.set_title('Oil: single operating points (NOT PR curves)')
    fig.tight_layout();fig.savefig(out/'figure_oil_operating_points.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.6))
    values=[m['summary']['mean_pipeline_ms'] or 0 for m in data['models']]
    ax.bar(names,values,color='#8a6552');ax.set_ylabel('Latency (ms / image)')
    for idx,v in enumerate(values):ax.text(idx,v+max(values+[1])*.015,f'{v:.1f}',ha='center',fontsize=9)
    ax.set_title('Observed pipeline latency (hardware-dependent)')
    fig.tight_layout();fig.savefig(out/'figure_latency.png',dpi=180);plt.close(fig)


def _table_docx(data,out):
    from docx import Document
    from docx.shared import Pt, Inches
    doc=Document();sec=doc.sections[0];sec.top_margin=Inches(.7);sec.bottom_margin=Inches(.7)
    doc.add_heading('手机屏幕缺陷检测｜论文数据表格',0)
    doc.add_paragraph('可直接在Word中复制、编辑表格。所有数字从评测CSV读取；未采用LLM修改。')
    for title,rows in comparison_tables(data).items():
        doc.add_heading(title,level=2)
        t=doc.add_table(rows=1,cols=len(rows[0]));t.style='Table Grid'
        for j,v in enumerate(rows[0]):t.rows[0].cells[j].text=v
        for row in rows[1:]:
            cells=t.add_row().cells
            for j,v in enumerate(row):cells[j].text=v
    doc.add_heading('研究局限',level=2)
    for line in data['caveats']:doc.add_paragraph(line,style='List Bullet')
    path=out/'论文可编辑表格.docx';doc.save(path)
    return path


def _paper_pdf(data,out,ai_text=None):
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image as RLImage,PageBreak,KeepTogether
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.lib.units import mm
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    gold=colors.HexColor('#a78d60');ink=colors.HexColor('#24252b');light=colors.HexColor('#f5f3ef')
    suffix='AI辅助分析' if ai_text else '标准'
    dest=out/f'黑玉兰_论文评测报告_{suffix}.pdf'
    styles=getSampleStyleSheet()
    normal=ParagraphStyle('cn_normal',parent=styles['Normal'],fontName='STSong-Light',fontSize=9.3,leading=16,textColor=ink,spaceAfter=9)
    title=ParagraphStyle('cn_title',parent=normal,fontSize=21,leading=30,alignment=TA_CENTER,spaceAfter=25)
    section=ParagraphStyle('cn_sec',parent=normal,fontSize=13,leading=19,spaceBefore=16,spaceAfter=10,textColor=colors.HexColor('#6b5334'))
    small=ParagraphStyle('cn_small',parent=normal,fontSize=7.5,leading=12)
    doc=SimpleDocTemplate(str(dest),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=20*mm,bottomMargin=19*mm)
    story=[Spacer(1,17*mm),Paragraph('手机屏幕缺陷检测',title),Paragraph('四种推理方案 · 语义分割评测报告',ParagraphStyle('cn_sub',parent=normal,alignment=TA_CENTER,fontSize=13,leading=22)),
           Spacer(1,12*mm),Paragraph('SCREENQC / 黑玉兰主题',ParagraphStyle('cn_tag',parent=normal,alignment=TA_CENTER,textColor=gold)),
           Spacer(1,16*mm),Paragraph('本报告由可复核的评测数据生成。所有表格为文本，不是图片。',normal),
           Paragraph('数据划分：'+html.escape(str(data.get('experiment',{}).get('split','未记录')))+'　｜模型数量：'+str(len(data['models'])),normal),PageBreak()]
    tables=comparison_tables(data)
    for heading,rows in tables.items():
        story.append(Paragraph(heading,section))
        max_col=max(1,len(rows[0]));col_w=(A4[0]-36*mm)/max_col
        white_header=ParagraphStyle('white_head_'+str(len(story)),parent=small,textColor=colors.white)
        wrapped=[[Paragraph(html.escape(str(v)),white_header if i==0 else small) for v in row] for i,row in enumerate(rows)]
        t=Table(wrapped,colWidths=[col_w]*max_col,repeatRows=1,hAlign='CENTER')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),ink),('TEXTCOLOR',(0,0),(-1,0),colors.white),
                    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,light]),
                    ('GRID',(0,0),(-1,-1),.3,colors.HexColor('#dfd9cf')),
                    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                    ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        story.append(t);story.append(Spacer(1,6*mm))
    story.append(Paragraph('论文插图',section))
    for f in ('figure_miou.png','figure_recall.png','figure_class_iou.png','figure_oil_operating_points.png','figure_latency.png'):
        path=out/f
        if path.is_file():story.append(RLImage(str(path),width=168*mm,height=75.6*mm));story.append(Spacer(1,5*mm))
    story.append(Paragraph('实验边界与可信度声明',section))
    for line in data['caveats']+data['issues']:
        story.append(Paragraph('• '+html.escape(line),normal))
    if ai_text:
        story.append(PageBreak());story.append(Paragraph('大模型辅助讨论（非原始实验数据）',section))
        for paragraph in ai_text.split('\n'):
            if paragraph.strip():story.append(Paragraph(html.escape(paragraph.strip()),normal))
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(gold)
        canvas.line(18*mm,17*mm,A4[0]-18*mm,17*mm)
        canvas.setFont('STSong-Light',8);canvas.drawString(18*mm,12*mm,'ScreenQC · 黑玉兰学术报告 / 非官方模板')
        canvas.drawRightString(A4[0]-18*mm,12*mm,str(doc.page));canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return dest


def _html_report(data,out,ai_text=None):
    table_html=[]
    for heading,rows in comparison_tables(data).items():
        header=''.join('<th>'+html.escape(v)+'</th>' for v in rows[0])
        body=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows[1:])
        table_html.append(f'<section><h2>{html.escape(heading)}</h2><div class="scroll"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div></section>')
    caveats=''.join('<li>'+html.escape(t)+'</li>' for t in data['caveats']+data['issues'])
    images=''.join(f'<figure><img src="{f}" alt="{f}"><figcaption>{f}</figcaption></figure>' for f in ('figure_miou.png','figure_recall.png','figure_class_iou.png','figure_oil_operating_points.png','figure_latency.png'))
    ai=f'<section><h2>AI辅助讨论</h2><div class="ai">{html.escape(ai_text)}</div></section>' if ai_text else ''
    css='''body{background:#f6f5f2;font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;color:#24252b;margin:0}main{max-width:1100px;margin:auto;padding:38px 18px 60px}header{background:#24252b;color:white;padding:38px;border-radius:9px;border-bottom:5px solid #a78d60}header p{color:#dbd4c7}section{background:white;border:1px solid #e7e2da;border-radius:9px;margin:18px 0;padding:20px}h1{margin:0;font-size:29px}h2{color:#84683d;font-size:19px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:11px 9px;text-align:left;border-bottom:1px solid #eee7dc;white-space:normal}th{background:#f3eee7}img{max-width:100%;display:block;margin:auto}figure{margin:14px auto;max-width:800px}.ai{white-space:pre-wrap;line-height:1.8}li{line-height:1.7}button{border:1px solid #d1bf9d;border-radius:6px;padding:7px 13px;cursor:pointer;background:white}@media print{button{display:none}main{padding:0}section{break-inside:avoid}}'''
    src='''<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>ScreenQC 论文报告</title><style>'''+css+'''</style></head><body><main><header><h1>手机屏幕缺陷检测 · 论文评测工作台</h1><p>全部数值来自本地实际评测CSV，支持复制Word</p><button onclick="window.print()">打印或另存为PDF</button></header>'''+''.join(table_html)+f'<section><h2>核心论文插图</h2>{images}</section>'+ai+'<section><h2>实验限制与风险</h2><ul>'+caveats+'</ul></section></main></body></html>'
    path=out/'paper_report.html';path.write_text(src,encoding='utf-8');return path


def _review_checklist(data,out):
    meta=data['experiment']
    lines=['# 毕业论文评测 / 评审检查清单','','## 1. 实验环境与数据证据',
           '- [ ] 确认数据集路径、标签类别0背景/1油污/2污渍/3划痕',
           '- [ ] 存档 train.txt、val.txt、test.txt 的SHA256，核对是否有重叠',
           '- [ ] 确认三模型训练权重来源及是否与验证样本交叉',
           '- [ ] 记录设备型号、PyTorch/CUDA版本和单图输入尺度',
           '','## 2. 比较协议',
           '- [ ] 基线SegFormer和油污优化版严格使用同一权重、同一推理尺寸和共同ROI/基础阈值',
           '- [ ] 唯一差异为油污类别阈值增量+0.30（参数在验证集选定）',
           '- [ ] 标注FDSNet来自作者权重；不可当作受控架构消融',
           '- [ ] 分别报告 mIoU、各类IoU、Precision、Recall、F1、延迟、混淆矩阵',
           '','## 3. 消融与可复现',
           '- [ ] 将旧SegFormer分辨率实验（512→640×1152）与现有结果逐项核对',
           '- [ ] 提供ROI、类别阈值、后处理的逐项消融，禁止混用实验口径',
           '- [ ] 若要宣称独立泛化，训练新权重并在未参与调参的测试集上复验',
           '','## 4. 论文制作',
           '- [ ] 从论文可编辑表格.docx复制原生表格至毕业论文',
           '- [ ] 使用figure*.png插图并补图题/图注/误差与局限性',
           '- [ ] 打开黑玉兰PDF预览；核对文字与数字不溢出',
           '- [ ] LLM辅助分析仅作为初稿，逐数字核对，不能生成虚假参考文献',
           '','## 本次报告证据',
           'split: '+str(meta.get('split','未提供')),
           'models: '+', '.join(m['id'] for m in data['models']),
           'source: '+data['source_dir'],
           '','## 重要局限']+['- '+x for x in data['caveats']]
    (out/'论文答辩评审流程_检查清单.md').write_text('\n'.join(lines),encoding='utf-8')


def build_package(run:Path,output:Path|None=None,ai_text:str|None=None)->Path:
    run=Path(run);output=Path(output or (run/'paper_package'));output.mkdir(parents=True,exist_ok=True)
    data=make_structured(run)
    (output/'structured_metrics.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    _chart_png(data,output)
    _table_docx(data,output)
    _html_report(data,output,ai_text)
    _paper_pdf(data,output,ai_text)
    _review_checklist(data,output)
    (output/'README_报告说明.txt').write_text('本目录中的数值从评测CSV计算和转录，LLM不得修改数值。\n'+
       'structured_metrics.json：可信结构化指标；论文可编辑表格.docx：Word原生表格；figure*.png：论文插图。\n'+
       'paper_report.html：浏览器可复制表格；黑玉兰_论文评测报告_*.pdf：本地PDF预览。\n'+
       '这是非官方学术报告风格，不是学校论文排版规范。\n',encoding='utf-8')
    return output
