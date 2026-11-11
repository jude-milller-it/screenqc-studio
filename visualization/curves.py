"""Render existing epoch logs and previous PR search to local standalone SVG charts."""
from __future__ import annotations
import csv
from html import escape
from pathlib import Path
from core.settings import ROOT


def _svg_lines(series: list[tuple[str,list[float]]], title: str) -> str:
    W,H=720,250
    palette=['#2273d4','#00a98c','#c65fc1','#c9762e','#8958e3','#dc556c']
    if not series:return '<p>没有有效日志</p>'
    allnums=[v for _,arr in series for v in arr]
    lo,hi=min(allnums),max(allnums)
    if abs(hi-lo)<1e-9:hi=lo+1
    paths=[]
    for idx,(name,data) in enumerate(series):
        pts=[]
        for i,v in enumerate(data):
            x=45+float(i)/max(1,len(data)-1)*(W-65)
            y=H-25-(v-lo)/(hi-lo)*(H-55)
            pts.append(f'{x:.1f},{y:.1f}')
        col=palette[idx%len(palette)]
        paths.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2"/><text x="{52+(idx%3)*215}" y="{19+(idx//3)*16}" font-size="12" fill="{col}">{escape(name[:27])}</text>')
    return f'<div class="panel"><h3>{escape(title)}</h3><svg viewBox="0 0 {W} {H}" style="width:100%;max-width:850px"><line x1="45" y1="25" x2="45" y2="225" stroke="#bdc6d2"/><line x1="45" y1="225" x2="700" y2="225" stroke="#bdc6d2"/><text x="4" y="37" font-size="11">{hi:.2f}</text><text x="4" y="225" font-size="11">{lo:.2f}</text>{"".join(paths)}</svg></div>'


def _read_log(path:Path):
    values=[]
    if not path.is_file():return values
    for line in path.read_text(encoding='utf-8-sig',errors='replace').splitlines():
        try:values.append(float(line.strip()))
        except ValueError:continue
    return values


def create_training_report(output:Path,legacy:Path=ROOT/'research') -> Path:
    cards=[]
    for d in ('logs_baseline','logs_loss_opt','logs_cbam','logs_full_all','logs_segformer_mit_b2'):
        logdir=legacy/d
        if not logdir.is_dir():continue
        folders=sorted([p for p in logdir.iterdir() if p.is_dir() and p.name.startswith('loss_')])
        series=[]
        for child in folders:
            data=_read_log(child/'epoch_miou.txt')
            if data:series.append((child.name.removeprefix('loss_'),data))
        if series:
            cards.append('<h2>'+escape(d)+'</h2>'+_svg_lines(series,'Epoch mIoU (as originally recorded)'))
        series=[]
        for child in folders:
            data=_read_log(child/'epoch_loss.txt')
            if data:series.append((child.name.removeprefix('loss_'),data))
        if series:cards.append(_svg_lines(series,'Training Loss'))
    output.parent.mkdir(parents=True,exist_ok=True)
    html=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>历史训练曲线</title><style>body{{background:#f3f6fb;color:#233d5f;font:15px "Microsoft YaHei",sans-serif;padding:35px}}main{{max-width:1000px;margin:auto}}h2{{margin-top:28px}}.panel{{background:white;border-radius:14px;padding:20px;margin:15px 0;box-shadow:0 2px 12px #142b4312}}</style><main><h1>历史训练记录</h1><p>按原始epoch_miou/epoch_loss.txt绘制，不改变训练过程；原训练中缺失的日志不会伪造。</p>{''.join(cards) or '没有找到有效的历史曲线'}<p>不同训练阶段/配置不可只凭曲线最高点断言独立测试集泛化差异。</p></main></html>'''
    output.write_text(html,encoding='utf-8')
    return output


def create_oil_pr_report(csv_path:Path,output:Path) -> Path:
    with csv_path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    all_rows=[r for r in rows if r.get('subset','all')=='all']
    if not all_rows:all_rows=rows
    xlist=[]
    for r in all_rows:
        try:
            precision=float(r['oil_Precision_pct']);recall=float(r['oil_Recall_pct'])
            xlist.append((recall,precision,r))
        except (ValueError,KeyError):pass
    if not xlist:raise ValueError('找不到oil_Precision_pct/oil_Recall_pct列')
    xmin=min(x[0] for x in xlist)-.5;xmax=max(x[0] for x in xlist)+.5
    ymin=min(x[1] for x in xlist)-1;ymax=max(x[1] for x in xlist)+1
    dots=[]
    for r,p,record in xlist:
        x=55+(r-xmin)/max(1e-8,xmax-xmin)*620
        y=245-(p-ymin)/max(1e-8,ymax-ymin)*208
        # historical output uses string 'True' for accepted
        good=str(record.get('eligible_no_iou_drop','')).lower()=='true'
        col='#16a085' if good else '#e58b34'
        title=escape(str(record.get('variant','?'))+f' P={p:.2f}% R={r:.2f}%')
        dots.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{col}"><title>{title}</title></circle>')
    output.parent.mkdir(parents=True,exist_ok=True)
    content=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>油污P-R实验曲线</title><style>body{{background:#f3f6fb;font:16px "Microsoft YaHei",sans-serif;padding:35px;color:#243c57}}main{{max-width:900px;margin:auto;background:white;padding:35px;border-radius:16px}}svg{{width:100%}}</style><main><h1>SegFormer油污 Precision-Recall 候选</h1><p>每个点来自已有候选配置，绿色表示CSV标记符合全量无退步规则。尚未独立验证，不等同于正式PR曲线/AUPRC。</p><svg viewBox="0 0 720 288"><line x1="55" y1="35" x2="55" y2="245" stroke="#90a6c1"/><line x1="55" y1="245" x2="675" y2="245" stroke="#90a6c1"/>{''.join(dots)}<text x="270" y="274">Oil Recall (%)</text><text x="3" y="20">Precision (%)</text><text x="55" y="260">{xmin:.1f}</text><text x="650" y="260">{xmax:.1f}</text><text x="5" y="45">{ymax:.1f}</text><text x="5" y="245">{ymin:.1f}</text></svg><p>参数搜索在同一批验证图片上反复进行，因此所有收益都属于探索性结果；选择参数必须在新的独立样本上复核。</p></main></html>'''
    output.write_text(content,encoding='utf-8')
    return output
