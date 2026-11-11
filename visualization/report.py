"""Self-contained HTML comparison: no network/CDN dependencies."""
from __future__ import annotations
import csv
import html
from pathlib import Path
from core.metrics import CLASSES, CHINESE


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():return []
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def _float(value) -> float:
    try:return float(value)
    except (TypeError,ValueError):return 0.0


def _bar_table(rows: list[dict], label: str, key: str) -> str:
    segments=[]
    for r in rows:
        v=_float(r.get(key))
        width=min(100,max(0,v))
        segments.append(f'<div class="bar-item"><span>{html.escape(str(r.get("model","?")))}</span>'
                        f'<div class="bar"><i style="width:{width:.2f}%"></i></div><b>{v:.2f}%</b></div>')
    return f'<article><h3>{html.escape(label)}</h3>{"".join(segments)}</article>'


def make_report(results_dir: Path) -> Path:
    rows=_read_csv(results_dir/'model_comparison.csv')
    classes=_read_csv(results_dir/'per_class_metrics.csv')
    if not rows:
        raise FileNotFoundError(f'找不到有内容的 model_comparison.csv: {results_dir}')
    titles=['模型','mIoU','缺陷mIoU','平均P','平均R','平均F1','推理ms','FPS']
    fields=['model','mIoU_pct','defect_mIoU_pct','mPrecision_pct','mRecall_pct','mF1_pct','mean_pipeline_ms','fps']
    def cell(row,k):
        v=row.get(k,'')
        return html.escape(str(v)) if k=='model' else f'{_float(v):.2f}'
    body=''.join('<tr>'+''.join(f'<td>{cell(r,k)}</td>' for k in fields)+'</tr>' for r in rows)
    bars=''.join(_bar_table(rows, label, key) for label,key in [('总体 mIoU','mIoU_pct'),('缺陷平均IoU','defect_mIoU_pct'),('平均召回率','mRecall_pct')])
    per_class=''
    for cls,zh in zip(CLASSES,CHINESE):
        subset=[r for r in classes if r.get('class')==cls]
        if not subset:continue
        table=''.join('<tr><td>'+html.escape(str(r['model']))+'</td>'+''.join(f'<td>{_float(r.get(k)):.2f}%</td>' for k in ('IoU_pct','Precision_pct','Recall_pct','F1_Dice_pct'))+'</tr>' for r in subset)
        per_class+=f'<article><h3>{zh} / {cls}</h3><table><thead><tr><th>模型</th><th>IoU</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{table}</tbody></table></article>'
    css="""body{font-family:'Segoe UI','Microsoft YaHei',sans-serif;background:#f3f6fc;color:#17253d;margin:0;padding:36px}main{max-width:1160px;margin:auto}h1{font-size:30px}h2{margin:32px 0 16px}p{color:#5c6a81}article,.hero{background:white;border:1px solid #e4eaf4;border-radius:16px;padding:22px;margin:12px 0;box-shadow:0 3px 12px #1225440b}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:13px 10px;border-bottom:1px solid #e9eef6;text-align:left}th{color:#52647d;background:#f7f9fc;white-space:nowrap}.bar-item{display:flex;align-items:center;gap:10px;margin:14px 0;font-size:12px}.bar-item span{width:150px;word-wrap:break-word}.bar{flex:1;background:#e7eef9;border-radius:9px;height:12px;overflow:hidden}.bar i{display:block;background:linear-gradient(90deg,#15998d,#457eea);height:100%}.bar-item b{width:60px;text-align:right}.warning{border-left:4px solid #db9750;padding-left:15px}.scroll{overflow:auto}footer{color:#6a778c;font-size:12px;margin:32px 0}"""
    doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>三模型统一评测报告</title><style>{css}</style></head><body><main>
    <div class="hero"><h1>手机屏幕缺陷检测 · 三模型对比报告</h1><p>来自同一评测目录的实际 CSV。类别：背景／油污／污渍／划痕。指标单位：百分比。</p></div>
    <h2>总体比较</h2><article class="scroll"><table><thead><tr>{''.join(f'<th>{x}</th>' for x in titles)}</tr></thead><tbody>{body}</tbody></table></article>
    <div class="grid">{bars}</div><h2>逐类别比较</h2><div class="grid">{per_class}</div>
    <article class="warning"><h3>研究与部署边界</h3><p>此报告只展示测试数据的像素级指标；不代表整机漏检率或量产质量判定。FDSNet原作者的公开权重可能与本地验证数据存在训练集重叠。SegFormer油污阈值+0.30是当前验证集上探索出的候选，尚未独立验证。推理速度受设备、输入尺度、CUDA预热影响。</p></article>
    <footer>报告为本地静态HTML，无需联网。原始数据：model_comparison.csv / per_class_metrics.csv。</footer></main></body></html>'''
    output=results_dir/'report.html'
    output.write_text(doc,encoding='utf-8')
    return output
