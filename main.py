# -*- coding: utf-8 -*-
"""DeepLabV3 Studio: independent migration, prediction, evaluation, GUI, report."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from core.settings import ROOT, get_config, model_weights, path_value, save_config
from model_adapters.registry import AVAILABLE


def safe_destination(out: str | None, category: str) -> Path:
    if out: return Path(out).expanduser().resolve()
    return path_value(get_config()['outputs'])/category/time.strftime('%Y%m%d_%H%M%S')


def files_from_args(files: list[str],folder: str | None, recursive: bool) -> list[Path]:
    from core.dataset import IMAGE_EXTS
    paths=[Path(x).expanduser().resolve() for x in files]
    if folder:
        d=Path(folder).expanduser().resolve()
        if not d.is_dir():raise NotADirectoryError(d)
        paths += [p for p in (d.rglob('*') if recursive else d.iterdir())
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    seen=set();out=[]
    for p in paths:
        if not p.is_file():raise FileNotFoundError(p)
        if p.suffix.lower() not in IMAGE_EXTS:raise ValueError(f'非图片文件: {p}')
        if p not in seen:out.append(p);seen.add(p)
    if not out:raise ValueError('没有找到图片；请指定--files或--folder')
    return out


def cmd_predict(args):
    from PIL import Image
    from core.artifacts import make_artifacts,write_csv
    from model_adapters.registry import Predictor
    file_list=list(args.files)
    if args.file_list:
        extra=json.loads(Path(args.file_list).read_text(encoding='utf-8'))
        if not isinstance(extra,list) or not all(isinstance(x,str) for x in extra):
            raise ValueError('图片清单必须为JSON字符串数组')
        file_list.extend(extra)
    images=files_from_args(file_list,args.folder,args.recursive)
    cfg=get_config()
    if args.oil_delta is not None:
        cfg['segformer_oil_delta']=args.oil_delta
    if args.deeplab_scheme is not None:
        cfg['deeplab_scheme']=args.deeplab_scheme
    output=safe_destination(args.output,'predict')
    output.mkdir(parents=True,exist_ok=True)
    print(f'[START] 图片{len(images)}张 / 模型{args.model} / 输出{output}',flush=True)
    engine=Predictor(args.model,cfg)
    summaries=[];regions=[]
    try:
        for index,path in enumerate(images,1):
            with Image.open(path) as im: rgb=im.convert('RGB')
            engine.synchronize();start=time.perf_counter()
            pred=engine.predict(rgb)
            engine.synchronize()
            guard_info={'status':'disabled'}
            guard_mask=None
            if args.roi_guard:
                from core.screen_guard import apply_guard,save_guard_preview
                pred,guard_info,guard_mask=apply_guard(rgb,pred,args.roi_inset)
                if guard_mask is not None and index<=24:
                    save_guard_preview(rgb,guard_mask,output/'roi_previews'/f'{index:04d}_{path.stem[:32]}.png')
            elapsed=(time.perf_counter()-start)*1000
            record,found=make_artifacts(rgb,pred,path,args.model,output,min_area=args.min_area)
            record['inference_ms']=round(elapsed,2)
            record['display_roi_guard']=guard_info
            # Update per-image JSON with inference timing and provenance.
            jpath=output/'json'/f"{Path(record['mask_path']).stem}.json"
            jpath.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
            summaries.append({'source':str(path),'model':args.model,'width':rgb.width,
                'height':rgb.height,'regions':len(found),'time_ms':round(elapsed,2),
                'oil_pixels':record['predicted_class_pixels']['oil'],
                'stain_pixels':record['predicted_class_pixels']['stain'],
                'scratch_pixels':record['predicted_class_pixels']['scratch'],
                'overlay':record['overlay_path'],'boxes':record['boxes_path'],
                'roi_status':guard_info.get('status','disabled')})
            for i,r in enumerate(found,1):
                x,y,w,h=r['bbox_xywh']
                regions.append({'image':str(path),'model':args.model,'region_id':i,
                    'class_id':r['class_id'],'class':r['class_name'],
                    'x':x,'y':y,'width':w,'height':h,'area_px':r['pixel_area'],
                    'cover_pct':r['image_coverage_pct'],
                    'center_x':r['center_xy'][0],'center_y':r['center_xy'][1]})
            print(f'[PROGRESS] {index}/{len(images)} {path.name} regions={len(found)} {elapsed:.1f}ms',flush=True)
    finally:
        engine.close()
    write_csv(output/'images.csv',summaries,
              ['source','model','width','height','regions','time_ms','oil_pixels','stain_pixels','scratch_pixels','overlay','boxes','roi_status'])
    write_csv(output/'regions.csv',regions,
              ['image','model','region_id','class_id','class','x','y','width','height','area_px','cover_pct','center_x','center_y'])
    (output/'run_info.json').write_text(json.dumps({'model':args.model,
            'images':len(images),'source':'user-selected files','config':cfg,
            'display_roi_guard':{'enabled':bool(args.roi_guard),'inset_px':args.roi_inset,
                                 'scope':'DISPLAY SURFACE ONLY; separate from original research benchmark'},
            'warnings':['Pixel areas and locations are candidates, not QC acceptance results']},
            ensure_ascii=False,indent=2),encoding='utf-8')
    from industrial.report import build_inspection
    report=build_inspection(output)
    print(f'[OUTPUT] {output}\n[INDUSTRIAL REPORT] {report}',flush=True)
    print(f'[FINISH] 共{len(images)}张、{len(regions)}个缺陷连通域',flush=True)


def cmd_audit(args):
    from paper.audit import audit_dataset
    cfg=get_config()
    dataset=Path(args.dataset or cfg['dataset'])
    path=Path(args.output) if args.output else ROOT/'outputs'/'dataset_audit.json'
    result=audit_dataset(dataset,path)
    for k,v in result['split_counts'].items():print(f'[SPLIT] {k}: {v["count"]} unique={v["unique"]}',flush=True)
    for err in result['errors']:print('[ERROR]',err,flush=True)
    for msg in result['warnings']:print('[WARNING]',msg,flush=True)
    print('[OUTPUT]',path,flush=True)
    if result['errors']:raise SystemExit(3)


def cmd_paper(args):
    from paper.report_v3 import build_package
    out=build_package(Path(args.from_dir),Path(args.output) if args.output else None)
    print(f'[OUTPUT] {out}',flush=True)
    print(f'[PDF] {out/"论文实验分析_黑玉兰.pdf"}',flush=True)
    print(f'[DOCX] {out/"论文实验分析_可编辑.docx"}',flush=True)


def cmd_doctor(_):
    import importlib.util
    cfg=get_config()
    print('=== DeepLabV3 Studio 环境检查 ===')
    print('项目:',ROOT,'\nPython:',sys.executable,'\nPython版本:',sys.version.split()[0])
    ready=True
    for name in ('numpy','PIL','torch','cv2','tkinter'):
        try: ok=importlib.util.find_spec(name) is not None
        except Exception:ok=False
        print(f'[依赖] {name}: {"OK" if ok else "缺少"}')
        if name!='tkinter':ready &= ok
    for model in ('deeplab','segformer','fdsnet','deeplab_base'):
        p=model_weights(model,cfg)
        ok=p.is_file()
        print(f'[权重] {model}: {"OK" if ok else "缺少"} {p}')
        if model!='deeplab_base':ready &= ok
    p=Path(cfg['dataset'])
    ok=(p/'JPEGImages').is_dir() and (p/'SegmentationClass').is_dir() and (p/'ImageSets'/'Segmentation'/'val.txt').is_file()
    print(f'[数据集] {"OK" if ok else "缺少"} {p}')
    ready &= ok
    for item in ('deeplab.py','segformer_infer.py','nets/segformer.py',
                 'third_party/fdsnet/fdsnet_core/fdsnet.py'):
        src=ROOT/'research'/item
        ok=src.is_file()
        print(f'[源码] {item}: {"OK" if ok else "缺少"}')
        ready &= ok
    try:
        import torch
        print(f'[GPU] CUDA={torch.cuda.is_available()}',
              (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback'))
    except Exception as e:print('[GPU] 未能读取:',e)
    print('[READY]' if ready else '[NOT READY] 见上面的缺失项；先迁移权重并确认数据集路径。')
    print('检查不会改变模型、文件或安装依赖。')
    if not ready:raise SystemExit(2)


def cmd_migrate(args):
    original=Path(args.from_dir).expanduser().resolve()
    if original==ROOT:
        raise ValueError('原项目与新项目相同；不能从自身迁移')
    if not (original/'deeplab.py').is_file():
        raise FileNotFoundError(f'源目录不像旧 DeepLabV3 项目: {original}')
    cfg=get_config();mapping={
      'deeplab':original/'logs_full_all'/'best_epoch_weights.pth',
      'segformer':original/'logs_segformer_mit_b2'/'best_epoch_weights.pth',
      'fdsnet':original/'third_party'/'fdsnet'/'fdsnet__phone_voc_best_model.pth',
    }
    print('只复制，不删除或覆盖旧项目。')
    for model,src in mapping.items():
        dst=(path_value(cfg['weights']['deeplab']) if model=='deeplab' else model_weights(model,cfg))
        if not src.is_file():
            print(f'[缺少] {model} 原权重: {src}');continue
        if dst.is_file() and not args.force:
            print(f'[跳过] {model} 新目录已有: {dst}');continue
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        if dst.stat().st_size!=src.stat().st_size:
            raise IOError(f'复制后大小不一致: {src} -> {dst}')
        print(f'[OK] {model}: {src} -> {dst}')
    if args.history or args.training:
        history=ROOT/'checkpoints'/'history'
        for name in ('logs_baseline','logs_loss_opt','logs_cbam','logs_full_all','logs_segformer_mit_b2'):
            old=original/name
            if not old.is_dir():continue
            for src in old.glob('*.pth'):
                dst=history/name/src.name
                if args.history:
                    if not dst.exists() or args.force:
                        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
                        print(f'[历史权重] {dst}')
                if args.training:
                    legacy_dest=ROOT/'research'/name/src.name
                    if not legacy_dest.exists() or args.force:
                        legacy_dest.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(src,legacy_dest)
                        print(f'[兼容旧训练入口] {legacy_dest}')
    base_source=original/'logs_baseline'/'best_epoch_weights.pth'
    base_dest=model_weights('deeplab_base',cfg)
    if base_source.is_file() and (not base_dest.exists() or args.force):
        base_dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(base_source,base_dest)
        print('[OK] DeepLab基础权重：',base_dest)
    elif not base_source.is_file():
        print('[WARNING] 缺少DeepLab基础权重；该方案会提示不可用，其他模型可继续评测')
    print('迁移完成。MSD数据集保留原位置，config/local.json中的dataset仅指向它。')


def build_parser():
    p=argparse.ArgumentParser(prog='python main.py',description='ScreenQC Studio V5: 工业检测 + 论文实验，三架构、五项核心推理对照')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor',help='检查源码/数据/权重/环境').set_defaults(fn=cmd_doctor)
    m=sub.add_parser('migrate',help='从旧项目复制权重，保留原项目')
    m.add_argument('--from',dest='from_dir',required=True);m.add_argument('--force',action='store_true');m.add_argument('--history',action='store_true');m.add_argument('--training',action='store_true')
    m.set_defaults(fn=cmd_migrate)
    q=sub.add_parser('predict',help='单张/多张/文件夹批量检测、坐标CSV/JSON、可视化PNG')
    q.add_argument('--model',choices=AVAILABLE,default='segformer_oil')
    q.add_argument('--files',nargs='*',default=[],help='多个图片路径，如 --files a.jpg b.png')
    q.add_argument('--folder',help='输入图片文件夹');q.add_argument('--recursive',action='store_true')
    q.add_argument('--file-list',help='UTF-8 JSON file containing list of image file paths')
    q.add_argument('--roi-guard',action='store_true',help='可选：仅检测LCD显示区域；未通过区域可信度检查时保留原预测')
    q.add_argument('--roi-inset',type=int,default=16,help='有效显示区域内缩像素；建议8~24；需在验证集上单独评测')
    q.add_argument('--output');q.add_argument('--min-area',type=int,default=1,help='仅筛选导出连通域，绝不改变模型原始mask')
    q.add_argument('--oil-delta',type=float,help='实验性油污阈值偏移；仅用于segformer_oil')
    q.add_argument('--deeplab-scheme',type=int,choices=(0,1,2,3),help='DeepLab训练方案编号；0-2需要migrate --history')
    q.set_defaults(fn=cmd_predict)
    e=sub.add_parser('evaluate',help='统一评测和HTML对比')
    e.add_argument('--dataset',help='MSD目录');e.add_argument('--split',choices=('val','test','train','trainval'),default='val')
    e.add_argument('--models',nargs='+',choices=AVAILABLE,default=['deeplab_base','deeplab','segformer_raw','segformer_oil','fdsnet_tuned'])
    e.add_argument('--output');e.add_argument('--max-images',type=int,default=0,help='仅用于调试的样本上限')
    e.add_argument('--samples',type=int,default=0,help='每模型另存前N张可视化样例')
    e.add_argument('--deeplab-scheme',type=int,choices=(0,1,2,3),help='DeepLab训练方案编号')
    e.set_defaults(fn=cmd_evaluate)
    au=sub.add_parser('audit',help='检查MSD数据源、划分重叠、图片/标签是否一一对应')
    au.add_argument('--dataset',help='MSD根目录，默认与旧项目同一路径')
    au.add_argument('--output',help='输出dataset_audit.json')
    au.set_defaults(fn=cmd_audit)
    pp=sub.add_parser('paper',help='生成结构化指标、Word原生表格、PNG插图、黑玉兰PDF和可复制HTML')
    pp.add_argument('--from',dest='from_dir',required=True,help='已有model_comparison.csv的评测目录')
    pp.add_argument('--output',help='论文材料输出目录，默认在评测目录paper_package')
    pp.set_defaults(fn=cmd_paper)
    r=sub.add_parser('report',help='现有评测CSV生成可视化HTML报告（不需要权重）')
    r.add_argument('--from',dest='from_dir',required=True)
    r.set_defaults(fn=cmd_report)
    a=sub.add_parser('analyze',help='仅统计真实标注的缺陷面积与位置（无需权重）')
    a.add_argument('--dataset');a.add_argument('--split',choices=('val','test','train','trainval'),default='val')
    a.add_argument('--output');a.add_argument('--max-images',type=int,default=0)
    a.set_defaults(fn=cmd_analyze)
    c=sub.add_parser('curves',help='历史训练曲线和油污PR候选可视化')
    c.add_argument('--kind',choices=('train','oil'),default='train')
    c.add_argument('--csv',help='kind=oil时指向precision_recall_curve.csv')
    c.add_argument('--output',help='输出HTML路径')
    c.set_defaults(fn=cmd_curves)
    t=sub.add_parser('train',help='调用保留的旧训练脚本；使用前必须确认权重与数据路径')
    t.add_argument('--model',choices=('deeplab','segformer'),required=True)
    t.add_argument('--confirm',action='store_true',help='确认已检查原始训练源码路径、配置、显存并允许执行')
    t.set_defaults(fn=cmd_train)
    g=sub.add_parser('gui',help='启动新版本地桌面Web图形界面');g.set_defaults(fn=cmd_desktop)
    desk=sub.add_parser('desktop',help='新版本地工作台');desk.set_defaults(fn=cmd_desktop)
    ir=sub.add_parser('inspection-report',help='对现有工业预测结果生成完整可编辑报告')
    ir.add_argument('--from',dest='from_dir',required=True);ir.set_defaults(fn=cmd_inspection_report)
    return p


def cmd_evaluate(args):
    from evaluation.runner import evaluate
    cfg=get_config()
    if args.deeplab_scheme is not None:cfg['deeplab_scheme']=args.deeplab_scheme
    dataset=Path(args.dataset or cfg['dataset']).expanduser()
    output=safe_destination(args.output,'evaluate')
    evaluate(dataset,args.split,args.models,output,cfg,args.max_images,args.samples)
    # Generate the dissertation materials in the same run when the dependencies exist.
    try:
        from paper.report_v3 import build_package
        pkg=build_package(output)
        print('[PAPER] '+str(pkg),flush=True)
    except Exception as e:
        print('[REPORT ERROR] 评测CSV已保存，但自动报告生成失败：'+str(e),flush=True)
        raise


def cmd_report(args):
    from visualization.report import make_report
    print('[REPORT]',make_report(Path(args.from_dir).expanduser()))


def cmd_train(args):
    if not args.confirm:
        raise ValueError('训练属于长任务，先检查 docs/训练与复现实验.md，再添加--confirm')
    import subprocess
    file=ROOT/'research'/('train_segformer.py' if args.model=='segformer' else 'train.py')
    print('[WARN] 将执行未改写的历史训练脚本，使用其内部训练参数。',flush=True)
    print('[WARN] 原脚本的预训练权重、恢复路径和数据配置可能需要你先核对。',flush=True)
    print('[START] ',file,flush=True)
    completed=subprocess.run([sys.executable,'-u',str(file)],cwd=str(ROOT/'research'),check=False)
    if completed.returncode!=0:raise RuntimeError(f'旧训练脚本退出码 {completed.returncode}')


def cmd_curves(args):
    from visualization.curves import create_training_report,create_oil_pr_report
    output=Path(args.output) if args.output else ROOT/'outputs'/('train_curves.html' if args.kind=='train' else 'oil_pr.html')
    if args.kind=='train':done=create_training_report(output)
    else:
        source=Path(args.csv) if args.csv else ROOT/'research'/'experiments'/'oil_optimization_results'/'precision_recall_curve.csv'
        done=create_oil_pr_report(source,output)
    print('[REPORT]',done)


def cmd_analyze(args):
    from evaluation.size_analysis import analyze
    config=get_config()
    ds=Path(args.dataset or config['dataset'])
    from app.desktop import _split_counts
    split=args.split
    if not _split_counts(ds).get(split):
        if split=='test' and _split_counts(ds).get('val'):
            split='val'
            print('[NOTICE] test.txt为空，尺寸分析自动改用val.txt；不能当作独立测试集。')
        else: raise ValueError(f'{split}.txt无有效图片')
    analyze(ds,split,safe_destination(args.output,'dataset_analysis'),args.max_images)




def cmd_inspection_report(args):
    from industrial.report import build_inspection
    out=build_inspection(Path(args.from_dir))
    print(f'[OUTPUT] {out}\n[PDF] {out/"工业批量检测报告.pdf"}',flush=True)


def cmd_desktop(_):
    from app.desktop import start
    start()


if __name__=='__main__':
    try:
        parser=build_parser();args=parser.parse_args();args.fn(args)
    except (Exception,KeyboardInterrupt) as err:
        import traceback
        traceback.print_exc()
        sys.exit(1)
