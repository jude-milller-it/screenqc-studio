# -*- coding: utf-8 -*-
"""Local browser desktop front-end. No web-framework, Node or Qt installation required.

Only listens on 127.0.0.1; request token protects write operations. API key is never placed in subprocess arguments or returned status objects.
On Windows, it is optionally saved encrypted with per-user DPAPI.
"""
from __future__ import annotations
import json,os,secrets,sys,subprocess,threading,webbrowser,time,mimetypes,shutil
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs,unquote
from pathlib import Path
from core.settings import ROOT,get_config,save_config,path_value
from model_adapters.registry import AVAILABLE,MAIN_MODELS,STUDY_MODELS,INDUSTRIAL_MODELS,DISPLAY
from paper.audit import audit_dataset

ASSETS=ROOT/'app'/'web_assets'
EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}


def _output_roots():
    """Read historic reports in-place from earlier releases; do not copy images."""
    current = path_value(get_config().get('outputs','outputs'))
    roots = [current]
    for version in ('v5_1','v5','v4'):
        previous = ROOT.parent / ('DeepLabV3_Studio_' + version) / 'outputs'
        if previous.is_dir() and previous.resolve() not in [r.resolve() for r in roots]:
            roots.append(previous)
    return roots


def _split_counts(dataset):
    root=Path(dataset)/'ImageSets'/'Segmentation'
    result={}
    for sp in ('train','val','test','trainval'):
        p=root/f'{sp}.txt'
        if p.is_file():
            try: result[sp]=len([x for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()])
            except Exception:result[sp]=None
        else:result[sp]=None
    return result


def _latest(kind):
    dirs=[]
    for root in _output_roots():
        for prefix in ('evaluation','evaluate','predict','inspection'):
            p=root/prefix
            if p.is_dir():
                dirs.extend(q for q in p.iterdir() if q.is_dir())
    if kind=='paper':
        dirs=[d for d in dirs if (d/'model_comparison.csv').is_file()]
    else:
        dirs=[d for d in dirs if (d/'images.csv').is_file()]
    return str(sorted(dirs,key=lambda p:p.stat().st_mtime,reverse=True)[0]) if dirs else ''


def _run_image_path(run: Path, value: str) -> Path:
    """Resolve historical images after outputs were copied to a different project.

    Never serve arbitrary paths from a CSV. Prefer a PNG inside the selected run.
    """
    raw=Path(str(value or ''))
    # Normalize the filename only. Do not trust old absolute paths after migration.
    name=str(value or '').replace('\\','/').rsplit('/',1)[-1]
    if not name.lower().endswith('.png'):
        return run/'_invalid.png'
    for folder in ('boxes','overlays','masks'):
        candidate=run/folder/name
        if candidate.is_file():return candidate
    # Direct path is permitted only within selected run.
    try:
        if raw.is_file() and raw.resolve().is_relative_to(run.resolve()):return raw
    except (OSError,ValueError):pass
    return run/'_missing.png'


REPORT_FILES = {
    'inspection': [('PDF','工业批量检测报告.pdf'),('Word','工业批量检测报告.docx'),('HTML','工业批量检测报告.html'),('Markdown','工业批量检测报告.md')],
    'paper': [('PDF','论文实验分析_黑玉兰.pdf'),('Word','论文实验分析_可编辑.docx'),('HTML','论文实验分析.html'),('Markdown','论文实验分析.md')],
}

def report_catalog(run: Path, mode: str) -> dict:
    if mode not in REPORT_FILES: raise ValueError('不支持的报告类别')
    run = Path(run).resolve()
    folders = ('inspection_report', 'inspection_report_ai') if mode=='inspection' else ('paper_package','paper_package_ai')
    result = {'run':str(run), 'mode':mode}
    for label, dirname in zip(('standard','ai'), folders):
        folder = run / dirname
        result[label] = {'directory':str(folder) if folder.is_dir() else '',
                         'files':{fmt:str(folder / filename) for fmt,filename in REPORT_FILES[mode] if (folder / filename).is_file()}}
    return result


def _choose(kind,initial):
    """Use a short-lived Tk process so the native OS picker has its own main thread."""
    if kind not in ('folder','files','file'):raise ValueError('不支持的选择器')
    root=Path(initial or '').expanduser()
    if root.is_file():root=root.parent
    if not root.is_dir():root=Path.home()
    snippet='''import json,sys,tkinter as tk\nfrom tkinter import filedialog\nroot=tk.Tk();root.withdraw();root.attributes('-topmost',True)\ninitial=sys.argv[2]\nif sys.argv[1]=='files': result=list(filedialog.askopenfilenames(parent=root,initialdir=initial,filetypes=[('图片文件','*.jpg *.jpeg *.png *.bmp *.tif *.tiff'),('所有文件','*.*')]))\nelif sys.argv[1]=='file': result=filedialog.askopenfilename(parent=root,initialdir=initial,filetypes=[('PyTorch模型','*.pth *.pt *.ckpt'),('所有文件','*.*')])\nelse: result=filedialog.askdirectory(parent=root,initialdir=initial)\nprint(json.dumps(result,ensure_ascii=False))\nroot.destroy()'''
    proc=subprocess.run([sys.executable,'-c',snippet,kind,str(root)],capture_output=True,text=True,encoding='utf-8',timeout=180,check=False)
    if proc.returncode:raise RuntimeError('系统文件选择器无法启动：'+proc.stderr[-450:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


class Tasks:
    def __init__(self):
        self.lock=threading.RLock()
        self.token=secrets.token_urlsafe(24)
        self.task={'id':0,'phase':'idle','action':'','progress':'','lines':[],'result':'','report':'','error':''}
        self.proc=None
        self.last={'paper':_latest('paper'),'inspection':_latest('inspection')}
    def status(self):
        with self.lock:
            return {'task':json.loads(json.dumps(self.task)),'last':dict(self.last)}
    def log(self,msg):
        with self.lock:
            self.task['lines']=(self.task['lines']+[str(msg).rstrip()])[-140:]
            if str(msg).startswith('[PROGRESS]'):self.task['progress']=str(msg).split('[PROGRESS]',1)[1].strip()
    def start(self,action,args):
        with self.lock:
            if self.task['phase']=='running':raise ValueError('已有任务正在运行，请先结束或取消它')
            self.task={'id':self.task['id'],'phase':'running','action':action,'progress':'准备中','lines':[],'result':'','report':'','error':''}
        thread=threading.Thread(target=self._worker,args=(action,args),daemon=True)
        thread.start()
        return self.status()
    def _worker(self,action,args):
        output=None
        try:
            commands=[]
            cfg=get_config()
            output_root=path_value(cfg.get('outputs','outputs'))
            ds=str(args.get('dataset') or cfg['dataset'])
            split=str(args.get('split') or 'val')
            if split not in ('train','val','test','trainval'):raise ValueError('不支持的数据划分')
            if action=='evaluate':
                ids=args.get('models') or list(MAIN_MODELS)
                if not isinstance(ids,list) or not ids or any(m not in AVAILABLE for m in ids):raise ValueError('请选择有效模型')
                if not (Path(ds)/'JPEGImages').is_dir():raise ValueError('请在设置中指定包含JPEGImages的MSD数据集目录')
                counts=_split_counts(ds)
                if not counts.get(split):raise ValueError(f'{split}.txt目前没有可评测图片；当前val={counts.get("val")}张，可选择val')
                output=output_root/'evaluation'/time.strftime('%Y%m%d_%H%M%S')
                # Save 2 representative figures for every class by default.
                commands=['evaluate','--dataset',ds,'--split',split,'--models',*ids,'--samples',str(int(args.get('samples',2))),
                          '--deeplab-scheme','3','--output',str(output)]
            elif action=='predict':
                files=[str(Path(f).resolve()) for f in (args.get('files') or []) if str(f).strip()]
                folders=[str(Path(f).resolve()) for f in (args.get('folders') or []) if str(f).strip()]
                if not files and not folders:raise ValueError('请至少添加一张图片或一个图片文件夹')
                for f in files:
                    if not Path(f).is_file() or Path(f).suffix.lower() not in EXTS:raise ValueError('图片路径无效：'+f)
                for f in folders:
                    if not Path(f).is_dir():raise ValueError('文件夹路径无效：'+f)
                model=str(args.get('model') or 'segformer_oil')
                if model not in INDUSTRIAL_MODELS:raise ValueError('工业检测仅支持DeepLab完整、SegFormer最终优化、FDSNet优化候选')
                output=output_root/'inspection'/time.strftime('%Y%m%d_%H%M%S')
                commands=['predict','--model',model,'--deeplab-scheme','3',
                          '--min-area',str(max(1,int(args.get('min_area',1)))),'--output',str(output)]
                if bool(args.get('roi_guard')):
                    commands += ['--roi-guard','--roi-inset',str(max(0,min(40,int(args.get('roi_inset',16)))))]
                # Avoid Windows CreateProcess command-line limit (WinError 206).
                # JSON contains all selected files; the subprocess reads it in Python.
                if folders:
                    from main import files_from_args
                    for folder in folders:
                        files.extend(str(p) for p in files_from_args([],folder,True))
                unique=list(dict.fromkeys(files))
                if not unique:raise ValueError('所选文件夹内没有图片')
                output.mkdir(parents=True,exist_ok=True)
                manifest=output/'input_file_list.json'
                manifest.write_text(json.dumps(unique,ensure_ascii=False,indent=2),encoding='utf-8')
                commands+=['--file-list',str(manifest)]
            elif action=='analyze':
                if not (Path(ds)/'SegmentationClass').is_dir():raise ValueError('找不到MSD分割标签目录')
                counts=_split_counts(ds)
                if not counts.get(split):
                    if split=='test' and counts.get('val'):
                        self.log('[INFO] test.txt为空；微小缺陷尺寸统计使用val.txt（验证集），不会把val伪装成test。')
                        split='val'
                    else:raise ValueError(f'{split}.txt没有数据，请选择val/train')
                output=output_root/'dataset_analysis'/time.strftime('%Y%m%d_%H%M%S')
                commands=['analyze','--dataset',ds,'--split',split,'--output',str(output)]
            elif action=='audit':
                output=output_root/'audits'/time.strftime('%Y%m%d_%H%M%S')
                commands=['audit','--dataset',ds,'--output',str(output/'dataset_audit.json')]
            elif action=='paper':
                src=Path(args.get('source') or self.last['paper'] or '')
                if not (src/'model_comparison.csv').is_file():
                    fallback=ROOT/'research'/'experiments'/'three_models'/'unified_results_val'
                    if (fallback/'model_comparison.csv').is_file():
                        src=fallback;self.log('[INFO] 尚无新的评测结果，使用项目内历史三模型CSV生成演示报告；不可当作新四方案评测。')
                    else:raise ValueError('尚无评测数据。请先点击“运行论文评测”')
                output=src
                commands=['paper','--from',str(src)]
            elif action=='inspection_report':
                src=Path(args.get('source') or self.last['inspection'] or '')
                if not (src/'images.csv').is_file():raise ValueError('请先完成工业图片检测')
                output=src
                commands=['inspection-report','--from',str(src)]
            elif action=='doctor':commands=['doctor']
            elif action=='migrate':
                src=str(args.get('old_project') or r'D:\AI\projects\DeepLabV3')
                commands=['migrate','--from',src]
            else:raise ValueError('不支持的任务')
            cmd=[sys.executable,'-u',str(ROOT/'main.py'),*commands]
            env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['PYTHONUNBUFFERED']='1'
            self.log('[RUN] '+action)
            self.proc=subprocess.Popen(cmd,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env)
            assert self.proc.stdout
            for line in self.proc.stdout:self.log(line)
            code=self.proc.wait()
            self.proc=None
            if code:raise RuntimeError(f'任务退出代码 {code}；请查看日志中的Traceback。')
            if action=='evaluate':
                self.last['paper']=str(output);report=output/'paper_package'/'论文实验分析_黑玉兰.pdf'
            elif action=='paper':
                self.last['paper']=str(output);report=output/'paper_package'/'论文实验分析_黑玉兰.pdf'
            elif action=='predict':
                self.last['inspection']=str(output)
                from industrial.report import build_inspection
                self.log('[REPORT] 正在生成工业全链路报告…')
                rep=output/'inspection_report'
                if not (rep/'工业批量检测报告.pdf').is_file():rep=build_inspection(output)
                report=rep/'工业批量检测报告.pdf'
                self.log('[REPORT] '+str(report))
            elif action=='inspection_report':report=output/'inspection_report'/'工业批量检测报告.pdf'
            elif action=='analyze':report=output/'size_report.html'
            elif action=='audit':report=output/'dataset_audit.json'
            else:report=None
            with self.lock:
                self.task['phase']='done';self.task['progress']='完成'
                self.task['result']=str(output) if output else ''
                self.task['report']=str(report) if report and report.is_file() else ''
        except Exception as ex:
            self.log('[ERROR] '+str(ex))
            with self.lock:self.task['phase']='failed';self.task['error']=str(ex)
    def cancel(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate();self.log('[STOP] 用户终止任务，输出可能不完整')
                return True
        return False
    def ai_start(self,args):
        with self.lock:
            if self.task['phase']=='running':raise ValueError('已有任务正在运行')
            self.task={'id':self.task['id'],'phase':'running','action':'ai','progress':'准备结构化数据',
                       'lines':[],'result':'','report':'','error':''}
        threading.Thread(target=self._ai_worker,args=(args,),daemon=True).start()
        return self.status()
    def _ai_worker(self,args):
        try:
            kind=args.get('mode','paper')
            source=Path(args.get('source') or self.last['paper' if kind=='paper' else 'inspection'])
            provider=str(args.get('provider') or get_config().get('ai',{}).get('provider','DeepSeek'))
            model=str(args.get('model') or get_config().get('ai',{}).get('model','deepseek-flash'))
            base=str(args.get('base') or get_config().get('ai',{}).get('base','https://api.deepseek.com'))
            from core.secure_key import load_api_key
            key=str(args.get('key') or load_api_key() or '')
            from paper.llm import analyze
            if kind=='paper':
                from paper.workflow import make_structured
                from paper.report_v3 import build_package
                data=make_structured(source)
                from paper.evidence import research_evidence
                data['evidence']=research_evidence(data)
            else:
                from industrial.report import build_inspection
                src=source/'inspection_report'/'inspection_data.json'
                if not src.is_file():build_inspection(source)
                data=json.loads(src.read_text(encoding='utf-8'))
            self.log('[AI] 只发送结构化聚合指标，不传图片或权重；API Key只用于当前请求。')
            text,usage=analyze(data,provider,base,model,key,mode='industrial' if kind!='paper' else 'paper')
            from paper.llm import parse_sections
            structured=parse_sections(text,kind)
            from paper.ai_quality import audit_analysis
            quality=audit_analysis(data,structured,mode=kind)
            if quality['unverified_claims']:
                self.log(f"[AI REVIEW] 发现{len(quality['unverified_claims'])}处未能自动核实的数值/断言；已记录待核清单，仍需人工审核。")
            self.log('[AI] 已取得模型分析，正在按章节插入图表旁。')
            if kind=='paper':
                out=build_package(source,source/'paper_package_ai',ai_text=structured)
                report=out/'论文实验分析_黑玉兰.pdf'
                filename=out/'AI辅助分析_可审阅.json'
            else:
                out=build_inspection(source,ai_text=structured)
                report=out/'工业批量检测报告.pdf'
                filename=out/'AI辅助分析_可审阅.json'
            filename.write_text(json.dumps({'analysis':structured,'quality_audit':quality,'usage':usage,
               'source_evidence_file':'evidence_manifest.json','input_excludes_images':True},ensure_ascii=False,indent=2),encoding='utf-8')
            (out/'AI分析校验.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2),encoding='utf-8')
            with self.lock:
                self.task['phase']='done';self.task['progress']='完成'
                self.task['result']=str(source);self.task['report']=str(report)
        except Exception as e:
            self.log('[ERROR] '+str(e))
            with self.lock:self.task['phase']='failed';self.task['error']=str(e)



class TaskHub:
    """Independent task records. GPU jobs queue to avoid OOM on 6GB cards;
    CPU/report/AI operations run simultaneously with GPU work."""
    def __init__(self):
        self.token=secrets.token_urlsafe(24)
        self.lock=threading.RLock()
        self.gpu_gate=threading.Semaphore(1)
        self.jobs={}
        self.serial=0
        self.last={'paper':_latest('paper'),'inspection':_latest('inspection')}

    def _new(self,action):
        with self.lock:
            self.serial+=1
            jid=self.serial
            slot=Tasks()
            slot.task.update(id=jid,action=action,phase='queued',progress='等待GPU资源' if action in ('predict','evaluate') else '排队准备',lines=[],report='',result='',error='')
            slot.last.update(self.last)
            self.jobs[jid]=slot
            return jid,slot

    def _launch(self,jid,slot,action,args,ai=False):
        gpu=action in ('predict','evaluate')
        try:
            if gpu:self.gpu_gate.acquire()
            with slot.lock:
                if slot.task['phase']=='cancelled':return
                slot.task['phase']='idle'
            if ai:slot.ai_start(args)
            else:slot.start(action,args)
            with slot.lock:slot.task['id']=jid
            while True:
                with slot.lock:phase=slot.task['phase']
                if phase in ('done','failed','cancelled'):break
                time.sleep(.12)
            with self.lock:
                for key in ('paper','inspection'):
                    old=slot.last.get(key)
                    if old and Path(old).is_dir():self.last[key]=old
        except Exception as ex:
            with slot.lock:
                slot.task.update(phase='failed',error=str(ex))
        finally:
            if gpu:self.gpu_gate.release()

    def start(self,action,args):
        if action not in ('predict','evaluate','analyze','audit','paper','inspection_report','doctor','migrate'):
            raise ValueError('未知任务')
        jid,slot=self._new(action)
        threading.Thread(target=self._launch,args=(jid,slot,action,args),daemon=True).start()
        return self.status()

    def ai_start(self,args):
        jid,slot=self._new('ai')
        kind=args.get('mode','paper')
        if not args.get('source'):
            args={**args,'source':self.last.get('paper' if kind=='paper' else 'inspection','')}
        threading.Thread(target=self._launch,args=(jid,slot,'ai',args,True),daemon=True).start()
        return self.status()

    def status(self):
        with self.lock:
            jobs=[]
            for jid,slot in sorted(self.jobs.items(),reverse=True)[:40]:
                with slot.lock:jobs.append(json.loads(json.dumps(slot.task)))
            current=jobs[0] if jobs else {'id':0,'phase':'idle','action':'','progress':'','lines':[],'result':'','report':'','error':''}
            return {'task':current,'tasks':jobs,'last':dict(self.last)}

    def cancel(self,jid=None):
        with self.lock:
            if jid is None:jid=next((i for i,s in sorted(self.jobs.items(),reverse=True) if s.task['phase']=='running'),None)
            slot=self.jobs.get(int(jid)) if jid else None
        if slot is None:return False
        with slot.lock:
            if slot.task['phase']=='queued':slot.task.update(phase='cancelled',error='任务已取消');return True
        return slot.cancel()


class AppHandler(BaseHTTPRequestHandler):
    manager:Tasks
    def log_message(self,*args):pass
    def _json(self,data,code=200):
        raw=json.dumps(data,ensure_ascii=False).encode('utf-8')
        self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    def _file(self,p,mime=None):
        if not p.is_file():self.send_error(404);return
        data=p.read_bytes();self.send_response(200);self.send_header('Content-Type',mime or mimetypes.guess_type(p.name)[0] or 'application/octet-stream');self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def _permitted(self,p):
        p=p.resolve();roots=tuple(_output_roots())+(ROOT/'research'/'experiments',)
        return any(p.is_relative_to(r.resolve()) for r in roots)
    def do_GET(self):
        path=urlparse(self.path);route=path.path;args=parse_qs(path.query)
        if route=='/':return self._file(ASSETS/'index.html','text/html; charset=utf-8')
        if route.startswith('/assets/'):
            p=(ASSETS/route[len('/assets/'):]).resolve()
            return self._file(p) if p.is_relative_to(ASSETS.resolve()) else self.send_error(403)
        if route=='/api/state':
            cfg=get_config()
            weights={m:str((ROOT/cfg['weights']['segformer' if m.startswith('segformer') else m]).resolve())
                     for m in ('deeplab','segformer','fdsnet')}
            return self._json({'settings':{'dataset':cfg['dataset'],'language':cfg.get('language','zh'),
                'old_project':r'D:\AI\projects\DeepLabV3','outputs':str(path_value(cfg.get('outputs','outputs'))),
                'ai':cfg.get('ai',{}),'ui_scale':cfg.get('ui_scale',110)},
                'weights_paths':{**cfg.get('weights',{}),'deeplab_base':cfg.get('deeplab_variant_weights',{}).get('0','')},
                'split_counts':_split_counts(cfg['dataset']),
                'weights':{**{m:Path(p).is_file() for m,p in weights.items()},'deeplab_base':__import__('core.settings',fromlist=['model_weights']).model_weights('deeplab_base',cfg).is_file()},
                'models':[{'id':k,'name':DISPLAY[k]} for k in AVAILABLE],
                'industrial_models':list(INDUSTRIAL_MODELS),'research_models':list(MAIN_MODELS),
                'ai_key_saved':__import__('core.secure_key',fromlist=['has_api_key']).has_api_key(),
                **self.manager.status()})
        if route=='/api/preview':
            raw=args.get('path',[''])[0];p=Path(raw)
            if p.is_file() and p.suffix.lower() in EXTS and p.stat().st_size<35*1024*1024:return self._file(p)
            return self.send_error(404)
        if route=='/api/result':
            raw=args.get('path',[''])[0];p=Path(raw)
            if self._permitted(p):return self._file(p)
            return self.send_error(403)
        if route=='/api/inspection-preview':
            import csv
            run=Path(args.get('run',[''])[0]).resolve()
            if not self._permitted(run) or not (run/'images.csv').is_file():return self.send_error(403)
            with (run/'images.csv').open(encoding='utf-8-sig',newline='') as f:
                records=list(csv.DictReader(f))
            # Sample class-diverse pictures; do not show the first six oil samples.
            diverse=[];chosen=set()
            for cl in ('oil','stain','scratch'):
                pool=sorted(records,key=lambda r:float(r.get(cl+'_pixels') or 0),reverse=True)
                for row in pool:
                    key=row.get('source')
                    if key not in chosen and float(row.get(cl+'_pixels') or 0)>0:
                        diverse.append(row);chosen.add(key)
                        if sum(float(r.get(cl+'_pixels') or 0)>0 for r in diverse)>=2:break
            for row in records:
                if len(diverse)>=6:break
                if row.get('source') not in chosen:diverse.append(row);chosen.add(row.get('source'))
            previews=[]
            for row in diverse:
                image=_run_image_path(run,row.get('boxes') or row.get('overlay') or '')
                if image.is_file() and image.suffix.lower()=='.png' and image.resolve().is_relative_to(run):
                    previews.append({'filename':Path(row.get('source') or '').name,
                        'image':str(image),'regions':row.get('regions') or '0',
                        'time_ms':row.get('time_ms') or '0'})
            return self._json({'previews':previews})
        if route=='/api/report-catalog':
            run=Path(args.get('run',[''])[0]).resolve()
            mode=args.get('mode',['inspection'])[0]
            if not self._permitted(run) or not run.is_dir():return self.send_error(403)
            try:return self._json(report_catalog(run,mode))
            except ValueError:return self.send_error(400)
        if route=='/api/report-status':
            run=Path(args.get('run',[''])[0]).resolve()
            if not self._permitted(run) or not run.is_dir():return self.send_error(403)
            return self._json({
                'paper_standard':(run/'paper_package'/'论文实验分析_黑玉兰.pdf').is_file(),
                'paper_ai':(run/'paper_package_ai'/'论文实验分析_黑玉兰.pdf').is_file(),
                'inspection_standard':(run/'inspection_report'/'工业批量检测报告.pdf').is_file(),
                'inspection_ai':(run/'inspection_report_ai'/'工业批量检测报告.pdf').is_file()})
        if route=='/api/history':
            typ=args.get('type',['inspection'])[0]
            if typ not in ('inspection','paper'):return self.send_error(400)
            roots=[r/('inspection' if typ=='inspection' else 'evaluation') for r in _output_roots()]
            entries=[]
            candidates=[d for root in roots if root.is_dir() for d in root.iterdir() if d.is_dir()]
            if candidates:
                for d in sorted(candidates,key=lambda x:x.stat().st_mtime,reverse=True)[:100]:
                    images=d/'images.csv';info=d/'run_info.json'
                    metadata={}
                    if info.is_file():
                        try:metadata=json.loads(info.read_text(encoding='utf-8'))
                        except Exception:pass
                    count=0
                    if images.is_file():
                        import csv
                        with images.open(encoding='utf-8-sig',newline='') as f:count=max(0,sum(1 for _ in f)-1)
                    entries.append({'path':str(d),'name':d.name,'images':count,'model':metadata.get('model',''),
                        'created':time.strftime('%Y-%m-%d %H:%M',time.localtime(d.stat().st_mtime)),
                        'report':str(d/('inspection_report' if typ=='inspection' else 'paper_package')/('工业批量检测报告.pdf' if typ=='inspection' else '论文实验分析_黑玉兰.pdf'))})
            return self._json({'runs':entries})
        if route=='/api/inspection-images':
            import csv
            run=Path(args.get('run',[''])[0]).resolve()
            if not self._permitted(run) or not (run/'images.csv').is_file():return self.send_error(403)
            with (run/'images.csv').open(encoding='utf-8-sig',newline='') as f:images=list(csv.DictReader(f))
            regions=[]
            if (run/'regions.csv').is_file():
                with (run/'regions.csv').open(encoding='utf-8-sig',newline='') as f:regions=list(csv.DictReader(f))
            from collections import Counter
            byimage={}
            for r in regions:
                x=byimage.setdefault(r.get('image',''),Counter());x[r.get('class','')]+=1
            tag=args.get('class',['all'])[0]
            sort=args.get('sort',['name'])[0]
            items=[]
            for i,r in enumerate(images):
                pixels={k:int(float(r.get(k+'_pixels') or 0)) for k in ('oil','stain','scratch')}
                areas=dict(byimage.get(r.get('source',''),{}))
                if tag in ('oil','stain','scratch') and not (pixels[tag]>0 or areas.get(tag,0)>0):continue
                items.append({'index':i,'filename':Path(r.get('source','')).name,'source':r.get('source'),
                    'image':str(_run_image_path(run,r.get('boxes') or r.get('overlay') or '')),'regions':int(float(r.get('regions') or 0)),
                    'pixels':pixels,'areas':areas,'time_ms':r.get('time_ms')})
            if sort=='pixels':items.sort(key=lambda r:sum(r['pixels'].values()),reverse=True)
            elif sort=='regions':items.sort(key=lambda r:r['regions'],reverse=True)
            elif sort=='time':items.sort(key=lambda r:float(r['time_ms'] or 0),reverse=True)
            else:items.sort(key=lambda r:r['filename'])
            offset=max(0,int(args.get('offset',['0'])[0]));limit=min(60,max(1,int(args.get('limit',['24'])[0])))
            return self._json({'total':len(items),'items':items[offset:offset+limit]})
        if route=='/api/entries':
            typ=args.get('type',['paper'])[0]
            root=path_value(get_config().get('outputs','outputs'))/('evaluation' if typ=='paper' else 'inspection')
            roots=[d for d in root.iterdir() if d.is_dir()] if root.is_dir() else []
            return self._json({'runs':[{'path':str(d),'name':d.name} for d in sorted(roots,key=lambda p:p.name,reverse=True)[:20]]})
        return self.send_error(404)
    def do_POST(self):
        if self.headers.get('X-Session-Token')!=self.manager.token:return self._json({'error':'拒绝未经授权的请求'},403)
        try:
            size=int(self.headers.get('Content-Length','0'))
            if size<0 or size>10_000_000:raise ValueError('请求内容过大')
            payload=json.loads(self.rfile.read(size) or b'{}')
            route=urlparse(self.path).path
            if route=='/api/start':return self._json(self.manager.start(payload.get('action'),payload))
            if route=='/api/ai':return self._json(self.manager.ai_start(payload))
            if route=='/api/cancel':return self._json({'stopped':self.manager.cancel(payload.get('id'))})
            if route=='/api/credentials':
                key=str(payload.get('api_key','')).strip()
                if not key:raise ValueError('请输入API Key')
                from core.secure_key import save_api_key,load_api_key
                save_api_key(key)
                if load_api_key()!=key:raise RuntimeError('密钥加密保存后校验失败')
                return self._json({'ok':True,'saved':True})
            if route=='/api/settings':
                cfg=get_config();fields=payload.get('settings',{})
                for k in ('language','dataset','deeplab_scheme','segformer_oil_delta','outputs','ui_scale'):
                    if k in fields:
                        v=fields[k]
                        if k=='language' and v not in ('zh','en'):raise ValueError('语言只能是zh/en')
                        if k=='dataset' and not (Path(v)/'JPEGImages').is_dir():raise ValueError('请选择MSD数据集根目录')
                        cfg[k]=v
                if 'weights' in fields:
                    paths=fields['weights']
                    if not isinstance(paths,dict):raise ValueError('权重路径格式无效')
                    for model in ('deeplab','segformer','fdsnet'):
                        if model in paths:
                            value=str(paths[model]).strip()
                            if not value:raise ValueError(model+'权重路径不能为空')
                            cfg['weights'][model]=value
                if 'ai' in fields:
                    ai=fields['ai']
                    if not isinstance(ai,dict):raise ValueError('AI配置格式错误')
                    for k in ('provider','model','base'):
                        if k in ai: cfg.setdefault('ai',{})[k]=str(ai[k]).strip()[:500]
                if 'deeplab_base' in fields and str(fields['deeplab_base']).strip():
                    cfg.setdefault('deeplab_variant_weights',{})['0']=str(fields['deeplab_base']).strip()
                if 'api_key' in payload and str(payload['api_key']).strip():
                    from core.secure_key import save_api_key
                    save_api_key(str(payload['api_key']).strip())
                save_config(cfg)
                return self._json({'ok':True,'split_counts':_split_counts(cfg['dataset']),
                                   'ai_key_saved':__import__('core.secure_key',fromlist=['has_api_key']).has_api_key()})
            if route=='/api/open-folder':
                target=Path(payload.get('path','')).resolve()
                if not self._permitted(target) or not target.is_dir():raise ValueError('不允许打开这个目录')
                if os.name=='nt':os.startfile(str(target))
                else:subprocess.Popen(['xdg-open',str(target)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                return self._json({'ok':True})
            if route=='/api/choose':return self._json({'selection':_choose(payload.get('kind','folder'),payload.get('initial',''))})
            raise ValueError('未知的操作')
        except Exception as e:self._json({'error':str(e)},400)


def start():
    manager=TaskHub()
    class Handler(AppHandler):pass
    Handler.manager=manager
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    url=f'http://127.0.0.1:{server.server_address[1]}/?token={manager.token}'
    print('[SCREENQC] Local desktop portal: '+url,flush=True)
    if os.environ.get('SCREENQC_NO_BROWSER')!='1':
        threading.Timer(.6,lambda:webbrowser.open(url)).start()
    try:server.serve_forever(poll_interval=.3)
    finally:server.server_close()


if __name__=='__main__':start()
