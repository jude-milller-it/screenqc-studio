"""Fair same-list per-pixel comparison; model-specific inference settings are disclosed."""
from __future__ import annotations
import gc
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
from core.metrics import CLASSES, score, confusion_update
from core.dataset import image_by_stem,read_ids,read_gt
from core.artifacts import write_csv,make_artifacts
from model_adapters.registry import Predictor
from core.settings import model_weights
from visualization.report import make_report


def evaluate(dataset: Path, split: str, models: list[str], output: Path,
             settings: dict, max_images: int = 0, samples: int = 0) -> Path:
    # Fail early with a complete missing-checkpoint list, never halfway through five models.
    missing=[f'{key}: {model_weights(key,settings)}' for key in models
             if not model_weights(key,settings).is_file()]
    if missing:
        raise FileNotFoundError('以下已选方案缺少权重，请运行01_COPY_WEIGHTS.cmd或在设置页指定：\n'+'\n'.join(missing))
    ids=read_ids(dataset,split)
    if max_images>0:ids=ids[:max_images]
    output.mkdir(parents=True,exist_ok=True)
    collected=[]; per_classes=[]; per_images=[]
    for mi,key in enumerate(models,1):
        print(f'[MODEL] {mi}/{len(models)} {key}',flush=True)
        predictor=Predictor(key,settings)
        matrix=np.zeros((4,4),dtype=np.int64)
        times=[];path=output/key
        sample_counts={c:0 for c in CLASSES[1:]}
        path.mkdir(exist_ok=True,parents=True)
        try:
            for j,stem in enumerate(ids,1):
                path_img=image_by_stem(dataset,stem)
                with Image.open(path_img) as original:rgb=original.convert('RGB')
                gt=read_gt(dataset/'SegmentationClass'/f'{stem}.png',(rgb.height,rgb.width))
                predictor.synchronize();start=time.perf_counter()
                pred=predictor.predict(rgb)
                predictor.synchronize();elapsed=(time.perf_counter()-start)*1000
                cm1=np.zeros((4,4),dtype=np.int64)
                confusion_update(cm1,gt,pred)
                confusion_update(matrix,gt,pred)
                metric=score(cm1)
                times.append(elapsed)
                per_images.append({'model':key,'image':stem,'time_ms':round(elapsed,2),
                    **{f'{cl}_{kind}':metric['per_class'][cl][kind+'_pct']
                       for cl in CLASSES for kind in ('IoU','Precision','Recall')},
                    'scratch_GT_pixels':metric['per_class']['scratch']['GT_pixels']})
                if samples>0:
                    # Preserve the same original-image examples across all model outputs.
                    classes_here=[c for k,c in enumerate(CLASSES) if k>0 and (gt==k).any()]
                    for cls in classes_here:
                        if sample_counts[cls]<samples:
                            make_artifacts(rgb,pred,path_img,key,path/'examples'/cls)
                            sample_counts[cls]+=1
                if j==1 or j%10==0 or j==len(ids):
                    print(f'[PROGRESS] {j}/{len(ids)} {key}',flush=True)
        finally:
            predictor.close()
            del predictor
            gc.collect()
        final=score(matrix)
        final['model']=key
        final['images']=len(ids)
        final['mean_pipeline_ms']=float(np.mean(times))
        final['p95_pipeline_ms']=float(np.percentile(times,95))
        final['fps']=float(1000/np.mean(times)) if np.mean(times)>0 else 0.0
        row={k:v for k,v in final.items() if k!='per_class'}
        for cls in CLASSES:
            for short,long in (('IoU','IoU_pct'),('P','Precision_pct'),('R','Recall_pct'),('F1','F1_Dice_pct')):
                row[f'{cls}_{short}']=final['per_class'][cls][long]
            per_classes.append({'model':key,'class':cls,**final['per_class'][cls]})
        collected.append(row)
        write_csv(path/'confusion_matrix.csv',
                  [{'gt_class':c, **{CLASSES[i]:int(v) for i,v in enumerate(matrix[j])}}
                   for j,c in enumerate(CLASSES)],['gt_class',*CLASSES])
        (path/'metrics.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'[SCORE] {key}: mIoU={final["mIoU_pct"]:.4f}% Oil={final["per_class"]["oil"]["IoU_pct"]:.4f}%',flush=True)
        # Keep partial results if a subsequent model fails to load.
        _write_summary(output,collected,per_classes,per_images)
    metadata={'split':split,'dataset':str(dataset),'images':len(ids),
        'class_order':CLASSES,'selected_models':models,
        'semantic_stages': {
            'segformer_raw':'512x512 raw / no ROI / no thresholds / same final checkpoint',
            'segformer_scale':'640x1152 raw / no ROI / no thresholds / same final checkpoint',
            'segformer':'640x1152 ROI + class thresholds / same final checkpoint',
            'segformer_oil':'same inference + oil-specific threshold delta / same final checkpoint',
            'segformer_roi':'640x1152 plus ROI only / same final checkpoint',
            'segformer_conf':'640x1152 plus class threshold only / same final checkpoint',
            'segformer_post_bad':'old size/morphology filtering; historical negative control',
            'segformer_edge_guard':'same final oil configuration + opt-in central-display ROI and 16px inner border; experimental',
            'segformer_edge_guard':'same final oil configuration + opt-in central-display ROI and 16px inner border; experimental',
            'deeplab_base':'base checkpoint (different weights from scheme3)',
            'deeplab':'full scheme3 checkpoint',
            'fdsnet_tuned':'author checkpoint; exploratory stain probability factor 1.10 (development search selected baseline)'},
        'weights':{k:str(model_weights(k,settings)) for k in models},
        'settings':{'segformer_hw':[640,1152],'segformer_roi':True,'segformer_fast':True,
            'segformer_oil_delta':settings.get('segformer_oil_delta',.30),
            'deeplab_scheme':settings.get('deeplab_scheme',3),'fdsnet_scale':.75,
            'fdsnet_stain_scale':settings.get('fdsnet_stain_scale',1.10)},
        'limitations':['FDSNet author weights may overlap local validation images',
            'This is pixel-level measurement, not product-level defect detection rate',
            'SegFormer oil delta selected on prior validation data; not an independent test',
            'Models have different inference resolutions and pretrained checkpoints',
            'FDSNet probability calibration is a validation-set exploratory candidate; author development search selected unmodified baseline']}
    (output/'experiment_info.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    report=make_report(output)
    print(f'[OUTPUT] {output}\n[REPORT] {report}',flush=True)
    return output


def _write_summary(output,rows,classes,images):
    write_csv(output/'model_comparison.csv',rows)
    write_csv(output/'per_class_metrics.csv',classes)
    write_csv(output/'per_image.csv',images)
