# -*- coding: utf-8 -*-
"""Explicit evidence trail for split provenance and existing MSD layout."""
from pathlib import Path
import json, hashlib
from core.dataset import image_by_stem


def audit_dataset(ds:Path,output:Path|None=None)->dict:
    ds=Path(ds)
    required=[ds/'JPEGImages',ds/'SegmentationClass',ds/'ImageSets'/'Segmentation']
    errors=[];warnings=[]
    for p in required:
        if not p.is_dir():errors.append('目录不存在: '+str(p))
    splits={}
    if not errors:
        for name in ('train','val','test','trainval'):
            path=ds/'ImageSets'/'Segmentation'/f'{name}.txt'
            if not path.is_file():
                errors.append('缺少划分文件: '+str(path));continue
            items=[x.strip() for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
            splits[name]={'count':len(items),'unique':len(set(items)),
                          'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'ids':items}
            if len(items)!=len(set(items)):errors.append(name+'包含重复ID')
        for name in ('train','val','test'):
            if name in splits and not splits[name]['count']:
                warnings.append(name+'为空：不能据此进行独立评测')
        for a,b in (('train','val'),('train','test'),('val','test')):
            if a in splits and b in splits:
                overlaps=sorted(set(splits[a]['ids'])&set(splits[b]['ids']))
                if overlaps:errors.append(f'{a}与{b}重叠{len(overlaps)}张：示例{overlaps[:5]}')
        for key in ('train','val','test'):
            if key in splits:
                for stem in splits[key]['ids']:
                    try:image_by_stem(ds,stem)
                    except FileNotFoundError:errors.append(f'{key}缺原图: {stem}')
                    if not (ds/'SegmentationClass'/f'{stem}.png').is_file():
                        errors.append(f'{key}缺标签: {stem}')
                    if len(errors)>30:errors.append('文件缺失过多，停止枚举');break
        previous=ds/'ImageSets'/'Segmentation_old'
        if previous.is_dir():
            old={}
            for name in ('train','val','test'):
                f=previous/f'{name}.txt'
                if f.is_file():old[name]=set(x.strip() for x in f.read_text(encoding='utf-8-sig').splitlines() if x.strip())
            if 'train' in splits and 'test' in old:
                overlap=len(set(splits['train']['ids']) & old['test'])
                if overlap:warnings.append(f'旧版test与当前train重叠{overlap}张；不可把旧版test直接当当前权重的独立测试集。')
    counts={k:{a:v for a,v in row.items() if a!='ids'} for k,row in splits.items()}
    result={'dataset':str(ds),'valid_structure':not errors,'split_counts':counts,'errors':errors,'warnings':warnings,
            'source_usage':{'legacy_deeplab_train':'ImageSets/Segmentation/train.txt',
            'legacy_segformer_train':'ImageSets/Segmentation/train.txt',
            'legacy_eval':'ImageSets/Segmentation/val.txt',
            'rgb':'JPEGImages/<id>.jpg','mask':'SegmentationClass/<id>.png'},
            'important':'目录和划分完整性检查不等于证明训练历史无数据泄漏；历史模型可能使用过其他划分。'}
    if output:
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result
