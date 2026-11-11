# -*- coding: utf-8 -*-
"""Non-generative audit of AI analysis against the structured evidence package.
Flags ungrounded numerical percentages rather than silently accepting unsupported claims.
"""
from __future__ import annotations
import re,math

def audit_analysis(source:dict, sections:dict, mode='paper') -> dict:
    numbers=[]
    def walk(o):
        if isinstance(o,dict):
            for k,v in o.items():
                if k.lower() in {'x','y','width','height','time_ms','source_dir'}:continue
                walk(v)
        elif isinstance(o,list):
            for v in o:walk(v)
        elif isinstance(o,(int,float)) and not isinstance(o,bool) and math.isfinite(o):numbers.append(float(o))
    walk(source)
    # Values are allowed with reasonable rounding drift (e.g. 78.657 -> 78.66).
    flags=[]
    for key,text in sections.items():
        for match in re.finditer(r'(?<![\d.])(-?\d+(?:\.\d+)?)\s*(?:%|％|个百分点)',text):
            val=float(match.group(1))
            if val in (0,100) or any(abs(val-x)<=.055 for x in numbers):continue
            flags.append({'section':key,'claim':match.group(0),'reason':'该数值无法在传入的证据包中直接匹配；请人工核对'} )
    terms=('已经通过出厂质检','已投入实际生产线','真实产品合格率','无缺陷产品全部通过','已经替代人工质检')
    for k,t in sections.items():
        for term in terms:
            if term in t:flags.append({'section':k,'claim':term,'reason':'超出模型预测数据的证据范围'})
    return {'schema':'screenqc.ai.audit.v1','mode':mode,'checked_sections':sorted(sections),
            'unverified_claims':flags,'status':'review_required' if flags else 'automatic_checks_passed',
            'disclaimer':'自动检查不能证明LLM分析正确；最终结论仍需人工核对。'}
