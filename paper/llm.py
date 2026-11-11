# -*- coding: utf-8 -*-
"""Structured LLM commentary. API key supplied by caller; caller may retrieve from Windows encrypted store."""
from __future__ import annotations
import json,re
from paper.evidence import research_evidence
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError

PROVIDERS={
 'DeepSeek':('https://api.deepseek.com','deepseek-flash'),
 'OpenAI':('https://api.openai.com/v1','gpt-4.1-mini'),
 '硅基流动 SiliconFlow':('https://api.siliconflow.cn/v1',''),
 '自定义 OpenAI兼容':('',''),
}
SECTIONS_PAPER=['protocol','overall','oil','stain','scratch','ablation','runtime','limits','conclusion']
SECTIONS_INDUSTRIAL=['overview','locations','safety']
BASE_PROMPT='''你是工业视觉研究论文的数据分析助理。分析必须严格基于已提供的结构化指标；缺失的指标只能标注未提供。
硬约束：
1. 严禁杜撰模型精度、训练轮数、架构内部贡献、生产数据、文献或引用。
2. 实验数据由Python确定性生成。你只能解释，不得更改数字或重新预测。
3. Precision（精确率）与Recall（召回率）分开解释；单位变化是“百分点”，不得混写%。
4. SegFormer raw / scale / base / oil 属于同一最终训练权重的不同推理配置，绝不能误称“不同训练阶段权重”。
5. 优化油污时说明对召回率的代价；如FDSNet总体领先，应正面说明并揭示作者权重与val重叠风险。
6. 不能把像素级召回率写成“产品检出率”；没有真实产线部署证据不得宣称上线。
7. 文字必须信息密度高，优先对照实测数值分析改进幅度与误差代价，不复述整张表，不编造图像观察。
8. 仅输出合法JSON对象，不用Markdown代码围栏。每段70至180字。改进过程、负优化及研究限制要具体；未提供的数据不能臆测。
9. 每个章节必须回答一个研究问题；先说结论，再引用具体配置及数值，最后说明代价和限制。绝不把相关性写为架构因果。
10. 优先引用数据包内controlled_pairs的受控增量，并区分【完整推理进展】和【最后油污阈值单项增量】。
11. 禁止模糊套话，如显著提升、具有巨大价值、达到工业级；如没有统计显著性检验则不能称统计显著。
'''
INDUSTRIAL_PROMPT='''你是视觉质检产线报告撰写员。输入只含模型批量预测，不含人工标注。不可宣称检测精度、通过率、漏检率、合格/报废；不可编造客户、设备或人工复核。报告应解释批次发现、输出坐标可供复核、风险与复核流程。必须输出合法JSON对象，且仅含overview、locations、safety三个字段。每字段70-180字，不得编造数值。只能引用传入的候选统计、重点复核列表和处理耗时；按优先级排序是人工审核队列，不可宣称真实缺陷严重程度。overview解释受影响图片数和候选分布；locations介绍为何先复核候选密集的图片，不重复罗列所有坐标；safety提出具体可执行的复核步骤及信息缺口。不得编造质检员、班次、产线、处置结果。'''

def prompt_text(structured:dict,mode:str='paper')->str:
    """Compact evidence contract: metrics, differences and known limits, no raw images."""
    if mode=='paper':
        needed=('mIoU_pct','defect_mIoU_pct','mPrecision_pct','mRecall_pct','mF1_pct','mean_pipeline_ms')
        cls=('IoU_pct','Precision_pct','Recall_pct','F1_Dice_pct','FP','FN')
        models=[]
        for m in structured.get('models',[]):
            metrics=m.get('summary',{})
            models.append({'id':m.get('id'),
              'overall':{k:round(v,3) if isinstance(v,(float,int)) else v for k,v in metrics.items() if k in needed},
              'classes':{c:{k:(round(v,3) if isinstance(v,(float,int)) and k not in ('FP','FN') else v)
                                for k,v in m.get('classes',{}).get(c,{}).items() if k in cls}
                         for c in ('oil','stain','scratch')}})
        evidence=structured.get('evidence') or research_evidence(structured)
        payload={'models':models,
                 'controlled_evidence':evidence.get('controlled_pairs',[]),
                 'main_rankings':evidence.get('primary_rankings',{}),
                 'experiment':{k:v for k,v in structured.get('experiment',{}).items() if k in ('split','images','dataset')},
                 'limitations':structured.get('caveats',[])[:6],
                 'rules':['Compare the 5 primary models separately from SegFormer ablations',
                          'Distinguish pixel recall from product detection rate',
                          'Explain oil precision gain and recall cost, cite the exact model variants',
                          'Do not claim image review: no images were sent',
                          'Include decision-relevant limitations and unresolved evidence']}
    else:
        payload={'source':structured.get('source'), 'images':structured.get('images'),
                 'regions':structured.get('regions'), 'model_names':structured.get('model_names',[]),
                 'region_counts':structured.get('region_counts',{}),
                 'predicted_pixels':structured.get('predicted_pixels',{}),
                 'mean_inference_ms':structured.get('mean_inference_ms'),
                 'distribution':structured.get('distribution',{}),
                 'review_queue':(structured.get('evidence') or {}).get('review_priority',[])[:6],
                 'candidate_image_rate_pct':(structured.get('evidence') or {}).get('candidate_image_rate_pct'),
                 'p95_processing_ms':(structured.get('evidence') or {}).get('p95_processing_ms'),
                 'limitations':['No manual ground truth; counts are predictions only',
                                'No image files sent to model; no visual claims are allowed',
                                'No pass/fail or model accuracy conclusions']}
    return ('请按结构化JSON返回。先识别最重要的差异，再解释证据及代价，'
            '避免重复表格；明确哪些结论没有数据支持。仅使用以下事实，不得杜撰。\n'
            +json.dumps(payload,ensure_ascii=False,separators=(',',':')))


def _url(base:str)->str:
    base=base.rstrip('/')
    if base.endswith('/chat/completions'):return base
    if not base.startswith('https://') and not (base.startswith('http://127.0.0.1') or base.startswith('http://localhost')):
        raise ValueError('API地址仅支持HTTPS或本机localhost')
    return base+'/chat/completions'

def parse_sections(reply:str,mode:str='paper')->dict:
    """Allow JSON or fenced JSON; reject unrelated prose, use safe deterministic fallback."""
    allowed=SECTIONS_PAPER if mode=='paper' else SECTIONS_INDUSTRIAL
    cleaned=reply.strip()
    if cleaned.startswith('```'):
        cleaned=re.sub(r'^```(?:json)?\s*','',cleaned,flags=re.I)
        cleaned=re.sub(r'\s*```$','',cleaned)
    try:
        raw=json.loads(cleaned)
    except json.JSONDecodeError:
        raise ValueError('LLM未按要求返回结构化JSON；原始数值和标准报告未改变。请更换兼容模型或重试') from None
    if not isinstance(raw,dict):raise ValueError('AI必须返回包含章节键的JSON对象')
    parts={}
    for k in allowed:
        item=raw.get(k)
        # Backwards compatible: section may be a plain paragraph or a structured evidence claim.
        if isinstance(item,str) and item.strip():parts[k]=item.strip()[:1200]
        elif isinstance(item,dict):
            claim=item.get('analysis') or item.get('text')
            if isinstance(claim,str) and claim.strip():parts[k]=claim.strip()[:1200]
    if not parts:raise ValueError('AI返回的JSON中没有可用的章节分析')
    return parts

def analyze(structured:dict,provider:str,base_url:str,model:str,api_key:str,timeout:int=110,mode:str='paper')->tuple[str,dict]:
    if not api_key.strip():raise ValueError('请输入API Key；可在设置页保存Windows用户加密密钥')
    if not model.strip():raise ValueError('请输入可用模型ID')
    if len(json.dumps(structured,ensure_ascii=False))>220000:raise ValueError('结构化数据过大，拒绝发送')
    keys=', '.join(SECTIONS_PAPER if mode=='paper' else SECTIONS_INDUSTRIAL)
    sys_prompt=BASE_PROMPT+f'\nJSON的键只能是：{keys}。每个章节的分析将插入对应图表附近。' if mode=='paper' else INDUSTRIAL_PROMPT
    body={'model':model.strip(),'messages':[{'role':'system','content':sys_prompt},{'role':'user','content':prompt_text(structured,mode)}],
          'temperature':0.15,'stream':False}
    req=Request(_url(base_url),data=json.dumps(body).encode('utf-8'),method='POST',headers={
        'Authorization':'Bearer '+api_key.strip(),'Content-Type':'application/json','User-Agent':'ScreenQCStudio/4.0'})
    try:
        with urlopen(req,timeout=timeout) as response:response_body=json.loads(response.read().decode('utf-8'))
    except HTTPError as e:raise RuntimeError(f'API返回HTTP {e.code}；检查模型ID、API地址和Key权限') from None
    except (URLError,TimeoutError) as e:raise RuntimeError('API无法连接或超时；检查网络和服务商配置') from None
    try:
        text=response_body['choices'][0]['message']['content']
        if not isinstance(text,str) or not text.strip():raise ValueError('空文本')
    except (KeyError,IndexError,TypeError,ValueError):raise RuntimeError('接口返回未包含可用消息') from None
    return text,{'provider':provider,'model_id':model.strip(),'usage':response_body.get('usage',{}),
                 'important':'辅助文字必须人工审核，原始指标与图表不会被覆盖'}
