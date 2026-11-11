"""One loader and one prediction signature per model; preserves validated legacy inference."""
from __future__ import annotations
import gc
import sys
import io
import contextlib
from pathlib import Path
import numpy as np
from PIL import Image
from core.settings import ROOT, get_config, model_weights

AVAILABLE = ('deeplab_base','deeplab','segformer_raw','segformer_scale',
             'segformer_roi','segformer_conf','segformer','segformer_oil',
             'segformer_post_bad','segformer_edge_guard','fdsnet','fdsnet_tuned')
# Production = exactly three deployable pipelines. Research = five primary entries.
INDUSTRIAL_MODELS = ('deeplab','segformer_oil','fdsnet_tuned')
MAIN_MODELS = ('deeplab_base','deeplab','segformer_raw','segformer_oil','fdsnet_tuned')
STUDY_MODELS = ('segformer_raw','segformer_scale','segformer_roi',
                'segformer_conf','segformer','segformer_oil','segformer_post_bad','segformer_edge_guard')
DISPLAY = {
 'deeplab_base':'DeepLabV3+ · 基础训练权重',
 'deeplab':'DeepLabV3+ · 完整优化权重',
 'segformer_raw':'SegFormer · 最终训练权重 / 基础推理512',
 'segformer_scale':'SegFormer · 仅尺度适配',
 'segformer_roi':'SegFormer · 尺度+ROI',
 'segformer_conf':'SegFormer · 尺度+置信度',
 'segformer':'SegFormer · 尺度+ROI+置信度',
 'segformer_oil':'SegFormer · 所有推理优化（油污+0.30）',
 'segformer_post_bad':'SegFormer · 历史负优化（旧形态学/面积过滤）',
 'segformer_edge_guard':'SegFormer · 最终配置+屏幕边缘约束（新实验，待验证）',
 'fdsnet':'FDSNet · 原作者权重 / 原始推理',
 'fdsnet_tuned':'FDSNet · 原作者权重+污渍概率校正候选',
}


def _import_legacy():
    base = str(ROOT/'research')
    if base not in sys.path:
        sys.path.insert(0, base)


def _device_cuda() -> bool:
    import torch
    return torch.cuda.is_available()


class Predictor:
    def __init__(self, model_name: str, settings: dict | None = None):
        if model_name not in AVAILABLE:
            raise ValueError(f'模型名称应是: {AVAILABLE}')
        self.key = model_name
        self.settings = settings or get_config()
        self.weights = model_weights(model_name,self.settings)
        if not self.weights.is_file():
            raise FileNotFoundError(f'缺少{model_name}权重: {self.weights}；请先运行 migrate')
        self.cuda = _device_cuda()
        _import_legacy()
        if model_name in ('deeplab','deeplab_base'):
            from deeplab import DeeplabV3
            scheme=0 if model_name=='deeplab_base' else 3
            with contextlib.redirect_stdout(io.StringIO()):
                self.instance = DeeplabV3(num_classes=4, model_path=str(self.weights),
                    use_cbam=(scheme>=2),use_postprocess=False,cuda=self.cuda)
            print(f'[LOADED] Deeplab scheme={scheme}; input_shape={self.instance.input_shape}; '
                  f'CBAM={scheme>=2}; CUDA={self.cuda}; weights={self.weights}',flush=True)
        elif model_name.startswith('segformer'):
            from segformer_infer import SegFormerInference
            with contextlib.redirect_stdout(io.StringIO()):
                self.instance = SegFormerInference(num_classes=4,variant='mit_b2',
                    model_path=str(self.weights),cuda=self.cuda,
                    input_shape=[512,512] if model_name=='segformer_raw' else [640,1152],
                    fast_mode=True,use_roi_mask=model_name not in ('segformer_raw','segformer_scale','segformer_conf'),
                    use_postprocess=False)
            print(f'[LOADED] SegFormer MiT-B2; input_shape={self.instance.input_shape}; '
                  f'ROI={model_name not in ("segformer_raw","segformer_scale","segformer_conf")}; morphology={model_name=="segformer_post_bad"}; oil_delta='
                  f'{self.settings.get("segformer_oil_delta",0.30) if model_name=="segformer_oil" else 0.0}; '
                  f'CUDA={self.cuda}; weights={self.weights}',flush=True)
        else:
            fds_path=str(ROOT/'research'/'third_party'/'fdsnet')
            if fds_path not in sys.path: sys.path.insert(0,fds_path)
            from fdsnet_core.fdsnet import FDSNet
            import torch
            self.instance=FDSNet(num_classes=4,aux=True)
            state=torch.load(str(self.weights),map_location='cpu',weights_only=True)
            if isinstance(state,dict) and 'state_dict' in state:state=state['state_dict']
            self.instance.load_state_dict(state,strict=True)
            self.instance.to('cuda' if self.cuda else 'cpu').eval()
            print(f'[LOADED] FDSNet; tuned={model_name=="fdsnet_tuned"}; resize_scale=0.75; '
                  f'class_order=(background,oil,stain,scratch); CUDA={self.cuda}; '
                  f'weights={self.weights}',flush=True)

    def predict(self, rgb: Image.Image) -> np.ndarray:
        rgb=rgb.convert('RGB')
        if self.key in ('deeplab','deeplab_base'):
            return np.asarray(self.instance.get_miou_png(rgb),dtype=np.uint8)
        if self.key.startswith('segformer'):
            import torch
            img=np.asarray(rgb,dtype=np.uint8)
            raw_mode=self.key in ('segformer_raw','segformer_scale')
            use_roi=self.key not in ('segformer_raw','segformer_scale','segformer_conf')
            use_conf=self.key not in ('segformer_raw','segformer_scale','segformer_roi')
            roi=self.instance._detect_screen_roi(img) if use_roi else None
            with torch.inference_mode():
                probs=self.instance._fast_predict(img)
            if raw_mode:
                return np.argmax(probs,axis=-1).astype(np.uint8)
            if use_roi:
                probs=self.instance._apply_roi_and_edge(probs.copy(),roi[0],roi[1])
            thresholds=(self.instance.conf_thresholds_dark
                        if (roi[2] if roi else float(np.mean(img)))<self.instance.screen_brightness_split
                        else self.instance.conf_thresholds_light)
            from model_adapters.segformer_rules import classify
            oil_delta=float(self.settings.get('segformer_oil_delta',0.30)) if self.key in ('segformer_oil','segformer_edge_guard') else 0.0
            oil_margin=float(self.settings.get('segformer_oil_margin',0.0)) if self.key in ('segformer_oil','segformer_edge_guard') else 0.0
            if not use_conf:
                return np.argmax(probs,axis=-1).astype(np.uint8)
            result=classify(probs,thresholds,oil_delta,oil_margin)
            if self.key=='segformer_post_bad':
                return self.instance.industrial_postprocess(result,roi[1] if roi else None)
            if self.key=='segformer_edge_guard':
                from core.screen_guard import apply_guard
                result,_,_=apply_guard(rgb,result,16)
            return result
        # FDSNet verified 0,1,2,3 class order; 0.75 resize, align to 32.
        import torch
        import torch.nn.functional as F
        w=max(32,int(rgb.width*.75));h=max(32,int(rgb.height*.75))
        w=(w+31)//32*32;h=(h+31)//32*32
        arr=np.asarray(rgb.resize((w,h),Image.Resampling.BILINEAR),dtype=np.float32)/255.0
        arr=(arr-[.485,.456,.406])/[.229,.224,.225]
        ten=torch.from_numpy(arr.transpose(2,0,1).copy()).float().unsqueeze(0)
        ten=ten.to('cuda' if self.cuda else 'cpu')
        with torch.inference_mode():
            raw=self.instance(ten)[0]
            raw=F.interpolate(raw,size=(rgb.height,rgb.width),mode='bilinear',align_corners=True)
            if self.key=='fdsnet_tuned':
                # Historical exploratory stain_scale=1.10: +~0.04 pp mIoU on val,
                # NOT selected by development-only FDSNet search; not independent evidence.
                probs=raw.softmax(1)
                probs[:,2] *= float(self.settings.get('fdsnet_stain_scale',1.10))
                return probs.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            return raw.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

    def synchronize(self):
        if self.cuda:
            import torch
            torch.cuda.synchronize()

    def close(self):
        del self.instance
        gc.collect()
        if self.cuda:
            import torch
            torch.cuda.empty_cache()
