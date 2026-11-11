# -*- coding: utf-8 -*-
"""FDSNet inference compatible with the previously verified 81.4263% adapter."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fdsnet_core.fdsnet import FDSNet

PROJECT = Path(__file__).resolve().parent
DEFAULT_DATASET = Path(r"D:\AI\dataset\MobilePhoneScreenSurfaceDefectSegmentation\MSD")
WEIGHTS = PROJECT / "fdsnet__phone_voc_best_model.pth"

# Empirically verified on the user's evaluation: output indexes match the label PNG
# indices (0 background, 1 oil, 2 stain, 3 scratch). NO swap of 2 and 3.
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def dataset_root(path: Path) -> Path:
    path = path.expanduser()
    candidates = [path, path/'MSD', path/'PKU_MSSDD', path/'VOC2007']
    for r in candidates:
        if (r/'JPEGImages').is_dir() and (r/'SegmentationClass').is_dir():
            return r
    raise FileNotFoundError(f'找不到 MSD 数据集的 JPEGImages 和 SegmentationClass：{path}')


def list_split(root: Path, split: str) -> list[str]:
    if split not in ('val', 'test', 'train', 'trainval'):
        raise ValueError(f'不支持的数据集划分：{split}')
    path = root/'ImageSets'/'Segmentation'/f'{split}.txt'
    if not path.is_file():
        raise FileNotFoundError(f'找不到名单: {path}；不会自动用其他划分代替')
    ids = [x.strip() for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    if len(set(ids)) != len(ids):
        raise ValueError(f'存在重复图片ID：{path}')
    if not ids:
        raise ValueError(f'图片名单为空：{path}')
    return ids


def image_for(root: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        p = root/'JPEGImages'/f'{stem}{ext}'
        if p.is_file():
            return p
    raise FileNotFoundError(f'找不到原图：{stem}；JPEGImages 中应有对应文件')


def read_label(path: Path, expected_hw: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim != 2:
        raise ValueError(f'标签不是单通道：{path}, shape={arr.shape}')
    # MSD may contain native palette indices or grayscale values. Don't conflate.
    u = set(map(int, np.unique(arr)))
    if u <= {0,1,2,3,255}:
        mapped = arr.astype(np.uint8)
    elif u <= {0,38,75,113,255}:
        mapped = np.zeros(arr.shape, dtype=np.uint8)
        for a,b in ((38,1),(75,2),(113,3),(255,255)):
            mapped[arr == a] = b
    else:
        raise ValueError(f'无法识别标签值: {path}, {sorted(u)}')
    if mapped.shape != expected_hw:
        raise ValueError(f'标签尺寸不匹配：{path} {mapped.shape} != {expected_hw}')
    return mapped


def load_model(device: torch.device, weights: Path = WEIGHTS) -> FDSNet:
    if not weights.is_file():
        raise FileNotFoundError(f'权重文件不存在：{weights}')
    model = FDSNet(num_classes=4, aux=True)
    state = torch.load(str(weights), map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if not isinstance(state, dict):
        raise TypeError('权重格式不是state_dict')
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(f'模型加载成功：{len(state)}项权重严格匹配；设备={device}', flush=True)
    return model


def prepare(rgb: Image.Image, scale: float) -> torch.Tensor:
    if not 0.1 <= scale <= 1.5:
        raise ValueError('scale应在0.1到1.5之间')
    w = (max(32, int(rgb.width*scale))+31)//32*32
    h = (max(32, int(rgb.height*scale))+31)//32*32
    # EXACT SAME as previous verified adapter: bilinear, Imagenet normalization.
    resized = rgb.resize((w,h), Image.Resampling.BILINEAR)
    x = np.asarray(resized, dtype=np.float32)/255.0
    x = (x - [.485,.456,.406]) / [.229,.224,.225]
    return torch.from_numpy(x.transpose(2,0,1).copy()).float().unsqueeze(0)


@torch.inference_mode()
def predict_probs(model: FDSNet, rgb: Image.Image, scale: float, device: torch.device,
                  flip_tta: bool = False) -> tuple[np.ndarray, float]:
    """Returns CxHxW softmax scores at ORIGINAL image resolution; timed model+upsample.

    Conversion to probabilities is after logit interpolation, preserving baseline argmax.
    """
    tensor = prepare(rgb, scale).to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    logits = model(tensor)[0]
    logits = F.interpolate(logits, size=(rgb.height,rgb.width), mode='bilinear', align_corners=True)
    if flip_tta:
        xflip = tensor.flip(-1)
        fl = model(xflip)[0]
        fl = F.interpolate(fl, size=(rgb.height,rgb.width), mode='bilinear', align_corners=True)
        logits = (logits + fl.flip(-1)) * 0.5
    probs = logits.softmax(dim=1).squeeze(0).cpu().numpy()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter()-start)*1000
    return probs, elapsed


def save_artifacts(rgb: Image.Image, pred: np.ndarray, stem: str, out: Path) -> None:
    out.mkdir(exist_ok=True,parents=True)
    Image.fromarray(pred).save(out / f'{stem}_mask.png')
    colors = np.array([[0,0,0],[240,144,50],[78,165,230],[226,68,75]], dtype=np.uint8)
    base = np.asarray(rgb.convert('RGB')).copy()
    active = pred > 0
    base[active] = (base[active].astype(np.float32)*0.55 + colors[pred[active]]*0.45).astype(np.uint8)
    Image.fromarray(base).save(out/f'{stem}_overlay.jpg', quality=92)
