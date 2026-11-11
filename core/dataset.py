"""MSD image and GT reading. Class mapping matches verified legacy evaluation."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

IMAGE_EXTS = ('.jpg','.jpeg','.png','.bmp','.tif','.tiff')


def image_by_stem(dataset: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        candidate = dataset/'JPEGImages'/f'{stem}{ext}'
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'MSD找不到原图: {stem}')


def read_ids(dataset: Path, split: str) -> list[str]:
    if split not in ('train', 'val', 'test', 'trainval'):
        raise ValueError('split仅支持train/val/test/trainval')
    split_file = dataset/'ImageSets'/'Segmentation'/f'{split}.txt'
    if not split_file.is_file():
        raise FileNotFoundError(f'找不到图像列表: {split_file}')
    ids = [x.strip() for x in split_file.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f'图像列表为空或包含重复编号: {split_file}')
    return ids


def read_gt(path: Path, hw: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.ndim != 2:
        raise ValueError(f'标签不是单通道: {path} {a.shape}')
    vals = set(map(int, np.unique(a)))
    if vals <= {0,1,2,3,255}:
        result = a.astype(np.uint8)
    elif vals <= {0,38,75,113,255}:
        result = np.zeros_like(a, dtype=np.uint8)
        for original, mapped in ((38,1),(75,2),(113,3),(255,255)):
            result[a == original] = mapped
    else:
        raise ValueError(f'标签值不识别（不会猜测映射）: {path} {sorted(vals)}')
    if result.shape != hw:
        raise ValueError(f'标签与图片尺寸不同: {path} {result.shape} vs {hw}')
    return result
