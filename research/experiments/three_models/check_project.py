# -*- coding: utf-8 -*-
"""不导入Torch，不改动文件：检查三模型统一评测是否具备必要文件。"""
from pathlib import Path
from sys import exit
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(r"D:\AI\dataset\MobilePhoneScreenSurfaceDefectSegmentation\MSD")
ITEMS = [
    ("DeepLabV3源码", ROOT / "deeplab.py"),
    ("SegFormer源码", ROOT / "segformer_infer.py"),
    ("DeepLabV3方案3权重", ROOT / "logs_full_all" / "best_epoch_weights.pth"),
    ("SegFormer权重", ROOT / "logs_segformer_mit_b2" / "best_epoch_weights.pth"),
    ("FDSNet源码", ROOT / "third_party" / "fdsnet" / "fdsnet_core" / "fdsnet.py"),
    ("FDSNet公开权重", ROOT / "third_party" / "fdsnet" / "fdsnet__phone_voc_best_model.pth"),
    ("MSD验证列表", DATA / "ImageSets" / "Segmentation" / "val.txt"),
]
print("母项目：", ROOT)
ok = True
for name, path in ITEMS:
    yes = path.is_file()
    print(("[OK]" if yes else "[缺少]") + " " + name + ": " + str(path))
    ok &= yes
print("\n" + ("全部路径正常，可以运行统一评测。" if ok else "缺少文件；请看整合说明，不要随意移动原模型权重。"))
exit(0 if ok else 1)
