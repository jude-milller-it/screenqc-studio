"""
mIoU evaluation for SegFormer models.

Usage:
    python get_miou_segformer.py

The script:
  1. Loads the trained SegFormer model
  2. Runs sliding-window+TTA inference on the validation set
  3. Computes per-class IoU, mIoU, mPA, and confusion matrix
"""
import os
from PIL import Image
from tqdm import tqdm

from segformer_infer import SegFormerInference
from utils.utils_metrics import compute_mIoU, show_results


if __name__ == "__main__":
    # ====================== 配置 ======================
    variant = "mit_b2"       # 与训练一致
    save_dir = f"logs_segformer_{variant}"
    weight_path = os.path.join(save_dir, "best_epoch_weights.pth")

    # fallback
    if not os.path.exists(weight_path):
        weight_path = os.path.join(save_dir, "last_epoch_weights.pth")
        print(f"[WARNING] best_epoch_weights not found, using {weight_path}")

    # ---- mIoU mode ----
    # 0 = full pipeline (predict + compute)
    # 1 = only predict
    # 2 = only compute (assumes predictions already exist)
    miou_mode = 0
    num_classes = 4
    name_classes = ["background", "oil", "stain", "scratch"]
    VOCdevkit_path = 'D:/AI/dataset/MobilePhoneScreenSurfaceDefectSegmentation/MSD'

    # ---- instantiate model ----
    segformer = SegFormerInference(
        num_classes=num_classes,
        variant=variant,
        model_path=weight_path,
        input_shape=[512, 512],
        stride_ratio=0.75,
        use_postprocess=False,  # raw output for fair evaluation
        use_tta=True,
        cuda=True,
    )

    # ---- load validation set ----
    val_txt_path = os.path.join(VOCdevkit_path, "ImageSets/Segmentation/val.txt")
    if not os.path.exists(val_txt_path):
        # try test set as validation
        val_txt_path = os.path.join(VOCdevkit_path, "ImageSets/Segmentation/test.txt")
        if not os.path.exists(val_txt_path):
            raise FileNotFoundError(f"Validation file not found at {val_txt_path}")

    image_ids = open(val_txt_path, 'r').read().splitlines()
    gt_dir = os.path.join(VOCdevkit_path, "SegmentationClass/")
    miou_out_path = f"miou_out_segformer_{variant}"
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    # ---- predict ----
    if miou_mode == 0 or miou_mode == 1:
        os.makedirs(pred_dir, exist_ok=True)
        print(f"Loading weights: {weight_path}")
        print(f"Generating predictions for {len(image_ids)} validation images...")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "JPEGImages", image_id + ".jpg")
            if not os.path.exists(image_path):
                print(f"  [SKIP] {image_path} not found")
                continue
            image = Image.open(image_path)
            pred = segformer.get_miou_png(image)
            pred.save(os.path.join(pred_dir, image_id + ".png"))
        print("Prediction done.")

    # ---- compute mIoU ----
    if miou_mode == 0 or miou_mode == 2:
        print("Computing mIoU...")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(
            gt_dir, pred_dir, image_ids, num_classes, name_classes
        )
        print("mIoU computation done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
