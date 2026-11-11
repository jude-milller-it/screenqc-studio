"""
Training health check — run while training or after a few epochs.

Usage:
    python diagnose.py                          # auto-find latest checkpoint
    python diagnose.py --weights logs_segformer_mit_b2/ep010-loss3.200-val_loss2.100.pth
    python diagnose.py --pretrained             # test if pretrained weights can be loaded
    python diagnose.py --inspect-weights model_data/segformer_b2_backbone_weights.pth
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def find_latest_checkpoint(log_dir="logs_segformer_mit_b2"):
    """Find latest epoch weights file in log dir."""
    if not os.path.isdir(log_dir):
        return None, None

    best = None
    best_epoch = -1
    for f in os.listdir(log_dir):
        if f.startswith("ep") and f.endswith(".pth"):
            try:
                epoch = int(f[2:5])
                if epoch > best_epoch:
                    best_epoch = epoch
                    best = os.path.join(log_dir, f)
            except ValueError:
                continue

    checkpoint = os.path.join(log_dir, "checkpoint.pth")
    return best, checkpoint


def check_loss_curve(log_dir):
    """Print loss/mIoU trends from log files."""
    print("\n" + "=" * 60)
    print("  Loss & mIoU Trend")
    print("=" * 60)

    for fname, label in [("epoch_loss.txt", "Train Loss"),
                          ("epoch_val_loss.txt", "Val Loss"),
                          ("epoch_miou.txt", "Train-Eval mIoU%")]:
        fpath = os.path.join(log_dir, fname)
        if not os.path.exists(fpath):
            print(f"  {label}: no data yet")
            continue
        with open(fpath) as f:
            values = [float(line.strip()) for line in f if line.strip()]
        if len(values) < 2:
            print(f"  {label}: {values}")
            continue

        # trend analysis
        recent = values[-5:] if len(values) >= 5 else values
        first_half = np.mean(values[:len(values) // 2]) if len(values) >= 4 else values[0]
        second_half = np.mean(values[len(values) // 2:])

        trend = "↓" if second_half < first_half else "↑"
        if "miou" in fname.lower():
            trend = "↑ (good)" if second_half > first_half else "↓ (bad!)"

        print(f"  {label}: {values[-1]:.4f} (latest)")
        print(f"    first half mean: {first_half:.4f} → second half: {second_half:.4f}  {trend}")
        if len(values) >= 3:
            print(f"    all: {[f'{v:.4f}' for v in values]}")

        # early warning signs
        if "miou" in fname.lower() and len(values) >= 10:
            if np.max(values[-5:]) < 5.0:
                print(f"    ⚠ mIoU still < 5% — model may not be learning. Check LR, pretrained weights.")
            elif np.max(values) < 10.0:
                print(f"    ⚠ mIoU < 10% — slow convergence. Pretrained weights highly recommended.")


def quick_inference(weights_path, variant="mit_b2", input_shape=(512, 512), num_classes=4):
    """Run inference on a few validation images, print per-class predictions."""
    print("\n" + "=" * 60)
    print("  Quick Inference Test")
    print("=" * 60)

    from nets.segformer import SegFormer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}, Variant: {variant}")

    model = SegFormer(num_classes=num_classes, variant=variant, pretrained=False)
    state = torch.load(weights_path, map_location=device)
    if 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    # Find a few validation images
    val_txt = "D:/AI/dataset/MobilePhoneScreenSurfaceDefectSegmentation/MSD/ImageSets/Segmentation/val.txt"
    if not os.path.exists(val_txt):
        print(f"  val.txt not found at {val_txt}")
        return

    with open(val_txt) as f:
        val_ids = [line.strip().split()[0] for line in f if line.strip()]

    jpg_dir = "D:/AI/dataset/MobilePhoneScreenSurfaceDefectSegmentation/MSD/JPEGImages"
    png_dir = "D:/AI/dataset/MobilePhoneScreenSurfaceDefectSegmentation/MSD/SegmentationClass"

    class_names = ["bg", "oil", "stain", "scratch"]

    for img_id in val_ids[:3]:
        jpg_path = os.path.join(jpg_dir, img_id + ".jpg")
        png_path = os.path.join(png_dir, img_id + ".png")
        if not os.path.exists(jpg_path):
            continue

        img = Image.open(jpg_path).convert("RGB")
        orig_w, orig_h = img.size
        img_resized = img.resize(input_shape, Image.BICUBIC)
        img_np = np.array(img_resized, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_tensor)
            out = F.interpolate(out, size=(orig_h, orig_w), mode='bilinear', align_corners=True)
            pred = out.argmax(dim=1).squeeze(0).cpu().numpy()

        # per-class pixel stats
        total = pred.size
        class_pcts = {}
        for cls_id, name in enumerate(class_names):
            pct = 100.0 * (pred == cls_id).sum() / total
            class_pcts[name] = pct

        # ground truth stats
        if os.path.exists(png_path):
            gt = np.array(Image.open(png_path))
            # map raw values
            gt_mapped = np.zeros_like(gt, dtype=np.uint8)
            gt_mapped[gt == 38] = 1
            gt_mapped[gt == 75] = 2
            gt_mapped[gt == 113] = 3
            gt_cls = {name: 100.0 * (gt_mapped == i).sum() / gt_mapped.size
                      for i, name in enumerate(class_names)}

            print(f"\n  [{img_id}] (GT: {orig_w}x{orig_h})")
            print(f"    {'Class':<10} {'Pred%':>8} {'GT%':>8} {'Diff':>8}")
            for name in class_names:
                diff = class_pcts[name] - gt_cls[name]
                flag = ""
                if name != "bg" and abs(diff) > 5:
                    flag = " ⚠"
                print(f"    {name:<10} {class_pcts[name]:7.2f}% {gt_cls[name]:7.2f}% {diff:+7.2f}%{flag}")
        else:
            print(f"\n  [{img_id}] Pred: " +
                  ", ".join(f"{n}={class_pcts[n]:.1f}%" for n in class_names))


def test_pretrained_weights(weight_path):
    """Test if pretrained weights can be loaded into the model."""
    print("\n" + "=" * 60)
    print("  Pretrained Weight Compatibility Test")
    print("=" * 60)

    if not os.path.exists(weight_path):
        print(f"  ⚠ File not found: {weight_path}")
        print(f"  Download from bubliiiing's Baidu Netdisk and place in model_data/")
        return

    from nets.segformer import SegFormer

    device = 'cpu'
    for variant in ['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5']:
        print(f"\n  Trying variant: {variant}...")
        try:
            model = SegFormer(num_classes=4, variant=variant, pretrained=False)
            state_dict = torch.load(weight_path, map_location=device)

            # unwrap common wrappers
            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            our_keys = set(model.encoder.state_dict().keys())
            src_keys = list(state_dict.keys())

            # direct match
            direct = sum(1 for k in src_keys if k in our_keys and
                        state_dict[k].shape == model.encoder.state_dict()[k].shape)
            print(f"    Direct match: {direct}/{len(our_keys)} keys")

            # try stripping prefixes
            for prefix in ['backbone.', 'encoder.', 'model.', 'module.']:
                stripped = {}
                for k, v in state_dict.items():
                    nk = k[len(prefix):] if k.startswith(prefix) else k
                    stripped[nk] = v
                hits = sum(1 for k in stripped if k in our_keys and
                          stripped[k].shape == model.encoder.state_dict()[k].shape)
                if hits > direct:
                    print(f"    After strip '{prefix}': {hits}/{len(our_keys)} keys — BETTER")

            # timm patterns
            import re
            timm_keys = [k for k in src_keys if re.match(r'(patch_embed|block)\d', k)]
            if timm_keys:
                print(f"    Has timm-format keys: {len(timm_keys)} (e.g. {timm_keys[0]})")

            # Show sample keys from both sides
            print(f"    Weight file sample keys:")
            for k in sorted(src_keys)[:5]:
                print(f"      {k}: {list(state_dict[k].shape)}")
            print(f"    Model encoder sample keys:")
            for k in sorted(our_keys)[:5]:
                print(f"      {k}")

        except Exception as e:
            print(f"    Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Training health check")
    parser.add_argument("--weights", type=str, default="",
                        help="Path to weights file (auto-detected if omitted)")
    parser.add_argument("--pretrained", action="store_true",
                        help="Test pretrained weight compatibility")
    parser.add_argument("--inspect-weights", type=str, default="",
                        help="Inspect a weight file's keys")
    args = parser.parse_args()

    if args.inspect_weights:
        test_pretrained_weights(args.inspect_weights)
        return

    if args.pretrained:
        for path in [
            "model_data/segformer_mit_b2_imagenet.pth",
            "model_data/segformer_b2_backbone_weights.pth",
        ]:
            if os.path.exists(path):
                test_pretrained_weights(path)
                break
        else:
            print("No pretrained weight file found in model_data/")
            print("Expected: model_data/segformer_mit_b2_imagenet.pth")
        return

    # ---- Training health check ----
    log_dir = "logs_segformer_mit_b2"
    best_weights, checkpoint = find_latest_checkpoint(log_dir)

    if best_weights is None and not args.weights:
        print("No training logs found. Run training first.")
        print("Or use: python diagnose.py --pretrained")
        return

    weights_path = args.weights or best_weights
    print(f"Using weights: {weights_path}")

    # 1. Loss trend
    loss_log_dir = None
    for subdir in os.listdir(log_dir):
        subpath = os.path.join(log_dir, subdir)
        if os.path.isdir(subpath) and subdir.startswith("loss_"):
            loss_log_dir = subpath
            break
    if loss_log_dir:
        check_loss_curve(loss_log_dir)
    else:
        print("No loss log subdirectory found.")

    # 2. Quick inference
    quick_inference(weights_path)


if __name__ == "__main__":
    main()
