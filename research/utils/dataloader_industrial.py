"""
Industrial-strength DataLoader with:
  1. Copy-Paste augmentation: paste real defects onto clean areas
  2. Biased cropping: 50% defect-centered + 50% random
  3. Multi-scale training (scaling + rotation + color jitter)
  4. All label mappings preserved (38→1, 75→2, 113→3)
"""
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


LABEL_MAP = {38: 1, 75: 2, 113: 3}  # standard MSD mapping


def _apply_label_map(png_np):
    """Convert raw pixel values (38,75,113) to class indices (1,2,3)."""
    mapped = np.zeros_like(png_np, dtype=np.uint8)
    mapped[png_np == 38] = 1
    mapped[png_np == 75] = 2
    mapped[png_np == 113] = 3
    return mapped


class IndustrialDataset(Dataset):
    """
    Dataset with Copy-Paste augmentation and biased cropping for small defect detection.

    Args:
        annotation_lines: list of "image_id" strings
        input_shape: (h, w) crop size, e.g. (512, 512)
        num_classes: number of classes including background
        train: if True, apply data augmentation
        dataset_path: root path to the dataset
        copy_paste: enable copy-paste augmentation
        cp_num_defects: number of defects to paste per image (2-4 range)
        cp_num_scratches: number of scratches to paste per image (1-2 range)
        biased_crop_prob: probability of defect-centered cropping (vs random)
        scale_range: (min, max) scaling factor range for multi-scale
    """

    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path,
                 copy_paste=True, cp_num_defects=(2, 4), cp_num_scratches=(1, 2),
                 cp_prob=0.6, biased_crop_prob=0.5, scale_range=(0.5, 2.0)):
        super().__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path
        self.copy_paste = copy_paste
        self.cp_num_defects = cp_num_defects
        self.cp_num_scratches = cp_num_scratches
        self.cp_prob = cp_prob
        self.biased_crop_prob = biased_crop_prob
        self.scale_range = scale_range

        # pre-extract defect instances for copy-paste (training only)
        self.defect_instances = []  # list of dicts: {image_id, class_id, crop_img, crop_mask, bbox}
        if self.train and self.copy_paste:
            self._extract_defect_instances()

    def _extract_defect_instances(self):
        """Pre-scan training set and extract all defect instances for copy-paste."""
        print("[IndustrialDataset] Extracting defect instances for Copy-Paste...")
        for line in self.annotation_lines:
            name = line.split()[0]
            try:
                png_path = os.path.join(self.dataset_path, "SegmentationClass", name + ".png")
                jpg_path = os.path.join(self.dataset_path, "JPEGImages", name + ".jpg")
                if not os.path.exists(png_path) or not os.path.exists(jpg_path):
                    continue
                png = Image.open(png_path)
                if png.mode == "P":
                    png = png.convert("RGB").convert("L")
                elif png.mode != "L":
                    png = png.convert("L")
                label_np = np.array(png)
                label_mapped = _apply_label_map(label_np)

                jpg = Image.open(jpg_path)
                img_np = np.array(cvtColor(jpg))

                # find connected components for each defect class
                for cls_id in [1, 2, 3]:  # oil, stain, scratch
                    binary = (label_mapped == cls_id).astype(np.uint8)
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                    for i in range(1, num_labels):
                        area = stats[i, cv2.CC_STAT_AREA]
                        if area < 2:  # skip single-pixel noise
                            continue
                        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                                     stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                        # expand bbox slightly for context
                        x1 = max(0, x - 2)
                        y1 = max(0, y - 2)
                        x2 = min(img_np.shape[1], x + w + 2)
                        y2 = min(img_np.shape[0], y + h + 2)

                        instance_mask = (labels[y1:y2, x1:x2] == i).astype(np.uint8)
                        instance_img = img_np[y1:y2, x1:x2].copy()

                        self.defect_instances.append({
                            'class_id': cls_id,
                            'crop_img': instance_img,
                            'crop_mask': instance_mask,
                            'bbox': (x1, y1, x2 - x1, y2 - y1),  # (x, y, w, h)
                        })
            except Exception:
                continue
        print(f"[IndustrialDataset] Extracted {len(self.defect_instances)} defect instances "
              f"from {len(self.annotation_lines)} images.")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        # load image and label
        jpg = Image.open(os.path.join(self.dataset_path, "JPEGImages", name + ".jpg"))
        png = Image.open(os.path.join(self.dataset_path, "SegmentationClass", name + ".png"))
        if png.mode == "P":
            png = png.convert("RGB").convert("L")
        elif png.mode != "L":
            png = png.convert("L")

        png_np = np.array(png)
        label_np = _apply_label_map(png_np)
        img_np = np.array(cvtColor(jpg))

        # Copy-Paste augmentation (training only)
        if self.train and self.copy_paste and len(self.defect_instances) > 0 and self.rand() < self.cp_prob:
            img_np, label_np = self._apply_copy_paste(img_np, label_np)

        # Convert to PIL for augmentations
        image = Image.fromarray(img_np)
        label = Image.fromarray(label_np.astype(np.uint8))

        # Multi-scale random augmentation or deterministic resize
        image, label = self.get_random_data(image, label, self.input_shape, random=self.train)

        # Final preprocessing
        image = np.transpose(preprocess_input(np.array(image, np.float64)), [2, 0, 1])
        label = np.array(label)

        # handle out-of-range labels
        label[label >= self.num_classes] = self.num_classes

        # one-hot encoding
        seg_labels = np.eye(self.num_classes + 1)[label.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return image, label, seg_labels

    def _apply_copy_paste(self, img_np, label_np):
        """Paste real defects from other images onto the current image."""
        h, w = img_np.shape[:2]

        # determine how many instances to paste
        n_defects = np.random.randint(self.cp_num_defects[0], self.cp_num_defects[1] + 1)
        n_scratches = np.random.randint(self.cp_num_scratches[0], self.cp_num_scratches[1] + 1)

        # separate instances by type
        defect_pool = [d for d in self.defect_instances if d['class_id'] in (1, 2)]  # oil, stain
        scratch_pool = [d for d in self.defect_instances if d['class_id'] == 3]       # scratch

        pasted = 0
        for pool, count in [(defect_pool, n_defects), (scratch_pool, n_scratches)]:
            if len(pool) == 0:
                continue
            for _ in range(count):
                instance = pool[np.random.randint(0, len(pool))]
                inst_img = instance['crop_img']
                inst_mask = instance['crop_mask']
                inst_h, inst_w = inst_mask.shape

                # random scale (0.7 ~ 1.3)
                scale = np.random.uniform(0.7, 1.3)
                new_w = max(2, int(inst_w * scale))
                new_h = max(2, int(inst_h * scale))
                inst_img_scaled = cv2.resize(inst_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                inst_mask_scaled = cv2.resize(inst_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                # random rotation
                if np.random.rand() < 0.5:
                    angle = np.random.randint(-30, 31)
                    M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
                    inst_img_scaled = cv2.warpAffine(inst_img_scaled, M, (new_w, new_h),
                                                     flags=cv2.INTER_LINEAR,
                                                     borderValue=(128, 128, 128))
                    inst_mask_scaled = cv2.warpAffine(inst_mask_scaled, M, (new_w, new_h),
                                                      flags=cv2.INTER_NEAREST,
                                                      borderValue=0)

                # random position (avoid overlapping existing defects)
                max_attempts = 10
                for attempt in range(max_attempts):
                    px = np.random.randint(0, max(1, w - new_w))
                    py = np.random.randint(0, max(1, h - new_h))

                    # check if position is clean (no existing defects in paste area)
                    existing = label_np[py:py + new_h, px:px + new_w]
                    if np.sum(existing > 0) < 5:  # nearly clean
                        break

                # paste with soft edge blending for natural look
                mask_soft = cv2.GaussianBlur(inst_mask_scaled.astype(np.float32), (3, 3), 0.6)
                mask_3ch = np.stack([mask_soft] * 3, axis=-1)
                mask_3ch = np.clip(mask_3ch, 0, 1)
                # brightness variation
                brightness = np.random.uniform(0.85, 1.15)
                inst_img_scaled = np.clip(inst_img_scaled * brightness, 0, 255).astype(np.uint8)

                roi = img_np[py:py + new_h, px:px + new_w]
                blended = (roi * (1 - mask_3ch * 0.9) + inst_img_scaled * mask_3ch * 0.9).astype(np.uint8)
                img_np[py:py + new_h, px:px + new_w] = blended

                # paste label (threshold softened mask)
                label_np[py:py + new_h, px:px + new_w][mask_soft > 0.3] = instance['class_id']
                pasted += 1

        return img_np, label_np

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=0.3, hue=0.1, sat=0.7, val=0.3, random=True):
        """Augmentation pipeline with biased cropping."""
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))

        iw, ih = image.size
        h, w = input_shape

        if not random:
            # deterministic resize with letterbox
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_label

        # ---- random scaling ----
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(self.scale_range[0], self.scale_range[1])
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        # ---- flip ----
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # ---- biased cropping via paste-offset (unified: handles all 4 cases) ----
        label_arr = np.array(label)
        if self.rand() < self.biased_crop_prob and np.any(label_arr > 0):
            # defect-centered: paste source so defect lands at canvas center
            defect_ys, defect_xs = np.where(label_arr > 0)
            center_idx = np.random.randint(0, len(defect_ys))
            cy, cx = defect_ys[center_idx], defect_xs[center_idx]
            # random jitter
            cx += int(self.rand(-w // 4, w // 4))
            cy += int(self.rand(-h // 4, h // 4))
            x_off = w // 2 - cx
            y_off = h // 2 - cy
        else:
            # random offset
            if nw > w:
                x_off = -int(self.rand(0, nw - w))
            else:
                x_off = (w - nw) // 2
            if nh > h:
                y_off = -int(self.rand(0, nh - h))
            else:
                y_off = (h - nh) // 2

        # clamp offsets so canvas is filled (no blank edges outside image bounds)
        if nw > w:
            x_off = max(-(nw - w), min(0, x_off))
        else:
            x_off = max(0, min(w - nw, x_off))
        if nh > h:
            y_off = max(-(nh - h), min(0, y_off))
        else:
            y_off = max(0, min(h - nh, y_off))

        # paste source image onto canvas (PIL handles clipping naturally)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (x_off, y_off))
        new_label.paste(label, (x_off, y_off))
        image = new_image
        label = new_label

        image_data = np.array(image, np.uint8)

        # ---- gaussian blur ----
        blur = self.rand() < 0.25
        if blur:
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        # ---- rotation ----
        rotate = self.rand() < 0.25
        if rotate:
            center = (w // 2, h // 2)
            rotation = np.random.randint(-10, 11)
            M = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC,
                                        borderValue=(128, 128, 128))
            label = cv2.warpAffine(np.array(label, np.uint8), M, (w, h),
                                   flags=cv2.INTER_NEAREST, borderValue=0)

        # ---- color jitter (HSV) ----
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue_ch, sat_ch, val_ch = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        image_data = cv2.merge((cv2.LUT(hue_ch, lut_hue),
                                cv2.LUT(sat_ch, lut_sat),
                                cv2.LUT(val_ch, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label


def industrial_dataset_collate(batch):
    """Collate function for DataLoader."""
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels
