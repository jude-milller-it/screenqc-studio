"""
SegFormer inference wrapper — equivalent to deeplab.py for the DeepLabV3 pipeline.

Features:
  - Sliding window inference with overlapping tiles
  - Multi-scale + flip TTA
  - Class-specific post-processing (scratch closing, per-class area filtering)
  - Defect counting & ratio
"""
import copy
import time
import colorsys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.segformer import SegFormer
from utils.utils import cvtColor, preprocess_input, resize_image, show_config


class SegFormerInference(object):
    _defaults = {
        "model_path": 'logs_segformer_mit_b2/best_epoch_weights.pth',
        "num_classes": 4,
        "variant": "mit_b2",
        "input_shape": [512, 512],
        "stride_ratio": 0.6,
        "cuda": True,
        "mix_type": 0,
        # ---- screen ROI mask ----
        "use_roi_mask": True,        # auto-detect screen area, suppress outside
        "edge_penalty_dist": 30,     # wide edge zone for position-based rejection only
        "edge_penalty_factor": 1.0,  # NO confidence penalty — position rejection handles edges
        "edge_reject_ratio": 0.70,   # if >70% of CC in edge zone -> discard
        # ---- post-processing ----
        "use_postprocess": True,
        "min_area_map": {0: 0, 1: 15, 2: 1, 3: 3},
        "max_area_map": {1: 50000, 2: 3000, 3: 15000},  # loose: kill only bezel-sized blobs
        "scratch_close_kernel": (1, 3),
        "scratch_open_kernel": (2, 2),
        # ---- per-class confidence thresholds (auto-adjusted by screen brightness) ----
        # dark screen (higher contrast): oil=0.65 stain=0.65 scratch=0.63
        # white screen (lower contrast): oil=0.56 stain=0.60 scratch=0.58
        "conf_thresholds_dark":  {0: 0.0, 1: 0.55, 2: 0.55, 3: 0.55},
        "conf_thresholds_light": {0: 0.0, 1: 0.40, 2: 0.45, 3: 0.40},
        "screen_brightness_split": 80,  # gray < 80 = dark screen, else light
        # ---- inference mode ----
        "fast_mode": True,          # use simple resize (matches training eval)
        # ---- TTA ----
        "use_tta": True,
        "tta_scales": [1.0],
        "tta_flip": True,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # color map
        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
                           (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                           (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                           (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                           (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                           (0, 64, 128), (128, 64, 12)]
        else:
            hsv = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv))
            self.colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in self.colors]

        self.generate()
        show_config(**self._defaults)

    def generate(self):
        self.net = SegFormer(
            num_classes=self.num_classes,
            variant=self.variant,
            pretrained=False,
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device, weights_only=False)

        # handle DataParallel prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        self.net.load_state_dict(new_state_dict)
        self.net = self.net.eval()
        print(f'{self.model_path} model loaded.')

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    # ==================== Sliding Window Inference ====================
    def _sliding_window_predict(self, image_np):
        """
        Sliding window inference with Gaussian blending on overlap regions.
        Args:
            image_np: (H, W, 3) numpy array (RGB), 0-255
        Returns:
            probas: (H, W, num_classes) probability map
        """
        h, w = image_np.shape[:2]
        patch_h, patch_w = self.input_shape
        stride_h = int(patch_h * self.stride_ratio)
        stride_w = int(patch_w * self.stride_ratio)

        # generate patches (dedup edge-clamped positions)
        patches = []
        positions = []
        seen = set()
        for y in range(0, h, stride_h):
            for x in range(0, w, stride_w):
                # adjust for edge
                x1 = min(x, w - patch_w) if x + patch_w > w else x
                y1 = min(y, h - patch_h) if y + patch_h > h else y
                x1 = max(0, x1)
                y1 = max(0, y1)

                if (y1, x1) in seen:
                    continue
                seen.add((y1, x1))

                patch = image_np[y1:y1 + patch_h, x1:x1 + patch_w]
                # pad if needed
                if patch.shape[0] < patch_h or patch.shape[1] < patch_w:
                    pad_bottom = patch_h - patch.shape[0]
                    pad_right = patch_w - patch.shape[1]
                    patch = np.pad(patch, ((0, pad_bottom), (0, pad_right), (0, 0)),
                                   mode='constant', constant_values=128)

                # preprocess
                patch_resized, _, _ = resize_image(
                    Image.fromarray(patch.astype(np.uint8)),
                    (patch_w, patch_h)
                )
                patch_data = preprocess_input(np.array(patch_resized, np.float32))
                patch_data = np.transpose(patch_data, [2, 0, 1])
                patches.append(patch_data)
                positions.append((y1, x1))

        if len(patches) == 0:
            return np.zeros((h, w, self.num_classes), dtype=np.float32)

        # batch inference
        batch_data = np.stack(patches, axis=0)
        batch_tensor = torch.from_numpy(batch_data).float()
        if self.cuda:
            batch_tensor = batch_tensor.cuda()

        with torch.no_grad():
            outputs = self.net(batch_tensor)
            probas_batch = F.softmax(outputs, dim=1).cpu().numpy()

        # stitch with Gaussian blending
        probas_full = np.zeros((h, w, self.num_classes), dtype=np.float32)
        weight_full = np.zeros((h, w), dtype=np.float32)

        # Gaussian weight map for blending
        gy, gx = np.meshgrid(
            np.linspace(-1, 1, patch_h),
            np.linspace(-1, 1, patch_w),
            indexing='ij'
        )
        gaussian = np.exp(-(gx ** 2 + gy ** 2) / 0.5)  # sigma=0.7

        for idx, (y1, x1) in enumerate(positions):
            proba = np.transpose(probas_batch[idx], (1, 2, 0))  # H, W, C
            actual_h = min(patch_h, h - y1)
            actual_w = min(patch_w, w - x1)

            probas_full[y1:y1 + actual_h, x1:x1 + actual_w] += \
                proba[:actual_h, :actual_w] * gaussian[:actual_h, :actual_w, np.newaxis]
            weight_full[y1:y1 + actual_h, x1:x1 + actual_w] += \
                gaussian[:actual_h, :actual_w]

        # normalize
        valid = weight_full > 0
        probas_full[valid] /= weight_full[valid, np.newaxis]

        # fill unvisited areas with nearest neighbor
        if not np.all(valid):
            probas_full[~valid] = 1.0 / self.num_classes  # uniform if no data

        return probas_full

    # ==================== TTA ====================
    def _predict_with_tta(self, image_np):
        """Multi-scale + flip TTA."""
        probas_list = []
        orig_h, orig_w = image_np.shape[:2]

        for scale in self.tta_scales:
            if scale != 1.0:
                new_h = int(orig_h * scale)
                new_w = int(orig_w * scale)
                scaled = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                scaled = image_np.copy()

            # original
            proba = self._sliding_window_predict(scaled)
            if scale != 1.0:
                proba = cv2.resize(proba, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            probas_list.append(proba)

            # horizontal flip
            if self.tta_flip:
                flipped = scaled[:, ::-1, :].copy()
                proba_flip = self._sliding_window_predict(flipped)
                proba_flip = proba_flip[:, ::-1, :]  # flip back
                if scale != 1.0:
                    proba_flip = cv2.resize(proba_flip, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                probas_list.append(proba_flip)

        # average
        return np.mean(probas_list, axis=0)

    # ==================== Fast Predict (simple resize, matches training eval) ====================
    def _fast_predict(self, image_np):
        """Simple resize inference — same as EvalCallback during training. No sliding window."""
        h, w = image_np.shape[:2]
        image = Image.fromarray(image_np)
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(
            np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data).float()
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)[0]
            probas = F.softmax(outputs.permute(1, 2, 0), dim=-1).cpu().numpy()
            # crop off letterbox padding
            probas = probas[int((self.input_shape[0] - nh) // 2):int((self.input_shape[0] - nh) // 2 + nh),
                            int((self.input_shape[1] - nw) // 2):int((self.input_shape[1] - nw) // 2 + nw)]
            probas = cv2.resize(probas, (w, h), interpolation=cv2.INTER_LINEAR)
        return probas

    # ==================== Screen ROI Detection ====================
    def _detect_screen_roi(self, image_np):
        """Auto-detect screen region via Otsu + largest component.

        Returns:
            screen_mask: (H,W) float32, 1=screen, 0=outside
            edge_zone:   (H,W) float32, 1=within edge_penalty_dist of screen boundary
            screen_mean_gray: float, mean brightness of screen region
        """
        h, w = image_np.shape[:2]
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Otsu threshold → pick larger region as screen
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates = []
        for mask_cand in [binary, 255 - binary]:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
            closed = cv2.morphologyEx(mask_cand, cv2.MORPH_CLOSE, kernel, iterations=2)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
            if num_labels > 1:
                largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                area = stats[largest, cv2.CC_STAT_AREA]
                candidates.append((area, (labels == largest).astype(np.uint8) * 255))
        if not candidates:
            return (np.ones((h, w), dtype=np.float32),
                    np.zeros((h, w), dtype=np.float32), 0.0)

        _, screen_mask = max(candidates, key=lambda x: x[0])
        screen_f = screen_mask.astype(np.float32) / 255.0

        # mean screen brightness → adaptive thresholds
        screen_mean = float(gray[screen_mask > 0].mean()) if screen_mask.any() else 128.0

        # Distance transform: how far each screen pixel is from the nearest boundary
        dist = cv2.distanceTransform(screen_mask, cv2.DIST_L2, 5)
        edge_zone = ((dist > 0) & (dist <= self.edge_penalty_dist)).astype(np.float32)

        return screen_f, edge_zone, screen_mean

    def _apply_roi_and_edge(self, probas, screen_mask, edge_zone):
        """Outside screen → zero defect prob. Edge zone → confidence penalty.
        Interior → untouched. No erosion = edge defects survive.
        """
        interior = screen_mask - edge_zone  # 1 deep inside, 0 elsewhere
        outside = 1.0 - screen_mask          # 1 outside screen

        for cls_id in range(1, self.num_classes):
            probas[:, :, cls_id] = (
                probas[:, :, cls_id] * interior +                          # inside: full
                probas[:, :, cls_id] * edge_zone * self.edge_penalty_factor  # edge: reduced
            )
            # outside: already zero

        # Renormalize: background absorbs suppressed mass
        probas[:, :, 0] = 1.0 - probas[:, :, 1:].sum(axis=-1)
        probas = np.clip(probas, 0.0, 1.0)
        return probas

    # ==================== Post-processing ====================
    def industrial_postprocess(self, mask, edge_zone=None):
        """
        Class-specific post-processing:
          - Per-class opening to remove noise
          - Scratch: elongated closing to connect broken segments
          - Per-class min/max area + edge-position filtering
        """
        mask = mask.astype(np.uint8)
        h, w = mask.shape
        result = np.zeros_like(mask)

        for cls_id in range(1, self.num_classes):
            binary = (mask == cls_id).astype(np.uint8)

            if cls_id == 3:  # scratch
                kernel_open = cv2.getStructuringElement(
                    cv2.MORPH_RECT, self.scratch_open_kernel)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
                kernel_close = cv2.getStructuringElement(
                    cv2.MORPH_RECT, self.scratch_close_kernel)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
            elif cls_id == 2:  # stain
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            else:  # oil
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            # connected component filtering
            min_area = self.min_area_map.get(cls_id, 10)
            max_area = self.max_area_map.get(cls_id, 999999)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if not (min_area <= area <= max_area):
                    continue

                # ---- edge-position rejection ----
                if edge_zone is not None:
                    cc_mask = (labels == i)
                    edge_frac = float(edge_zone[cc_mask].sum()) / max(area, 1)
                    if edge_frac > self.edge_reject_ratio:
                        continue  # >90% in edge zone → bezel artifact, discard

                result[labels == i] = cls_id

        return result

    # ==================== Main Detection ====================
    def detect_image(self, image, count=True, name_classes=None):
        image = cvtColor(image)
        old_img = copy.deepcopy(image)
        original_h = np.array(image).shape[0]
        original_w = np.array(image).shape[1]

        image_np = np.array(image, dtype=np.uint8)

        # inference: fast (simple resize) or sliding window
        if self.fast_mode:
            probas = self._fast_predict(image_np)
        elif self.use_tta:
            probas = self._predict_with_tta(image_np)
        else:
            probas = self._sliding_window_predict(image_np)

        # ---- screen ROI suppression ----
        edge_zone = None
        screen_mean = 128.0  # default: assume mid-brightness
        if self.use_roi_mask:
            roi_screen, edge_zone, screen_mean = self._detect_screen_roi(image_np)
            probas = self._apply_roi_and_edge(probas, roi_screen, edge_zone)

        # adaptive confidence thresholds based on screen brightness
        if screen_mean < self.screen_brightness_split:
            thresholds = self.conf_thresholds_dark
        else:
            thresholds = self.conf_thresholds_light

        for cls_id, thresh in thresholds.items():
            if thresh > 0:
                probas[:, :, cls_id] = np.where(
                    probas[:, :, cls_id] < thresh, 0.0, probas[:, :, cls_id]
                )

        # argmax → class map
        pr = probas.argmax(axis=-1)

        # post-processing
        if self.use_postprocess:
            pr = self.industrial_postprocess(pr, edge_zone)

        # ---- smart voting (trust model class, don't reclassify by shape) ----
        defect_classes = [c for c in range(1, self.num_classes) if (pr == c).any()]
        if len(defect_classes) > 1:
            # Count CCs per class, pick the class with most CCs (stains = many dots)
            cc_counts = {}
            for cls_id in defect_classes:
                binary = (pr == cls_id).astype(np.uint8)
                nl, _, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                cc_counts[cls_id] = nl - 1  # subtract background label
            # Dominant by CC count, tiebreak by area
            max_cc = max(cc_counts.values())
            top = [c for c in defect_classes if cc_counts[c] == max_cc]
            if len(top) == 1:
                dominant = top[0]
            else:
                areas = {c: int((pr == c).sum()) for c in top}
                dominant = max(areas, key=areas.get)
            for c in defect_classes:
                if c != dominant:
                    pr[pr == c] = 0

        # ---- counting ----
        if count:
            classes_nums = np.zeros([self.num_classes])
            total_points = original_h * original_w
            print('-' * 63)
            print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num = np.sum(pr == i)
                ratio = num / total_points * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|" % (
                        str(name_classes[i]) if name_classes else str(i),
                        str(int(num)), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        # ---- visualization ----
        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])],
                                 [original_h, original_w, -1])
            image = Image.fromarray(np.uint8(seg_img))
            image = Image.blend(old_img, image, 0.5)
        elif self.mix_type == 1:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])],
                                 [original_h, original_w, -1])
            image = Image.fromarray(np.uint8(seg_img))
        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            image = Image.fromarray(np.uint8(seg_img))

        return image

    def get_miou_png(self, image):
        """Generate prediction mask for mIoU evaluation (minimal processing)."""
        image = cvtColor(image)
        original_h = np.array(image).shape[0]
        original_w = np.array(image).shape[1]

        image_np = np.array(image, dtype=np.uint8)

        if self.fast_mode:
            probas = self._fast_predict(image_np)
        elif self.use_tta:
            probas = self._predict_with_tta(image_np)
        else:
            probas = self._sliding_window_predict(image_np)

        # Only ROI mask for evaluation (kill bezel/desktop, nothing else)
        if self.use_roi_mask:
            roi_screen, _, _ = self._detect_screen_roi(image_np)
            for cls_id in range(1, self.num_classes):
                probas[:, :, cls_id] *= roi_screen
            probas[:, :, 0] = 1.0 - probas[:, :, 1:].sum(axis=-1)
            probas = np.clip(probas, 0.0, 1.0)

        # No post-processing for eval — raw argmax only
        pr = probas.argmax(axis=-1)
        return Image.fromarray(np.uint8(pr))

    def get_FPS(self, image, test_interval):
        """Measure FPS (no sliding window, single patch)."""
        image = cvtColor(image)
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(
            np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            t1 = time.time()
            for _ in range(test_interval):
                _ = self.net(images)
            t2 = time.time()

        return (t2 - t1) / test_interval
