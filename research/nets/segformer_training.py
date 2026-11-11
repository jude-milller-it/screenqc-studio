"""
Advanced loss functions for industrial defect segmentation.

Includes:
  - Lovász-Softmax Loss (direct IoU optimization)
  - OHEM Loss Wrapper (100% defect pixels + top-N% background)
  - Boundary Loss (edge-aware)
  - Combined loss for SegFormer training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Lovász-Softmax Loss ====================
# Adapted from: https://github.com/bermanmaxim/LovaszSoftmax

def _lovasz_grad(gt_sorted):
    """Compute gradient of Lovász extension."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _flatten_probas(probas, labels, ignore=None):
    """Flatten predictions and labels in the batch."""
    if probas.dim() == 3:
        # already [C, N]
        return probas, labels.view(-1)

    b, c, h, w = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, c)  # (B*H*W, C)
    labels = labels.view(-1)
    if ignore is not None:
        valid = labels != ignore
        probas = probas[valid]
        labels = labels[valid]
    return probas, labels


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovász-Softmax loss (flat version).
    Args:
        probas: (N, C) softmax probabilities
        labels: (N,) ground truth labels
        classes: 'present' → per-class; 'all' → all classes including missing
    Returns:
        scalar loss
    """
    if probas.numel() == 0:
        return probas * 0.0

    c = probas.size(1)
    losses = []
    class_to_sum = list(range(c)) if classes == 'all' else labels.unique().tolist()

    for c_idx in class_to_sum:
        fg = (labels == c_idx).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        prob_c = probas[:, c_idx]
        errors = (fg - prob_c).detach()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))

    if len(losses) == 0:
        return probas.sum() * 0.0
    return torch.mean(torch.stack(losses))


def lovasz_softmax(logits, labels, classes='present', ignore=None):
    """
    Lovász-Softmax loss for semantic segmentation.

    Args:
        logits: (B, C, H, W) raw logits
        labels: (B, H, W) integer labels
        classes: 'present' or 'all'
        ignore: ignore index (pixels with this label are excluded)
    Returns:
        scalar loss
    """
    probas = F.softmax(logits, dim=1)
    probas_flat, labels_flat = _flatten_probas(probas, labels, ignore)
    return lovasz_softmax_flat(probas_flat, labels_flat, classes)


# ==================== OHEM-Enhanced Loss Wrapper ====================
class OHEMDefectLoss(nn.Module):
    """
    OHEM loss specifically for defect detection:
    - 100% of defect pixels (class > 0) participate in backprop
    - Top-K% of background pixels (class == 0) by loss
    This prevents background from drowning out rare defect pixels.
    """

    def __init__(self, bg_ratio=0.10, fg_ratio=0.70, ignore_index=255):
        super().__init__()
        self.bg_ratio = bg_ratio
        self.fg_ratio = fg_ratio  # sample 70% of defect pixels (not all)
        self.ignore_index = ignore_index

    def forward(self, criterion, logits, labels):
        if labels.dim() == 4 and labels.size(1) > 1:
            labels_idx = labels.argmax(dim=1)
        else:
            labels_idx = labels.squeeze(1) if labels.dim() == 4 else labels

        b, c, h, w = logits.shape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, c)
        labels_flat = labels_idx.reshape(-1)

        log_probs = F.log_softmax(logits_flat, dim=-1)
        per_pixel_loss = F.nll_loss(log_probs, labels_flat, reduction='none',
                                     ignore_index=self.ignore_index)

        fg_mask = (labels_flat > 0) & (labels_flat != self.ignore_index)
        bg_mask = (labels_flat == 0)

        # defect: random sample fg_ratio (70%) instead of 100%
        if fg_mask.any():
            fg_losses = per_pixel_loss[fg_mask]
            n_fg = fg_mask.sum().item()
            k_fg = max(1, int(n_fg * self.fg_ratio))
            idx = torch.randperm(n_fg, device=fg_losses.device)[:k_fg]
            fg_loss = fg_losses[idx].mean()
        else:
            fg_loss = 0.0

        # background: top-k by loss, ensure >= 1.2x defect count
        if bg_mask.any() and self.bg_ratio > 0:
            bg_losses = per_pixel_loss[bg_mask]
            n_bg_min = max(1, int(k_fg * 1.2)) if fg_mask.any() else 1
            k_bg = max(n_bg_min, int(len(bg_losses) * self.bg_ratio))
            k_bg = min(k_bg, len(bg_losses))
            topk_losses, _ = torch.topk(bg_losses, k=k_bg)
            bg_loss = topk_losses.mean()
        else:
            bg_loss = 0.0

        if isinstance(fg_loss, float) and fg_loss == 0.0:
            return bg_loss
        if isinstance(bg_loss, float) and bg_loss == 0.0:
            return fg_loss

        n_fg = max(fg_mask.sum().item(), 1)
        n_bg = max(bg_mask.sum().item(), 1)
        fg_weight = min(0.7, max(0.3, n_bg / (n_fg + n_bg)))
        bg_weight = 1.0 - fg_weight

        return fg_weight * fg_loss + bg_weight * bg_loss


# ==================== Boundary Loss ====================
class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss that penalizes edge misclassifications.
    Uses Sobel gradients to extract boundaries from both prediction and ground truth.
    """

    def __init__(self, theta=1.0, ignore_index=255):
        super().__init__()
        self.theta = theta
        self.ignore_index = ignore_index

        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def _sobel_edges(self, x):
        """Extract edges from a single-channel input using Sobel."""
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, H, W) → (B, 1, H, W)
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, logits, labels):
        """
        Args:
            logits: (B, C, H, W)
            labels: (B, H, W)
        """
        probas = F.softmax(logits, dim=1)  # B, C, H, W

        if labels.dim() == 4:
            labels = labels.argmax(dim=1)

        # one-hot labels — set ignore pixels to all-zero so they contribute nothing
        nc = logits.size(1)
        ignore_mask = (labels == self.ignore_index)
        labels_clean = labels.clamp(min=0, max=nc - 1)
        labels_onehot = F.one_hot(labels_clean, num_classes=nc).float()
        # zero out ignore pixels in one-hot
        if ignore_mask.any():
            labels_onehot[ignore_mask] = 0.0
        labels_onehot = labels_onehot.permute(0, 3, 1, 2)  # B, C, H, W

        b, c, h, w = probas.shape
        boundary_loss = 0.0
        for cls_idx in range(1, c):  # skip background
            pred_cls = probas[:, cls_idx:cls_idx + 1]      # B, 1, H, W
            gt_cls = labels_onehot[:, cls_idx:cls_idx + 1]  # B, 1, H, W

            pred_edge = self._sobel_edges(pred_cls)
            gt_edge = self._sobel_edges(gt_cls)

            # weighted BCE between predicted edges and GT edges
            diff = torch.abs(pred_edge - gt_edge)
            weight = 1.0 + self.theta * gt_edge  # emphasize true boundary regions
            boundary_loss += (weight * diff).mean()

        return boundary_loss / max(c - 1, 1)


# ==================== Focal Loss (used in OHEM) ====================
class FocalLoss(nn.Module):
    """Focal Loss with class weights."""

    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha  # tensor of shape (num_classes,)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        if labels.dim() == 4 and labels.size(1) > 1:
            labels = labels.argmax(dim=1)
        elif labels.dim() == 4:
            labels = labels.squeeze(1)

        b, c, h, w = logits.shape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, c)
        labels_flat = labels.reshape(-1)

        log_probs = F.log_softmax(logits_flat, dim=-1)
        probs = torch.exp(log_probs)

        # gather the log_probs of the target class
        log_probs_target = -F.nll_loss(log_probs, labels_flat, reduction='none',
                                        ignore_index=self.ignore_index)

        pt = torch.exp(log_probs_target)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            # clamp to valid range to avoid OOB when ignore_index >= num_classes
            labels_clamped = labels_flat.clamp(min=0, max=len(self.alpha) - 1)
            alpha_t = self.alpha[labels_clamped]
            focal_weight = focal_weight * alpha_t

        loss = focal_weight * (-log_probs_target)

        # mask out ignore pixels
        if self.ignore_index is not None:
            valid = labels_flat != self.ignore_index
            loss = loss[valid]

        return loss.mean()


# ==================== Dice Loss ====================
class DiceLoss(nn.Module):
    """Multi-class Dice Loss. Handles both integer and one-hot labels."""

    def __init__(self, smooth=1e-5, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        b, c, h, w = logits.shape

        # ---- determine label format (exact dim checks, not heuristics) ----
        if labels.dim() == 4 and labels.size(-1) == c + 1:
            # (B, H, W, C+1) one-hot — permute, exclude ignore channel
            labels_onehot = labels.permute(0, 3, 1, 2)[:, :-1]  # B, C, H, W
        elif labels.dim() == 4 and labels.size(1) == c + 1:
            # (B, C+1, H, W) one-hot — exclude ignore channel
            labels_onehot = labels[:, :-1]  # B, C, H, W
        elif labels.dim() == 4 and labels.size(1) == c:
            # (B, C, H, W) — already one-hot without ignore channel
            labels_onehot = labels.float()
        else:
            # integer (B, H, W) or (B, 1, H, W) → one-hot
            if labels.dim() == 4:
                labels = labels.squeeze(1)
            labels_onehot = F.one_hot(labels.clamp(min=0, max=c), num_classes=c + 1)
            labels_onehot = labels_onehot[..., :-1].permute(0, 3, 1, 2).float()

        probas = F.softmax(logits, dim=1)

        # flatten
        probas_flat = probas.view(b, c, -1)
        labels_flat = labels_onehot.view(b, c, -1)

        intersection = (probas_flat * labels_flat).sum(dim=2)
        union = probas_flat.sum(dim=2) + labels_flat.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# ==================== Combined Training Loss ====================
class SegFormerLoss(nn.Module):
    """
    Combined loss for SegFormer training:
      Total = w_dice * Dice + w_focal * Focal + w_lovasz * Lovász + w_boundary * Boundary

    Uses OHEM-style approach: all defect pixels + top-10% background for Focal component.

    Labels can be:
      - (B, H, W) integer class indices
      - (B, 1, H, W) — squeezed automatically
      - (B, H, W, C+1) one-hot (with ignore class at last channel) — argmax to indices
    """

    def __init__(self, num_classes, cls_weights=None,
                 w_dice=1.0, w_focal=2.0, w_lovasz=2.0, w_boundary=0.5,
                 gamma=2.0, ignore_index=None, ohem_bg_ratio=0.10):
        super().__init__()
        self.num_classes = num_classes
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.w_lovasz = w_lovasz
        self.w_boundary = w_boundary

        # ignore_index defaults to num_classes (matches one-hot encoding convention)
        self.ignore_index = num_classes if ignore_index is None else ignore_index

        if cls_weights is None:
            cls_weights = torch.ones(num_classes)
        self.register_buffer('cls_weights', cls_weights)

        self.dice_loss = DiceLoss(ignore_index=self.ignore_index)
        self.focal_loss = FocalLoss(alpha=cls_weights, gamma=gamma, ignore_index=self.ignore_index)
        self.ohem = OHEMDefectLoss(bg_ratio=ohem_bg_ratio, ignore_index=self.ignore_index)
        self.boundary_loss = BoundaryLoss(theta=1.0, ignore_index=self.ignore_index)

    def _to_index(self, labels):
        """Convert various label formats to (B, H, W) integer class indices.

        Uses exact dimension checks against num_classes to avoid ambiguity.
        """
        nc = self.num_classes if hasattr(self, 'num_classes') else 4
        if labels.dim() == 4 and labels.size(-1) == nc + 1:
            # (B, H, W, C+1) one-hot from dataloader → argmax to indices
            return labels.argmax(dim=-1)
        elif labels.dim() == 4 and labels.size(1) == nc + 1:
            # (B, C+1, H, W) one-hot → argmax along channel dim
            return labels.argmax(dim=1)
        elif labels.dim() == 4 and labels.size(1) == 1:
            # (B, 1, H, W) integer labels → squeeze channel dim
            return labels.squeeze(1)
        elif labels.dim() == 4 and labels.size(1) == nc:
            # (B, C, H, W) — already one-hot without ignore channel
            return labels.argmax(dim=1)
        elif labels.dim() == 4:
            # (B, unknown, H, W) — try argmax on channel dim
            return labels.argmax(dim=1)
        else:
            # (B, H, W) — already integer indices
            return labels

    def forward(self, logits, labels):
        labels_idx = self._to_index(labels)

        loss_total = 0.0

        # Dice Loss (uses original labels for one-hot handling)
        if self.w_dice > 0:
            loss_total += self.w_dice * self.dice_loss(logits, labels)

        # Focal Loss with OHEM (defect pixels 100% + top-10% background)
        if self.w_focal > 0:
            loss_total += self.w_focal * self.ohem(self.focal_loss, logits, labels_idx)

        # Lovász-Softmax Loss (direct IoU optimization)
        if self.w_lovasz > 0:
            loss_total += self.w_lovasz * lovasz_softmax(logits, labels_idx,
                                                          classes='present',
                                                          ignore=self.ignore_index)

        # Boundary Loss
        if self.w_boundary > 0:
            loss_total += self.w_boundary * self.boundary_loss(logits, labels_idx)

        return loss_total


# ==================== Helper: weights initialization ====================
def weights_init_segformer(model, init_type='trunc_normal'):
    """Initialize model weights. SegFormer uses trunc_normal internally, this is for the decoder."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
