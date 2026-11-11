# -*- coding: utf-8 -*-
"""FDSNet-specific confidence/geometry postprocessing (no access to GT labels)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import cv2
import numpy as np


@dataclass(frozen=True)
class Config:
    name: str = 'baseline'
    # All classes are canonical: background=0, oil=1, stain=2, scratch=3
    # Class-dependent probability calibration: can recover stain/scratch missed as background.
    oil_score_scale: float = 1.0
    stain_score_scale: float = 1.0
    scratch_score_scale: float = 1.0
    oil_min_conf: float = 0.0
    stain_min_conf: float = 0.0
    scratch_min_conf: float = 0.0
    scratch_tiny_area: int = 0
    scratch_tiny_mean_conf: float = 0.0
    stain_tiny_area: int = 0
    stain_tiny_mean_conf: float = 0.0
    scratch_bridge_px: int = 0       # 3 / 5: horizontal + vertical closing, background only
    scratch_bridge_min_prob: float = 0.2
    roi_safe: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def variants(extended: bool = False) -> list[Config]:
    """Predetermined hypotheses; no artificial grid search on all held-out samples.

    Baseline ALWAYS preserved; oil is not subjected to max-area deletion.
    """
    v = [
        Config('baseline'),
        Config('scratch_conf_050', scratch_min_conf=.50),
        Config('scratch_conf_055', scratch_min_conf=.55),
        Config('scratch_cc_soft', scratch_tiny_area=10, scratch_tiny_mean_conf=.70),
        Config('stain_cc_soft', stain_tiny_area=3, stain_tiny_mean_conf=.70),
        Config('stain_score_105', stain_score_scale=1.05),
        Config('stain_score_110', stain_score_scale=1.10),
        Config('scratch_score_105', scratch_score_scale=1.05),
        Config('scratch_bridge3', scratch_bridge_px=3, scratch_bridge_min_prob=.20),
        Config('scratch_conf_cc', scratch_min_conf=.50, scratch_tiny_area=10, scratch_tiny_mean_conf=.70),
        Config('scratch_bridge_cc', scratch_bridge_px=3, scratch_bridge_min_prob=.20,
               scratch_tiny_area=10, scratch_tiny_mean_conf=.70),
        Config('both_class_clean', scratch_tiny_area=10, scratch_tiny_mean_conf=.70,
               stain_tiny_area=3, stain_tiny_mean_conf=.70),
        Config('roi_safe', roi_safe=True),
    ]
    if extended:
        v.extend([
            Config('scratch_cc_strict', scratch_tiny_area=20, scratch_tiny_mean_conf=.75),
            Config('scratch_bridge5', scratch_bridge_px=5, scratch_bridge_min_prob=.25),
            Config('stain_conf_050', stain_min_conf=.50),
            Config('scratch_score_110', scratch_score_scale=1.10),
            Config('oil_score_103', oil_score_scale=1.03),
            Config('stain105_scratch_clean', stain_score_scale=1.05, scratch_tiny_area=10, scratch_tiny_mean_conf=.70),
            Config('roi_plus_clean', roi_safe=True, scratch_tiny_area=10, scratch_tiny_mean_conf=.70,
                   stain_tiny_area=3, stain_tiny_mean_conf=.70),
        ])
    return v


def detect_safe_roi(rgb: np.ndarray) -> tuple[np.ndarray, bool]:
    """Conservative auto screen ROI. If uncertain, do NOT erase anything.

    Unlike the old Otsu 'largest foreground OR inverse' heuristic, this gate
    rejects off-center and border-filling candidates to limit false removals.
    """
    h,w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_sm = cv2.GaussianBlur(gray,(5,5),0)
    _, th = cv2.threshold(gray_sm,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    opts = []
    cy,cx=h//2,w//2
    for candidate in (th,cv2.bitwise_not(th)):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate,8)
        if n<=1: continue
        for i in np.argsort(stats[1:,cv2.CC_STAT_AREA])[-3:]+1:
            x,y,ww,hh,area = map(int,stats[i])
            cover=area/(h*w)
            rectfill=area/max(ww*hh,1)
            if not (.38<=cover<=.95 and rectfill>=.69 and ww/w>=.58 and hh/h>=.58):
                continue
            if not (x<=cx<x+ww and y<=cy<y+hh and labels[cy,cx]==i):
                continue
            if x==0 and y==0 and x+ww>=w and y+hh>=h:
                continue
            # Low risk: center-looking and covers meaningful portion of image.
            opts.append((rectfill*cover, (labels==i).astype(np.uint8)))
    if not opts:
        return np.ones((h,w), dtype=np.uint8), False
    mask=max(opts,key=lambda x:x[0])[1]
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    return mask, True


def remove_small_low_conf(binary: np.ndarray, conf_map: np.ndarray,
                          max_small_area: int, low_mean_threshold: float) -> np.ndarray:
    """Remove a small component ONLY if its mean class probability is low.

    Large oil regions are never filtered; tiny genuine defects can survive if confidence is high.
    """
    if max_small_area <= 0:
        return binary
    n,labels,stats,_=cv2.connectedComponentsWithStats(binary.astype(np.uint8),8)
    if n<=1:
        return binary
    area=stats[:,cv2.CC_STAT_AREA].astype(np.float64)
    # For probabilistic mean, use all pixels but only components != 0 contribute.
    counts=np.bincount(labels.ravel(), weights=conf_map.ravel().astype(np.float64), minlength=n)
    avg=np.divide(counts, np.maximum(area,1))
    discard=(area<max_small_area)&(avg<low_mean_threshold)
    discard[0]=False
    if not discard.any():
        return binary
    return binary & (~discard[labels])


def bridge_scratch_only_into_background(mask: np.ndarray, scratch_prob: np.ndarray,
                                         width: int, min_prob: float) -> np.ndarray:
    if width<3:
        return mask
    src=(mask==3).astype(np.uint8)
    # Apply directional closing to both orientations (CV2 kernel is width,height).
    horizontal=cv2.morphologyEx(src,cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT,(width,1)))
    vertical=cv2.morphologyEx(src,cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT,(1,width)))
    new=((horizontal|vertical)>0)&(mask==0)&(scratch_prob>=min_prob)
    if not new.any():
        return mask
    out=mask.copy()
    out[new]=3
    return out


def apply(probs: np.ndarray, cfg: Config, rgb: np.ndarray | None = None,
          roi_mask: np.ndarray | None = None) -> np.ndarray:
    """probs: (4,H,W) float32. Returns (H,W) uint8; NEVER reads GT mask."""
    if probs.ndim!=3 or probs.shape[0]!=4:
        raise ValueError(f'Expected 4xHxW class probability map, got {probs.shape}')
    if (cfg.oil_score_scale,cfg.stain_score_scale,cfg.scratch_score_scale)!=(1.0,1.0,1.0):
        adjusted=probs.copy()
        for cls,fac in ((1,cfg.oil_score_scale),(2,cfg.stain_score_scale),(3,cfg.scratch_score_scale)):
            adjusted[cls]*=fac
        pred=np.argmax(adjusted,axis=0).astype(np.uint8)
    else:
        pred=np.argmax(probs,axis=0).astype(np.uint8)
    if cfg.roi_safe:
        if roi_mask is None:
            if rgb is None:
                raise ValueError('ROI requested but RGB image unavailable')
            roi_mask,_ = detect_safe_roi(rgb)
        pred[roi_mask==0]=0
    for cls,t in ((1,cfg.oil_min_conf),(2,cfg.stain_min_conf),(3,cfg.scratch_min_conf)):
        if t>0:
            pred[(pred==cls)&(probs[cls]<t)]=0
    for cls,area,threshold in ((2,cfg.stain_tiny_area,cfg.stain_tiny_mean_conf),
                               (3,cfg.scratch_tiny_area,cfg.scratch_tiny_mean_conf)):
        if area>0:
            orig=pred==cls
            kept=remove_small_low_conf(orig,probs[cls],area,threshold)
            pred[orig&(~kept)]=0
    if cfg.scratch_bridge_px:
        pred=bridge_scratch_only_into_background(pred,probs[3],
                                                cfg.scratch_bridge_px,cfg.scratch_bridge_min_prob)
    return pred
