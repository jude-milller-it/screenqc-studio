"""Pure NumPy implementation of legacy SegFormer class-threshold decision.

No tensor, no network, easy to regression-test against legacy optimizer.
"""
from __future__ import annotations
import numpy as np


def classify(probs: np.ndarray, thresholds: dict, oil_delta: float=0.0,
             oil_margin: float=0.0) -> np.ndarray:
    if probs.ndim!=3 or probs.shape[-1]!=4:
        raise ValueError('SegFormer probability map must be HxWx4')
    prob=probs.copy()
    for cls in (2,3):
        threshold=float(thresholds.get(cls,0.0))
        if threshold>0:
            prob[...,cls]=np.where(prob[...,cls]<threshold,0.0,prob[...,cls])
    oil_t=float(thresholds.get(1,0.0))
    if oil_delta==0 and oil_margin==0:
        if oil_t>0: prob[...,1]=np.where(prob[...,1]<oil_t,0.0,prob[...,1])
        return prob.argmax(-1).astype(np.uint8)
    # Same order and tie-break as experiments/segformer_oil_optimization.py.
    oil=prob[...,1]
    bg,stain,scratch=prob[...,0],prob[...,2],prob[...,3]
    oil_possible=(oil>bg)&(oil>=stain)&(oil>=scratch)
    competitors=np.maximum(np.maximum(bg,stain),scratch)
    other=np.array([0,2,3],dtype=np.uint8)[np.argmax(prob[...,[0,2,3]],axis=-1)]
    threshold=float(np.clip(oil_t+oil_delta,0.0,1.0))
    allowed=oil_possible & (oil>=threshold)
    if oil_margin>0:
        allowed &= (oil-competitors>=oil_margin)
    return np.where(allowed,1,other).astype(np.uint8)
