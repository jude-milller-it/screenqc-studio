"""Opt-in LCD display-surface guard, NEVER an unverified replacement for the trained model.

Detect the central dark LCD component with an Otsu proposal and strict geometry
checks. If evidence is inadequate, deliberately abstain instead of guessing an ROI.
Original-image coordinate system preserved. No hard-coded device corner positions.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path


def display_roi(rgb, inset: int = 16):
    import cv2
    arr=np.asarray(rgb.convert('RGB'), dtype=np.uint8)
    h,w=arr.shape[:2]
    gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
    cut,_=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    dark=np.uint8(gray<=cut)*255
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    dark=cv2.morphologyEx(dark,cv2.MORPH_CLOSE,kernel)
    n,labels,stats,_=cv2.connectedComponentsWithStats(dark,connectivity=8)
    possible=[]
    cx,cy=w//2,h//2
    for k in range(1,n):
        x,y,bw,bh,area=map(int,stats[k])
        relative=area/(w*h)
        # Avoid an entire dark tabletop/background being confused with the screen.
        if not (.18<=relative<=.86 and bw>=w*.52 and bh>=h*.38):continue
        center=labels[max(0,cy-h//20):min(h,cy+h//20),max(0,cx-w//20):min(w,cx+w//20)]
        if np.mean(center==k)<.25:continue
        possible.append((area,k,(x,y,bw,bh)))
    if not possible:
        return None,{'status':'abstain','reason':'screen display region could not be reliably located',
                     'otsu_threshold':round(float(cut),2)}
    _,k,bbox=max(possible)
    candidate=np.uint8(labels==k)*255
    # Fill bright oil/stain holes INSIDE the dark display: otherwise a true oil
    # spot would be erased by the ROI instead of segmented.
    contours,_=cv2.findContours(candidate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    filled=np.zeros_like(candidate)
    cv2.drawContours(filled,contours,-1,255,thickness=cv2.FILLED)
    inset=max(0,min(40,int(inset)))
    if inset:
        ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*inset+1,2*inset+1))
        filled=cv2.erode(filled,ker)
    coverage=float((filled>0).mean())
    if coverage<.12:
        return None,{'status':'abstain','reason':'effective ROI unexpectedly small', 'bbox_xywh':bbox}
    return filled>0,{'status':'applied','bbox_xywh':bbox,'inset_px':inset,
        'valid_display_fraction':round(coverage,5), 'otsu_threshold':round(float(cut),2)}


def apply_guard(image,pred,inset:int=16):
    mask,info=display_roi(image,inset)
    if mask is None:
        return pred,info,None
    cleaned=np.asarray(pred,dtype=np.uint8).copy()
    removed={str(k):int(np.count_nonzero((cleaned==k)&(~mask))) for k in (1,2,3)}
    cleaned[~mask]=0
    info['removed_pixels_by_class']=removed
    return cleaned,info,mask


def save_guard_preview(rgb,mask,path):
    import cv2
    from PIL import Image
    raw=np.asarray(rgb.convert('RGB'),dtype=np.uint8).copy()
    darkened=(raw.astype(np.float32)*.28).astype(np.uint8)
    darkened[mask]=raw[mask]
    contour=np.uint8(mask)*255
    borders,_=cv2.findContours(contour,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(darkened,borders,-1,(45,215,115),3)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(darkened).save(path)
