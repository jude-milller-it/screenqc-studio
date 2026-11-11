"""Industrial-style artifacts: indexed masks, overlays, defect bounding boxes and contours."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from core.metrics import CLASSES, CHINESE

RGB_COLORS = ((0,0,0),(255,164,38),(66,162,248),(248,75,95))


def regions_from_mask(mask: np.ndarray, min_area: int = 1, polygon: bool = True) -> list[dict]:
    # cv2 is an existing runtime dependency in all three model projects.
    import cv2
    h, w = mask.shape
    regions = []
    for cls in range(1, 4):
        binary = (mask == cls).astype(np.uint8)
        n, labs, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
        for i in range(1, n):
            x,y,bw,bh,area = map(int, stats[i])
            if area < min_area:
                continue
            contour = []
            if polygon:
                patch = (labs[y:y+bh,x:x+bw] == i).astype(np.uint8)
                contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    peri = cv2.arcLength(largest, True)
                    simplified = cv2.approxPolyDP(largest, max(1.0, .004*peri), True)
                    contour = [[int(v[0][0])+x, int(v[0][1])+y] for v in simplified[:500]]
            regions.append({
                'class_id': cls, 'class_name': CLASSES[cls], 'class_zh': CHINESE[cls],
                'bbox_xywh': [x,y,bw,bh], 'bbox_xyxy': [x,y,x+bw,y+bh],
                'bbox_normalized_xywh': [round(x/w,6),round(y/h,6),round(bw/w,6),round(bh/h,6)],
                'pixel_area': area, 'image_coverage_pct': round(area/(w*h)*100,6),
                'center_xy': [round(float(centers[i,0]),2),round(float(centers[i,1]),2)],
                'polygon_xy': contour,
            })
    return sorted(regions, key=lambda r:(r['class_id'],-r['pixel_area']))


def make_artifacts(image: Image.Image, prediction: np.ndarray, source: Path, model_name: str,
                   output: Path, min_area: int = 1, draw_boxes: bool = True) -> tuple[dict,list[dict]]:
    if prediction.ndim != 2 or prediction.shape != (image.height,image.width):
        raise ValueError('预测分割图形状与原图不一致')
    if not np.isin(prediction, [0,1,2,3]).all():
        raise ValueError('模型输出类别不为0~3')
    output.mkdir(exist_ok=True,parents=True)
    key = f'{source.stem[:45]}_{hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:8]}'
    mask_dir, overlay_dir, boxes_dir, json_dir = (output/p for p in ('masks','overlays','boxes','json'))
    for d in (mask_dir, overlay_dir, boxes_dir, json_dir): d.mkdir(exist_ok=True,parents=True)
    mask_path = mask_dir/f'{key}.png'
    overlay_path = overlay_dir/f'{key}.png'
    boxes_path = boxes_dir/f'{key}.png'
    json_path = json_dir/f'{key}.json'
    Image.fromarray(prediction.astype(np.uint8), mode='L').save(mask_path)
    rgb = np.array(image.convert('RGB'), dtype=np.uint8)
    colored = np.array(RGB_COLORS,dtype=np.uint8)[prediction]
    active = prediction != 0
    overlay = rgb.copy()
    overlay[active] = (rgb[active].astype(np.float32)*.56 + colored[active].astype(np.float32)*.44).astype(np.uint8)
    Image.fromarray(overlay).save(overlay_path)
    regions = regions_from_mask(prediction, min_area=min_area, polygon=True)
    rendered = Image.fromarray(overlay).copy()
    draw = ImageDraw.Draw(rendered)
    for i,r in enumerate(regions,1):
        x,y,x2,y2=r['bbox_xyxy']
        draw.rectangle((x,y,x2-1,y2-1),outline=RGB_COLORS[r['class_id']],width=max(2,round(image.width/800)))
        # ASCII text avoids font availability assumptions on Chinese/Windows
        draw.text((x,max(0,y-13)),f"{i}:{r['class_name']} {r['pixel_area']}px",fill=RGB_COLORS[r['class_id']])
    rendered.save(boxes_path)
    per_class_pixels = {CLASSES[k]:int(np.count_nonzero(prediction==k)) for k in range(1,4)}
    record = {'source':str(source),'model':model_name,'width':image.width,'height':image.height,
              'coordinate_system':'original-image pixels, top-left origin, x right, y down',
              'class_order':list(CLASSES),'defect_regions':len(regions),
              'predicted_class_pixels':per_class_pixels,'regions':regions,
              'note':'Predictions are candidates, NOT verified production pass/fail decisions.',
              'mask_path':str(mask_path),'overlay_path':str(overlay_path),'boxes_path':str(boxes_path)}
    json_path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    return record,regions


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
