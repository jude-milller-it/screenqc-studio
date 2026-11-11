"""Run with `python -m unittest discover -s tests -v` (no actual weights required)."""
from __future__ import annotations
import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from core.metrics import confusion_update,score
from core.dataset import read_gt, read_ids
from core.artifacts import make_artifacts, regions_from_mask
from model_adapters.segformer_rules import classify
from visualization.report import make_report
from visualization.curves import create_training_report,create_oil_pr_report


class MetricsTests(unittest.TestCase):
    def test_micro_confusion_and_ignore(self):
        cm=np.zeros((4,4),dtype=np.int64)
        gt=np.array([[0,1,2,3,255]],dtype=np.uint8)
        pred=np.array([[0,1,1,3,2]],dtype=np.uint8)
        confusion_update(cm,gt,pred)
        self.assertEqual(int(cm.sum()),4)
        self.assertEqual(cm[2,1],1)
        self.assertAlmostEqual(score(cm)['per_class']['oil']['Precision_pct'],50.0)
        self.assertAlmostEqual(score(cm)['per_class']['oil']['Recall_pct'],100.0)
        self.assertAlmostEqual(score(cm)['per_class']['oil']['IoU_pct'],50.0)

    def test_wrong_shape_caught(self):
        cm=np.zeros((4,4),np.int64)
        with self.assertRaises(ValueError):
            confusion_update(cm,np.zeros((4,4),np.uint8),np.zeros((4,3),np.uint8))

    def test_segformer_threshold_compatibility(self):
        rng=np.random.default_rng(2026)
        prob=rng.random((31,27,4),dtype=np.float32)
        prob/=prob.sum(-1,keepdims=True)
        th={0:0,1:.4,2:.45,3:.4}
        baseline=prob.copy()
        for cls,t in th.items():
            if t:baseline[...,cls]=np.where(baseline[...,cls]<t,0.,baseline[...,cls])
        self.assertTrue(np.array_equal(classify(prob,th),baseline.argmax(-1)))
        # Non-oil classes only change where original oil was suppressed.
        opt=classify(prob,th,.30)
        original=classify(prob,th)
        self.assertTrue(np.all(opt[original!=1]==original[original!=1]))


class ArtifactsTests(unittest.TestCase):
    def test_boxes_json_and_mask_are_pixel_aligned(self):
        with tempfile.TemporaryDirectory() as temp:
            mask=np.zeros((24,30),np.uint8)
            mask[4:9,6:17]=1
            mask[14:16,12:29]=3
            rgb=Image.new('RGB',(30,24),(85,85,85))
            result,found=make_artifacts(rgb,mask,Path('src.jpg'),'test',Path(temp))
            self.assertEqual(len(found),2)
            self.assertEqual(found[0]['bbox_xywh'],[6,4,11,5])
            self.assertEqual(found[1]['pixel_area'],34)
            img=Image.open(result['mask_path'])
            self.assertTrue(np.array_equal(np.asarray(img),mask))
            self.assertTrue(Path(result['overlay_path']).is_file())
            self.assertTrue(Path(result['boxes_path']).is_file())

    def test_gray_palette_mappings(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'gt.png'
            arr=np.array([[0,38,75,113,255]],np.uint8)
            Image.fromarray(arr).save(p)
            self.assertEqual(read_gt(p,(1,5)).tolist(),[[0,1,2,3,255]])

    def test_split_is_strict(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'ImageSets'/'Segmentation';p.mkdir(parents=True)
            (p/'val.txt').write_text('a\na\n',encoding='utf-8')
            with self.assertRaises(ValueError):read_ids(Path(temp),'val')


class ReportTests(unittest.TestCase):
    def test_html_from_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            with (path/'model_comparison.csv').open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['model','mIoU_pct']);w.writeheader();w.writerow({'model':'SegFormer','mIoU_pct':77.7})
            with (path/'per_class_metrics.csv').open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['model','class','IoU_pct','Precision_pct','Recall_pct','F1_Dice_pct']);w.writeheader();w.writerow({'model':'SegFormer','class':'oil','IoU_pct':77,'Precision_pct':83,'Recall_pct':95,'F1_Dice_pct':87})
            result=make_report(path)
            text=result.read_text(encoding='utf-8')
            self.assertIn('77.70',text)
            self.assertIn('油污',text)


if __name__=='__main__':unittest.main()
