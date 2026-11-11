# -*- coding: utf-8 -*-
"""Tests require only NumPy/OpenCV; run with python -m unittest discover -s tests -v."""
import unittest
import numpy as np
from postprocess import Config, apply, remove_small_low_conf, bridge_scratch_only_into_background, variants
from metrics import update, scores


class PostprocessTests(unittest.TestCase):
    def test_baseline_does_not_change_prediction(self):
        p=np.full((4,12,20),.01,dtype=np.float32)
        p[0]=.9
        p[1,2:8,3:15]=.99
        p[0,2:8,3:15]=.005
        out=apply(p,Config('baseline'))
        np.testing.assert_array_equal(out, p.argmax(0))

    def test_never_delete_large_oil(self):
        p=np.full((4,500,500),.01,dtype=np.float32)
        p[0]=.8
        p[1,5:495,5:495]=.99
        p[0,5:495,5:495]=.005
        # All candidate postprocessing modes protect large confident oil regions.
        for c in variants(True):
            if c.roi_safe:
                continue  # ROI depends on real input, tested separately.
            o=apply(p,c)
            self.assertTrue(np.all(o[5:495,5:495] == 1), c.name)

    def test_scrap_tiny_low_conf_only(self):
        m=np.zeros((10,15),dtype=bool)
        m[2:4,2:4]=True
        m[6:8,9:11]=True
        probs=np.zeros_like(m,dtype=np.float32)
        probs[2:4,2:4]=.51
        probs[6:8,9:11]=.9
        r=remove_small_low_conf(m,probs,max_small_area=10,low_mean_threshold=.7)
        self.assertFalse(r[2,2]);self.assertTrue(r[6,9])

    def test_bridge_only_background_and_confident(self):
        m=np.zeros((7,16),dtype=np.uint8)
        m[3,3:5]=3;m[3,6:9]=3;m[3,5]=1 # oil must not be overwritten
        m[2,11:13]=3;m[2,14:16]=3
        probs=np.zeros_like(m,dtype=np.float32)
        probs[2,13]=.40
        r=bridge_scratch_only_into_background(m,probs,3,.2)
        self.assertEqual(r[3,5],1)
        self.assertEqual(r[2,13],3)

    def test_confidence_threshold(self):
        p=np.zeros((4,4,4),dtype=np.float32)
        p[0]=.3;p[3,1,1]=.4;p[0,1,1]=.2
        self.assertEqual(apply(p,Config('baseline'))[1,1],3)
        self.assertEqual(apply(p,Config('th',scratch_min_conf=.5))[1,1],0)

    def test_metrics_normalization(self):
        gt=np.array([[0,1,2,3]],dtype=np.uint8)
        pred=np.array([[0,1,3,3]],dtype=np.uint8)
        cm=np.zeros((4,4),np.int64)
        update(cm,gt,pred)
        s=scores(cm)
        self.assertEqual(s['IoU']['background'],100)
        self.assertEqual(s['IoU']['oil'],100)
        self.assertEqual(s['IoU']['stain'],0)
        self.assertEqual(s['IoU']['scratch'],50)


if __name__=='__main__':
    unittest.main()
