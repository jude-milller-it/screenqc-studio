import unittest
from paper.evidence import inspection_evidence,research_evidence
from paper.ai_quality import audit_analysis

class EvidenceTests(unittest.TestCase):
    def test_counts_are_not_pixels(self):
        imgs=[{'source':'a.jpg','regions':'2','oil_pixels':'100','stain_pixels':'0','scratch_pixels':'2','time_ms':'15'},
              {'source':'b.jpg','regions':'0','oil_pixels':'0','stain_pixels':'0','scratch_pixels':'0','time_ms':'25'}]
        regions=[{'image':'a.jpg','class':'oil','area_px':'99'},{'image':'a.jpg','class':'scratch','area_px':'2'}]
        d=inspection_evidence(imgs,regions)
        self.assertEqual(d['regions'],2)
        self.assertEqual(d['predicted_pixels']['oil'],100)
        self.assertEqual(d['images_with_candidates'],1)
        self.assertEqual(d['candidate_image_rate_pct'],50)
        self.assertEqual(d['p95_processing_ms'],25)
        self.assertEqual(d['review_priority'][0]['image'],'a.jpg')
    def test_unverifiable_ai_percentage_warned(self):
        o=audit_analysis({'score':78.66},{'overall':'结果78.66%，另外999.12%则不可核。'})
        self.assertEqual(len(o['unverified_claims']),1)
    def test_research_pair_delta(self):
        def model(id,score):return {'id':id,'summary':{'mIoU_pct':score},'classes':{c:{k:score for k in ('IoU_pct','Precision_pct','Recall_pct','F1_Dice_pct')} for c in ('oil','stain','scratch')}}
        x=research_evidence({'models':[model('segformer_raw',71),model('segformer',76),model('segformer_oil',78)]})
        self.assertEqual([p for p in x['controlled_pairs'] if p['candidate']=='segformer_oil'][0]['mIoU_delta_pp'],2)

if __name__=='__main__':unittest.main()
