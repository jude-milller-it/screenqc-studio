import json, tempfile, csv, unittest
from pathlib import Path
from paper.workflow import make_structured, build_package
from paper.llm import _url, prompt_text
from paper.audit import audit_dataset

class PaperTests(unittest.TestCase):
  def setUp(self):
    self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
    with (self.p/'model_comparison.csv').open('w',encoding='utf-8',newline='') as f:
      w=csv.DictWriter(f,fieldnames=['model','mIoU_pct','mPrecision_pct','mRecall_pct','mF1_pct']);w.writeheader();w.writerow({'model':'segformer','mIoU_pct':76.73,'mPrecision_pct':79.4,'mRecall_pct':95.47,'mF1_pct':86.2})
    with (self.p/'per_class_metrics.csv').open('w',encoding='utf-8',newline='') as f:
      w=csv.DictWriter(f,fieldnames=['model','class','IoU_pct','Precision_pct','Recall_pct','F1_Dice_pct']);w.writeheader()
      for cls in ('background','oil','stain','scratch'):
        w.writerow({'model':'segformer','class':cls,'IoU_pct':71,'Precision_pct':75,'Recall_pct':94,'F1_Dice_pct':83})
  def tearDown(self):self.tmp.cleanup()
  def test_structured_fidelity(self):
    d=make_structured(self.p);self.assertEqual(d['models'][0]['summary']['mIoU_pct'],76.73)
    self.assertIn('model_comparison.csv',d['source_sha256'])
    self.assertNotIn('api_key',json.dumps(d).lower())
  def test_all_artifacts_and_pdf(self):
    out=build_package(self.p)
    for f in ('structured_metrics.json','论文可编辑表格.docx','paper_report.html','figure_miou.png','figure_class_iou.png','黑玉兰_论文评测报告_标准.pdf','论文答辩评审流程_检查清单.md'):
      self.assertTrue((out/f).is_file(),f)
    from docx import Document
    self.assertEqual(len(Document(out/'论文可编辑表格.docx').tables),5)
    import fitz
    pdf=fitz.open(out/'黑玉兰_论文评测报告_标准.pdf')
    self.assertIn('76.73', '\n'.join(page.get_text() for page in pdf))
  def test_deepseek_url_and_prompt_boundaries(self):
    self.assertEqual(_url('https://api.deepseek.com'),'https://api.deepseek.com/chat/completions')
    with self.assertRaises(ValueError):_url('http://insecure.example')
    p=prompt_text(make_structured(self.p))
    self.assertIn('SegFormer',p)
  def test_audit_missing_dataset(self):
    d=audit_dataset(self.p/'missing');self.assertFalse(d['valid_structure'])

if __name__=='__main__':unittest.main()
