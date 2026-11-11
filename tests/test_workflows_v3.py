"""V3 integration contracts: reports, data audit fallback and model stages."""
from __future__ import annotations
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.settings import ROOT
from model_adapters.registry import AVAILABLE, MAIN_MODELS


class V3WorkflowTests(unittest.TestCase):
    def test_stages_have_raw_and_optimized(self):
        self.assertEqual(len(AVAILABLE),len(set(AVAILABLE)))
        self.assertIn('segformer_raw',MAIN_MODELS)
        self.assertIn('deeplab_base',MAIN_MODELS)
        self.assertIn('deeplab',MAIN_MODELS)
        self.assertNotIn('segformer',MAIN_MODELS)
        self.assertIn('segformer_oil',MAIN_MODELS)
        self.assertIn('fdsnet_tuned',MAIN_MODELS)

    def test_empty_test_split_uses_val_only_for_size_analysis(self):
        from app.desktop import Tasks
        with tempfile.TemporaryDirectory() as tmp:
            ds=Path(tmp)/'MSD'
            (ds/'ImageSets'/'Segmentation').mkdir(parents=True)
            (ds/'SegmentationClass').mkdir()
            (ds/'ImageSets'/'Segmentation'/'test.txt').write_text('',encoding='utf-8')
            (ds/'ImageSets'/'Segmentation'/'val.txt').write_text('Oil_0001\n',encoding='utf-8')
            tasks=Tasks()
            fake=MagicMock();fake.stdout=['[PROGRESS] mocked\n'];fake.wait.return_value=0
            with patch('app.desktop.subprocess.Popen',return_value=fake) as mocked:
                tasks._worker('analyze',{'dataset':str(ds),'split':'test'})
            self.assertEqual(tasks.task['phase'],'done',tasks.task['lines'])
            self.assertIn('val.txt', '\n'.join(tasks.task['lines']))
            cmd=mocked.call_args.args[0]
            self.assertEqual(cmd[cmd.index('--split')+1],'val')

    def test_full_industrial_document_pack(self):
        from PIL import Image
        from industrial.report import build_inspection
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);(d/'boxes').mkdir()
            Image.new('RGB',(240,135),'#dde5ee').save(d/'boxes'/'sample.png')
            with (d/'images.csv').open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['source','model','regions','time_ms']);w.writeheader()
                w.writerow({'source':'sample.jpg','model':'segformer_oil','regions':1,'time_ms':'124.56'})
            with (d/'regions.csv').open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['image','class','x','y','width','height','area_px']);w.writeheader()
                w.writerow({'image':'sample.jpg','class':'scratch','x':17,'y':25,'width':81,'height':2,'area_px':150})
            result=build_inspection(d)
            for name in ('工业批量检测报告.pdf','工业批量检测报告.docx','工业批量检测报告.html','工业批量检测报告.md','inspection_data.json'):
                self.assertGreater((result/name).stat().st_size,100,name)
            html=(result/'工业批量检测报告.html').read_text(encoding='utf-8')
            self.assertIn('data:image/png;base64,',html)
            self.assertEqual(json.loads((result/'inspection_data.json').read_text(encoding='utf-8'))['images'],1)

    def test_research_report_uses_archived_real_csv(self):
        from paper.report_v3 import build_package
        from docx import Document
        run=ROOT/'research'/'experiments'/'three_models'/'unified_results_val'
        if not (run/'per_class_metrics.csv').exists():self.skipTest('legacy evidence not in this installation')
        with tempfile.TemporaryDirectory() as tmp:
            output=build_package(run,Path(tmp))
            for name in ('论文实验分析_黑玉兰.pdf','论文实验分析_可编辑.docx','论文实验分析.md','论文实验分析.html','structured_metrics.json','report_manifest.json'):
                self.assertGreater((output/name).stat().st_size,200,name)
            self.assertIn('data:image/png;base64,',(output/'论文实验分析.html').read_text(encoding='utf-8'))
            tables=Document(output/'论文实验分析_可编辑.docx').tables
            self.assertGreaterEqual(len(tables),5)
            doc=json.loads((output/'structured_metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(len(doc['models']),3)
            self.assertIn('background',(doc['models'][0]['classes']))


if __name__=='__main__':unittest.main()
