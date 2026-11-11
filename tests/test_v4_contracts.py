"""ScreenQC V4 core invariants, no GPU checkpoint required."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch,MagicMock
from core.settings import get_config,model_weights
from model_adapters.registry import INDUSTRIAL_MODELS,MAIN_MODELS,STUDY_MODELS,AVAILABLE
from app.desktop import Tasks

class V4Contracts(unittest.TestCase):
    def test_three_industrial_pipelines(self):
        self.assertEqual(INDUSTRIAL_MODELS,('deeplab','segformer_oil','fdsnet_tuned'))
    def test_five_primary_paper_pipelines(self):
        self.assertEqual(MAIN_MODELS,('deeplab_base','deeplab','segformer_raw','segformer_oil','fdsnet_tuned'))
        self.assertIn('segformer_post_bad',STUDY_MODELS)
    def test_checkpoint_mapping_only_one_segformer(self):
        cfg=get_config()
        self.assertEqual(model_weights('segformer_raw',cfg),model_weights('segformer_oil',cfg))
        self.assertEqual(model_weights('fdsnet',cfg),model_weights('fdsnet_tuned',cfg))
        self.assertNotEqual(model_weights('deeplab_base',cfg),model_weights('deeplab',cfg))
    def test_windows_batch_json_manifest_instead_of_many_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);files=[]
            for n in range(150):
                f=d/f'{n:04d}.jpg';f.write_bytes(b'x');files.append(str(f))
            t=Tasks();proc=MagicMock();proc.stdout=['[PROGRESS] 1/150\n'];proc.wait.return_value=0
            cfg=get_config();cfg['outputs']=str(d/'out')
            with patch('app.desktop.get_config',return_value=cfg),patch('app.desktop.subprocess.Popen',return_value=proc) as popen:
                with patch('industrial.report.build_inspection') as report:
                    report.return_value=d/'fake';t._worker('predict',{'files':files,'model':'segformer_oil'})
            self.assertEqual(t.task['phase'],'done',t.task['lines'])
            cmd=popen.call_args.args[0]
            self.assertIn('--file-list',cmd)
            self.assertNotIn(str(files[0]),cmd)
            manifest=Path(cmd[cmd.index('--file-list')+1])
            self.assertEqual(len(json.loads(manifest.read_text(encoding='utf-8'))),150)
    def test_missing_checkpoint_preflight_before_model_loading(self):
        from evaluation.runner import evaluate
        with tempfile.TemporaryDirectory() as d:
            cfg=get_config();cfg['weights']['fdsnet']=str(Path(d)/'unavailable.pth')
            with self.assertRaises(FileNotFoundError) as cm:
                evaluate(Path(d),'val',['fdsnet_tuned'],Path(d)/'out',cfg)
            self.assertIn('fdsnet_tuned',str(cm.exception))
if __name__=='__main__':unittest.main()
