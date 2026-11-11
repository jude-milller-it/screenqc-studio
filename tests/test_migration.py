"""Regression tests for safe cross-version migration; no old project is edited."""
from pathlib import Path
import json
from scripts.migrate_all import Migrator, same_file


def test_copy_weights_config_and_reports_from_two_versions(tmp_path):
    root=tmp_path/'ScreenQC_Studio';root.mkdir()
    one=tmp_path/'DeepLabV3_Studio_v5_3';two=tmp_path/'DeepLabV3_Studio_v5_2'
    for old in (one,two):old.mkdir()
    (one/'checkpoints').mkdir();(one/'checkpoints'/'segformer_mit_b2.pth').write_bytes(b'real-user-model')
    (one/'config').mkdir()
    (one/'config'/'local.json').write_text(json.dumps({'dataset':str(tmp_path/'MSD'),
        'outputs':str(one/'outputs'), 'weights':{'segformer':str(one/'checkpoints'/'segformer_mit_b2.pth')}}))
    (one/'config'/'api_key.dpapi').write_bytes(b'opaque-encrypted-user-key')
    run=one/'outputs'/'inspection'/'20261107_083159'
    for folder in ('inspection_report','inspection_report_ai','boxes'):(run/folder).mkdir(parents=True)
    (run/'images.csv').write_text('source,boxes\nabc,old.png\n',encoding='utf-8')
    (run/'inspection_report'/'工业批量检测报告.pdf').write_bytes(b'STD PDF')
    (run/'inspection_report_ai'/'工业批量检测报告.pdf').write_bytes(b'AI PDF')
    (run/'boxes'/'old.png').write_bytes(b'PNG')
    rr=two/'outputs'/'evaluation'/'20261107_081941'
    (rr/'paper_package').mkdir(parents=True)
    (rr/'paper_package_ai').mkdir(parents=True)
    (rr/'model_comparison.csv').write_text('model,miou\nsegformer,78\n')
    (rr/'paper_package'/'论文实验分析.html').write_text('STD')
    (rr/'paper_package_ai'/'论文实验分析.html').write_text('AI')
    m=Migrator(root)
    m.run([('v5_3',one),('v5_2',two)],None)
    assert (root/'checkpoints'/'segformer_mit_b2.pth').read_bytes()==b'real-user-model'
    assert (root/'config'/'api_key.dpapi').read_bytes()==b'opaque-encrypted-user-key'
    cfg=json.loads((root/'config'/'local.json').read_text())
    assert cfg['outputs']=='outputs'
    assert cfg['weights']['segformer']=='checkpoints/segformer_mit_b2.pth'
    assert (root/'outputs'/'inspection'/'20261107_083159'/'inspection_report'/'工业批量检测报告.pdf').read_bytes()==b'STD PDF'
    assert (root/'outputs'/'inspection'/'20261107_083159'/'inspection_report_ai'/'工业批量检测报告.pdf').read_bytes()==b'AI PDF'
    assert (root/'outputs'/'evaluation'/'20261107_081941'/'paper_package'/'论文实验分析.html').read_text()=='STD'
    assert (root/'outputs'/'evaluation'/'20261107_081941'/'paper_package_ai'/'论文实验分析.html').read_text()=='AI'
    assert (one/'outputs'/'inspection'/'20261107_083159'/'inspection_report_ai'/'工业批量检测报告.pdf').exists()
    # Idempotent when migration command is clicked a second time.
    n=Migrator(root);n.run([('v5_3',one),('v5_2',two)],None)
    assert n.stats['copied']==0
    assert not (root/'outputs'/'_conflicts').exists()


def test_timestamp_collision_preserves_distinct_batches(tmp_path):
    root=tmp_path/'new';root.mkdir()
    a=tmp_path/'a';b=tmp_path/'b'
    for old,letter in ((a,'a'),(b,'b')):
        run=old/'outputs'/'inspection'/'same_time'
        run.mkdir(parents=True)
        (run/'images.csv').write_text('run\n'+letter+'\n')
    Migrator(root).run([('a',a),('b',b)],None)
    assert (root/'outputs'/'inspection'/'same_time'/'images.csv').read_text()=='run\na\n'
    assert (root/'outputs'/'inspection'/'same_time__b'/'images.csv').read_text()=='run\nb\n'


def test_newer_local_weights_are_not_replaced(tmp_path):
    root=tmp_path/'new'; old=tmp_path/'v5';old.mkdir()
    (root/'checkpoints').mkdir(parents=True)
    (old/'checkpoints').mkdir()
    (root/'checkpoints'/'fdsnet_phone.pth').write_bytes(b'new')
    (old/'checkpoints'/'fdsnet_phone.pth').write_bytes(b'old')
    Migrator(root).run([('v5',old)],None)
    assert (root/'checkpoints'/'fdsnet_phone.pth').read_bytes()==b'new'


def test_old_absolute_image_paths_resolve_to_copied_run(tmp_path):
    from app.desktop import _run_image_path
    run=tmp_path/'outputs'/'inspection'/'20261107_083159'
    (run/'boxes').mkdir(parents=True)
    sample=run/'boxes'/'Oil_0001_123.png';sample.write_bytes(b'PNG')
    old='D:\\AI\\projects\\DeepLabV3_Studio_v5_2\\outputs\\inspection\\20261107_083159\\boxes\\Oil_0001_123.png'
    assert _run_image_path(run,old)==sample
