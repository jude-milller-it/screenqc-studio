"""Contracts for V5.2 report catalogs, history, and bilingual report controls."""
from pathlib import Path
from app.desktop import report_catalog
from core.settings import ROOT
from paper.report_v3 import _primary_models


def test_report_catalog_real_paths_only(tmp_path):
    standard=tmp_path/'inspection_report'
    standard.mkdir()
    (standard/'工业批量检测报告.pdf').write_bytes(b'%PDF-test')
    doc=report_catalog(tmp_path,'inspection')
    assert set(doc['standard']['files'])=={'PDF'}
    assert not doc['ai']['files']
    ai=tmp_path/'inspection_report_ai';ai.mkdir()
    (ai/'工业批量检测报告.docx').write_bytes(b'test')
    doc=report_catalog(tmp_path,'inspection')
    assert set(doc['ai']['files'])=={'Word'}


def test_paper_catalog_stays_separate(tmp_path):
    out=tmp_path/'paper_package';out.mkdir()
    (out/'论文实验分析.html').write_text('test',encoding='utf-8')
    doc=report_catalog(tmp_path,'paper')
    assert 'HTML' in doc['standard']['files']
    assert not doc['ai']['files']


def test_primary_models_exclude_intermediate_ablations():
    ids=['deeplab_base','deeplab','segformer_raw','segformer_scale','segformer_roi',
         'segformer_conf','segformer','segformer_oil','segformer_post_bad','fdsnet_tuned']
    models=[{'id':x} for x in ids]
    selected=_primary_models({'models':models})
    assert [x['id'] for x in selected]==['deeplab_base','deeplab','segformer_raw','segformer_oil','fdsnet_tuned']


def test_ui_has_two_report_groups_and_key_save():
    html=(ROOT/'app/web_assets/index.html').read_text(encoding='utf-8')
    js=(ROOT/'app/web_assets/app.js').read_text(encoding='utf-8')
    assert 'id="historyReports"' in html and 'id="saveApiKey"' in html
    assert 'function drawReportHub' in js
    assert '/api/report-catalog?' in js and '/api/credentials' in js
    assert 'Object.assign(strings.en,' in js
