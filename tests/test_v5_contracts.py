"""Regression checks for V5 work queue, pixel-vs-region accounting and HTML controls."""
from pathlib import Path
import csv,json
from PIL import Image
from app.desktop import TaskHub
from industrial.report import build_inspection
from core.settings import ROOT


def test_taskhub_has_independent_job_ids():
    hub=TaskHub()
    a=hub._new('predict')[0]
    b=hub._new('evaluate')[0]
    c=hub._new('ai')[0]
    assert (a,b,c)==(1,2,3)
    assert len(hub.status()['tasks'])==3


def test_inspection_reports_separate_pixel_counts_from_regions(tmp_path):
    img=tmp_path/'boxes'/'oil.png';img.parent.mkdir()
    Image.new('RGB',(90,60),'white').save(img)
    images=[{'source':'oil.png','model':'segformer_oil','time_ms':100,'regions':1,
         'oil_pixels':1234,'stain_pixels':0,'scratch_pixels':0,'boxes':str(img)}]
    regions=[{'image':'oil.png','class':'oil','x':4,'y':4,'width':20,'height':20,'area_px':1234}]
    for filename,rows in [('images.csv',images),('regions.csv',regions)]:
        with (tmp_path/filename).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    build_inspection(tmp_path)
    build_inspection(tmp_path,ai_text={'overview':'模拟AI讨论'})
    standard=json.loads((tmp_path/'inspection_report'/'inspection_data.json').read_text())
    assert standard['predicted_pixels']['oil']==1234
    assert standard['region_counts']['oil']==1
    assert (tmp_path/'inspection_report'/'工业批量检测报告.pdf').is_file()
    assert (tmp_path/'inspection_report_ai'/'工业批量检测报告.pdf').is_file()


def test_ui_history_and_multitask_controls_exist():
    html=(ROOT/'app/web_assets/index.html').read_text(encoding='utf-8')
    js=(ROOT/'app/web_assets/app.js').read_text(encoding='utf-8')
    assert all(x in html for x in ('id="taskDock"','id="inspectionHistoryPanel"','id="historyClass"'))
    assert 'historyGallery' in js and '/api/inspection-images?' in js
    assert "app.autoAI" in js
