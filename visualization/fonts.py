# -*- coding: utf-8 -*-
"""Matplotlib CJK configuration for Windows/Mac/Linux, before plotting."""
from __future__ import annotations
from pathlib import Path

def configure_chinese_matplotlib():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names={f.name for f in font_manager.fontManager.ttflist}
    preferred=['Microsoft YaHei','Microsoft YaHei UI','SimHei','SimSun','PingFang SC',
               'Noto Sans CJK SC','Noto Sans CJK JP','WenQuanYi Micro Hei','Arial Unicode MS']
    for name in preferred:
        if name in names:
            plt.rcParams['font.sans-serif']=[name,'DejaVu Sans']
            plt.rcParams['axes.unicode_minus']=False
            return name
    if __import__('os').name=='nt':
        for file in ('msyh.ttc','msyhbd.ttc','simhei.ttf','simsun.ttc'):
            font=Path('C:/Windows/Fonts')/file
            if font.is_file():
                font_manager.fontManager.addfont(str(font))
                name=font_manager.FontProperties(fname=str(font)).get_name()
                plt.rcParams['font.sans-serif']=[name,'DejaVu Sans']
                plt.rcParams['axes.unicode_minus']=False
                return name
    plt.rcParams['axes.unicode_minus']=False
    return None
