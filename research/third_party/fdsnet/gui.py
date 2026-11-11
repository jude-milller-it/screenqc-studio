# -*- coding: utf-8 -*-
"""Optional Windows desktop viewer using built-in Tkinter (no new service)."""
from __future__ import annotations
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import numpy as np
import torch
from PIL import Image, ImageTk

from engine import load_model, predict_probs, PROJECT, save_artifacts
from postprocess import variants, apply
from run import choose_config


class App:
    def __init__(self, root: tk.Tk):
        self.root=root
        self.root.title('FDSNet 手机屏幕缺陷检测 · 原版/改进后处理')
        self.root.geometry('1120x760')
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=None
        self.image_path:Path|None=None
        self.photo=None
        self.outdir=PROJECT/'results'/'gui_predictions'
        bar=ttk.Frame(root,padding=14)
        bar.pack(fill='x')
        ttk.Button(bar,text='① 选择图片',command=self.choose).pack(side='left',padx=4)
        ttk.Label(bar,text='后处理方案：').pack(side='left',padx=(14,4))
        self.variant=tk.StringVar(value='auto')
        combo=ttk.Combobox(bar,textvariable=self.variant,state='readonly',width=24,
                           values=['auto']+[v.name for v in variants(True)])
        combo.pack(side='left',padx=4)
        ttk.Button(bar,text='② 检测',command=self.infer).pack(side='left',padx=10)
        self.status=tk.StringVar(value='先点击“选择图片”，然后点击“检测”｜auto 未激活时默认原版')
        ttk.Label(root,textvariable=self.status,padding=10).pack(fill='x')
        self.picture=ttk.Label(root,anchor='center',text='等待选择图片')
        self.picture.pack(fill='both',expand=True,padx=20,pady=10)
        ttk.Label(root,text=f'输出目录：{self.outdir} ｜GPU：{self.device}',padding=12).pack(fill='x')

    def choose(self):
        selected=filedialog.askopenfilename(filetypes=[('图像文件','*.jpg *.jpeg *.png *.bmp *.tif *.tiff')])
        if not selected:return
        self.image_path=Path(selected)
        self.status.set(f'已选择：{self.image_path.name}')
        self.preview(self.image_path)

    def preview(self,path):
        with Image.open(path) as im:
            img=im.convert('RGB')
        img.thumbnail((1030,610))
        self.photo=ImageTk.PhotoImage(img)
        self.picture.configure(image=self.photo,text='')

    def infer(self):
        if self.image_path is None:
            messagebox.showwarning('先选择图片','请先点击“选择图片”选择需要检测的图片。')
            return
        try:
            self.status.set('模型加载/推理中...')
            self.root.update_idletasks()
            if self.model is None:
                self.model=load_model(self.device)
            cfg=choose_config(self.variant.get())
            with Image.open(self.image_path) as im:
                rgb=im.convert('RGB')
            probs,ms=predict_probs(self.model,rgb,.75,self.device)
            pred=apply(probs,cfg,np.asarray(rgb) if cfg.roi_safe else None)
            stem=self.image_path.stem+'_'+cfg.name
            save_artifacts(rgb,pred,stem,self.outdir)
            output=self.outdir/f'{stem}_overlay.jpg'
            self.preview(output)
            counts=[int(np.count_nonzero(pred==i)) for i in (1,2,3)]
            self.status.set(f'{cfg.name} | 油污{counts[0]}px / 污渍{counts[1]}px / 划痕{counts[2]}px | '
                            f'模型推理约{ms:.0f}ms | 已保存 {output.name}')
        except Exception as exc:
            self.status.set(f'运行失败：{exc}')
            messagebox.showerror('FDSNet检测失败',str(exc))


if __name__=='__main__':
    root=tk.Tk()
    App(root)
    root.mainloop()
