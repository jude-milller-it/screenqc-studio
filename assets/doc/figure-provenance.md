# Figure provenance and missing images

Public images in `assets/img/`:

- `qualitative-examples.png`: `Oil_0125`, `Sta_0217`, `Scr_0368`, each original plus a matching **archived** SegFormer overlay and enlarged crop. These are *not* final-checkpoint predictions, nor ground-truth masks. Source: original local DeepLabV3 project archive, `img/` and `out_segformer/`.
- `benchmark-val.png`, `segformer-ablation-val.png`: original project evaluation charts for the 240-image local validation split; see `docs/benchmarks.md` for numerical context.
- `scratch-recall-tradeoff.png`: chart assembled from scratch-class pixel Precision/Recall values in the 240-image validation reports (DeepLab3: 47.71/81.41, SegFormer final: 64.42/98.03, FDSNet: 77.00/76.14). These are not instance-level rates.
- `research-ui-crop.png`, `inspection-ui-crop.png`: cropped **existing real** UI screenshots; update these with unobstructed screenshots of the latest desktop release as desired.

**Normal (negative-control) screen**: not included because the local provided archives did not contain a verified intact-screen sample and its matching inference. The official MSD repository links a separate `good.zip` with 20 non-defective images: https://github.com/jianzhang96/MSD . Obtain a real sample there or from your local `normalPic`, run it through the same selected checkpoint, and add the original/prediction pair. Do not substitute a defect image or silently claim an unlabeled crop has no defect.

Verify the image reuse and GPL-3.0 requirements of the source MSD repository before public distribution. Photos are from the authors' dataset and are not original project captures.
