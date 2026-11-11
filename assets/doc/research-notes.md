# ScreenQC Studio — research notes

> These notes accompany the software repository. They distinguish implemented methods, evaluated configurations and exploratory ideas; they are not a thesis manuscript.

## Research questions

1. Can attention and loss changes improve a lightweight DeepLabV3+ baseline under strong background/defect pixel imbalance?
2. What training and inference decisions trade precision against recall for thin scratches and tiny spots in SegFormer?
3. Can changes be evaluated **without changing the trained weight**, and which attractive-looking postprocessors actually fail?

## Work completed

**DeepLabV3+** — MobileNetV2 backbone; experiments with CBAM channel/spatial attention, class-weighted Focal/Dice supervision, four training configurations (`baseline`, `loss-only`, `CBAM-only`, `combined`). Two main compared checkpoints: 68.19% and 72.72% validation mIoU. Optional morphological component filtering was implemented but disabled in the final comparison.

**SegFormer MiT-B2** — composite Dice/Focal/Lovász/Boundary training objective; OHEM hard-background mining, Copy-Paste synthetic defect augmentation and defect-biased cropping. Earlier overly aggressive weighting and cropping were revised; training-component effects have not all been independently remeasured.

**Crucial inference pivot** — older sliding-window/TTA/overlap fusion was not consistently aligned with training-time evaluation and was prone to false positives; resize-based inference became the benchmark reference. Subsequent same-checkpoint tests investigated input scale, approximate screen ROI, class confidence rules and oil-specific decisions. Final 240-image validation mIoU: 70.97% → 78.66%; the last oil-only increment was 76.73% → 78.66%. A historical morphology/area-filter variant fell to 67.42% overall, illustrating competing class effects.

**FDSNet comparison** — kept as a separate author-weight baseline with stronger mIoU/precision but lower scratch pixel recall in the local benchmark. Not an architecture-controlled comparison.

## What the results support — and what they do not

- They support **measured differences between configured pipelines** on the local MSD 240-image validation split.
- They support an engineering choice to favor **pixel-level recall** when false negatives are expensive, while making extra review cost explicit.
- They **do not** prove a percentage of real defects/devices found, generalization to new production conditions, or the isolated contribution of every training component.
- The validation split was used during tuning; the local `trainval` split mixes training and validation data.

## Evidence in the project

`research/train.py`, `research/train_segformer.py`, `research/nets/segformer_training.py`, `research/utils/dataloader_industrial.py`, `research/segformer_infer.py`, `research/segformer_ablation_9groups.py`, `model_adapters/registry.py`, `evaluation/runner.py`, `paper/evidence.py`.

Historical training logs, bulk predictions and full reports stay local and are excluded by `.gitignore`. Detailed chronological working notes are stored **locally only** in `assets/doc/private/research-master-zh.md`.
