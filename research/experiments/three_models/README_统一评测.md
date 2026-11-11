# 三模型统一评测（已整合至 DeepLabV3 母项目）

## 一键运行

在 `D:\AI\projects\DeepLabV3` 双击 `01_检查三模型文件.cmd`，检查后双击 `02_运行三模型统一评测.cmd`。

或在母项目目录的 PowerShell 运行：

```powershell
& "D:\AI\miniForge1\python.exe" .\experiments\three_models\evaluate_three_models.py
```

自动读取同一个 MSD `val.txt` 的全部图片，分开加载三种网络，输出 `unified_results_val`。

- `model_comparison.csv`：每模型 mIoU、mPrecision、mRecall、mF1、油污/污渍/划痕 IoU/P/R/F1、运行时间。
- `per_class_metrics.csv`：每模型每类别 IoU、Precision、Recall、F1/Dice、Specificity、TP/FP/FN。
- `<模型名>/metrics.json`、`confusion_matrix.csv`、`per_image.csv`：详细数据。

默认 DeepLabV3+ 方案3（CBAM+损失优化），SegFormer 640×1152（ROI+自适应置信度），FDSNet 公布的原始权重（修正类别编号，**不交换类别2和3**）。如需测 DeepLabV3+ 基线：

```powershell
& "D:\AI\miniForge1\python.exe" .\experiments\three_models\evaluate_three_models.py --models deeplab --deeplab-scheme 0
```

**提醒**：你之前报告中的72.72%是哪一份DeepLab权重，需要用这次统一评测重新核对。FDSNet作者原始训练样本可能与 `val.txt` 重叠，当前结果只作探索性对照，不能声称是严格的独立测试泛化指标。
