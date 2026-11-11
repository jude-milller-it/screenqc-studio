# FDSNet 手机屏幕缺陷检测 · 自适应后处理研究项目

## 这是什么

基于 FDSNet 原作者预训练权重构建的**可直接运行的独立项目**。主要目标是研究 FDSNet 在 MSD 数据集上的**划痕、污渍的误检和漏检**，而不是把针对 SegFormer 设计的所有后处理强行套用。

- 包含原作者的模型架构代码（`fdsnet_core/`），原始权重、许可文件和新的适配代码；**不需要重新训练**。
- 基准：前一版脚本在用户的 **240 张 val 图上实测 mIoU 81.4263%**（油污90.2279%，污渍73.5717%，划痕62.0309%）。这是用户实验值，**不是本项目已重新测得的结果**。
- 类别编号按用户验证集实际核实为：`0=背景 1=油污 2=污渍 3=划痕`，不要再交换2和3。
- 原始 FDSNet 权重不可直接用于 SegFormer：这是**以FDSNet为主体**的改进方案。

## 运行环境

直接使用已经配置好的 `D:\AI\miniForge1\python.exe`（Miniforge base，Python 3.14），不创建虚拟环境、不重装Torch。依赖：`torch`、`numpy`、`Pillow`、`cv2`。如已有 FDSNet 评测能跑，就不需要再安装依赖。

解压本ZIP所有内容到 `D:\AI\projects\FDSNet_Adaptive_Postprocess\`，保证 `run.py`、权重文件和 `fdsnet_core` 在同一级目录。

打开 PyCharm 左侧的 `run.py`，或直接使用 PowerShell：

```powershell
cd D:\AI\projects\FDSNet_Adaptive_Postprocess
& "D:\AI\miniForge1\python.exe" run.py check
```

### 可选：一键打开 Windows 检测界面（不需要命令行）

在 PyCharm 左侧双击 `gui.py`，右键选择 **运行 'gui'**；或执行：

```powershell
& "D:\AI\miniForge1\python.exe" gui.py
```

点击 **①选择图片 → 选择后处理方案 → ②检测**，界面会显示标注图并保存文件到 `results\gui_predictions`。该界面使用Windows自带Tkinter，如果你的Python构建没有Tkinter，仍可以正常使用 `run.py predict`。

### 一、推荐第一步：小样本检查

```powershell
& "D:\AI\miniForge1\python.exe" run.py tune --limit 5 --preview 1
```

检查能否正常读取 MSD 图片和权重、生成 `results/tune_val/summary.csv`。

### 二、对整个val验证集做针对性消融（默认）

```powershell
& "D:\AI\miniForge1\python.exe" run.py tune
```

默认使用 `D:\AI\dataset\MobilePhoneScreenSurfaceDefectSegmentation\MSD`，读取 `ImageSets\Segmentation\val.txt` 列出的全部图片（通常240张）。

**只做一轮模型推理，每张图片在内存中直接计算13个后处理配置。** 无需把整套模型重复运行13次。结果在：

- `results\tune_val\summary.csv`：最重要，13种方案的全量结果及开发/保留子集指标；
- `results\tune_val\details.json`：详细类别准确率、召回率、实验条件；
- `results\tune_val\cm_*.csv`：每组混淆矩阵；
- `results\tune_val\gallery.html`：双击浏览样例可视化；
- `results\tune_val\candidate_selected.json`：**只根据开发子集选出**的候选配置，不会自动覆盖原配置。

**选择原则（在运行前固定）**：候选组比baseline开发集至少提升0.20个百分点，且油污/污渍/划痕任何一类IoU下降不超过1个百分点。达不到就选择baseline。最终另外展示保留子集（holdout）指标，防止仅靠反复调参凑分。

确定后想将选中配置用于预测，可用：

```powershell
& "D:\AI\miniForge1\python.exe" run.py tune --activate
```

会保存 `configs\selected.json`。注意：此结果仍是对当前val列表的本次分割，并不是保证外部泛化的完全独立测试集。

### 三、比较原版与某个改进方案

```powershell
& "D:\AI\miniForge1\python.exe" run.py evaluate --variant scratch_conf_cc
```

结果：`results\eval_val_scratch_conf_cc\summary.csv`、`per_image.csv`、`worst_oil_top20.csv`、`worst_stain_top20.csv`、`worst_scratch_top20.csv`、混淆矩阵、对照可视化。

比较自动选型（如已通过 `--activate` 保存）：

```powershell
& "D:\AI\miniForge1\python.exe" run.py evaluate --variant auto
```

### 四、检测任意一张图片 / 整个图片目录

```powershell
& "D:\AI\miniForge1\python.exe" run.py predict --image "D:\AI\dataset\MobilePhoneScreenSurfaceDefectSegmentation\MSD\JPEGImages\Oil_0125.jpg" --variant auto
```

或者：

```powershell
& "D:\AI\miniForge1\python.exe" run.py predict --image "D:\AI\my_images" --variant baseline
```

生成每张图片的 `*_mask.png`、`*_overlay.jpg`、`predictions.csv`、`gallery.html`（本地浏览器双击打开，无需部署服务器）。

### 五、选项

```powershell
# 扩展到20组候选，但后处理开销更大
& "D:\AI\miniForge1\python.exe" run.py tune --extended

# 改数据路径；不改变项目源码
& "D:\AI\miniForge1\python.exe" run.py tune --dataset "D:\新MSD目录"

# 需要有独立test.txt才能测试test，不会偷偷改为val
& "D:\AI\miniForge1\python.exe" run.py evaluate --split test --variant auto

# 单独验证水平翻转TTA（推理时间会增加）
& "D:\AI\miniForge1\python.exe" run.py evaluate --flip-tta --variant baseline
```

## 为什么不沿用以前全部的SegFormer后处理？

前一次实测混淆矩阵（原FDSNet模型）：

| 类别 | IoU | Precision | Recall | 主要问题 |
| --- | ---: | ---: | ---: | --- |
| 油污 | 90.23% | 97.87% | 92.04% | 本来就强，强行大面积过滤会毁掉真阳性 |
| 污渍 | 73.57% | 88.86% | 81.05% | 部分微小污渍被漏掉 |
| 划痕 | 62.03% | 77.00% | 76.14% | 误检和漏检均较多 |

候选方法（13组）主要包括：

- **轻量类别概率校准（新增）**：针对污渍高精确率、较低召回率，尝试对污渍类别分数×1.05或×1.10；针对划痕另试×1.05。这不依赖任何标注信息预测，是否提高mIoU需实测。
- **低置信度过滤**：对划痕分别尝试0.50、0.55阈值（不改油污），验证降低误检是否超过漏检损失。
- **置信度感知连通域过滤**：只有“很小且置信度低”的划痕/污渍区域才清理，保留真正的小缺陷。
- **受概率图约束的划痕连接**：尝试短距离连通断线，只在模型已经有一定划痕概率的背景像素填补，**不会覆盖已有油污或污渍**。
- **保守 ROI**：采用安全门控，屏幕范围不确定时自动跳过，避免删除整个图像内容。

每个方法都有实验组。与旧代码不同，**完全不使用油污最大面积上限**、不做“所有边缘都删除”、不对油污强制形态学开运算。

## 数据集与评价注意

1. FDSNet 权重是原作者公开的 MSD 模型，**可能已经训练过你现在val.txt中的图片**。本项目不宣称81.43%具有独立泛化意义。要验证，需要获得作者原始训练ID并核对，或用全新、未参与训练的数据。
2. `tune` 会从已有 val 划分出开发子集与保留子集；只在开发子集选模型，保留子集只读一次做报告。但这不能排除原作者预训练权重的训练图片重叠。
3. 输出结果必须如实记录。若消融没有超过81.43%，应保留原版FDSNet；不能直接把FDSNet改进结果写成SegFormer本身的改进结果。
4. 原版对照的推理尺度0.75、32倍数对齐、归一化方式、logits上采样与之前成功评测的脚本保持一致。若完整240张的baseline无法复现约81.4263%，**先停止调参并核对评测数据和标签**。
5. 部分方法是本项目提出的实验性组合，并不是原作者已验证的算法；是否有效以本机实际消融结果为准。

## 项目结构

```text
FDSNet_Adaptive_Postprocess/
├─ run.py                 # tune / evaluate / predict / check
├─ gui.py                 # 可选Windows图形界面：选图、检测、显示结果
├─ engine.py              # FDSNet权重、输入、推理、图片和标签处理
├─ postprocess.py         # FDSNet专用新后处理模块（消融开关）
├─ metrics.py             # 混淆矩阵和四类IoU等指标
├─ fdsnet_core/            # 原作者FDSNet网络代码，MMCV依赖已移除
├─ fdsnet__phone_voc_best_model.pth  # 原作者训练好的权重
├─ configs/               # 选中配置示例；tune --activate后写入selected.json
├─ tests/                 # 单元测试
├─ LICENSE_FDSNet_original.txt
├─ requirements.txt
└─ results/               # 运行时自动生成
```

原始模型： [jianzhang96/fdsnet](https://github.com/jianzhang96/fdsnet)。保留其开源许可证和版权归属。
