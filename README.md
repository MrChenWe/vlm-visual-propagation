# VLM Visual Propagation

本项目用于研究视觉信息如何在视觉语言模型（Vision-Language Model，VLM）及多模态 Agent 系统中传播。

当前研究关注的问题是：当文本指令和图片中的视觉指令发生冲突时，VLM 会如何理解和采纳这些信息？视觉信息是否会进一步影响模型的规划、工具调用和最终行动？

长期研究链路如下：

```text
图像中的视觉信号
        ↓
外部可恢复性（R_ext）
        ↓
VLM 语义可访问性（R_sem）
        ↓
意图采纳与规划（R_plan）
        ↓
工具调用与行动执行（R_act）
```

需要特别区分：专门解码器能够恢复隐藏信息，不代表 VLM 能够理解该信息；VLM 能够理解，也不代表 Agent 一定会采纳并执行。

## 当前进度

目前已经完成或正在整理的内容包括：

- Qwen2.5-VL-3B-Instruct 本地推理环境；
- VLM 基础推理和 sanity check；
- 初始 E0 视觉—文本控制实验；
- E0 v2 语义基础控制实验；
- E0 v2.1 偏置诊断实验；
- 视觉指令强度实验；
- 实验图片的自动生成；
- 模型输出和 CSV 结果的自动保存。

当前实验主要使用 `Qwen2.5-VL-3B-Instruct`。

目前结果仍属于探索性结果，不能直接解释为对所有 VLM 或 Agent 都成立的普遍规律。后续需要继续进行更严格的对照实验、因果验证和多模型验证。

## 目录结构

```text
vlm_visual_propagation/
├── data/
│   └── images/                         # 实验图片
├── figs/                               # 分析和报告使用的图表
├── outputs/                            # CSV、JSON 等实验结果
├── scripts/
│   ├── test_vlm.py                     # 基础 VLM 推理测试
│   ├── run_sanity_check.py             # 基础一致性检查
│   ├── run_e0.py                       # 初始 E0 实验
│   ├── run_e0_v2.py                    # 语义基础 E0 v2 实验
│   ├── run_e0_v2_1_bias_diagnostic.py  # 偏置诊断实验
│   ├── run_intent_strength.py          # 视觉指令强度实验
│   ├── generate_e0_images.py           # 生成 E0 图片
│   ├── generate_e0_v2_images.py        # 生成 E0 v2 图片
│   ├── generate_images.py              # 图片生成脚本
│   └── fix_image.py                    # 图片修复工具
├── report_src.md                       # 研究记录和实验报告
├── md2pdf.py                           # Markdown 转 PDF 工具
├── .gitignore
└── README.md
```

模型权重没有上传到本仓库。模型文件体积较大，并且应当按照模型发布方的方式单独下载。

## 环境要求

- Ubuntu 24.04；
- Python 3.10 或更高版本；
- 支持 CUDA 的 PyTorch；
- NVIDIA GPU；
- Transformers；
- Pillow；
- Qwen-VL 相关工具。

可以创建独立的 conda 环境：

```bash
conda create -n vlm-propagation python=3.10 -y
conda activate vlm-propagation
```

根据本机 CUDA 和 PyTorch 配置安装依赖：

```bash
pip install transformers pillow
pip install qwen-vl-utils
```

PyTorch 应按照本机 CUDA 环境单独安装。后续会补充固定版本的 `requirements.txt`。

## 模型准备

请单独下载 `Qwen2.5-VL-3B-Instruct`，并放置在本地模型目录中，例如：

```text
models/Qwen2.5-VL-3B-Instruct/
```

模型目录已被 `.gitignore` 排除，不会上传到 GitHub。

## 运行实验

进入项目根目录：

```bash
cd ~/AI-Master/vlm_visual_propagation
```

运行基础 VLM 测试：

```bash
python scripts/test_vlm.py
```

运行 sanity check：

```bash
python scripts/run_sanity_check.py
```

运行初始 E0 实验：

```bash
python scripts/run_e0.py
```

运行语义基础 E0 v2 实验：

```bash
python scripts/run_e0_v2.py
```

运行 E0 v2.1 偏置诊断实验：

```bash
python scripts/run_e0_v2_1_bias_diagnostic.py
```

运行视觉指令强度实验：

```bash
python scripts/run_intent_strength.py
```

实验结果通常保存于：

```text
outputs/
```

## 阶段化研究指标

后续研究使用四个阶段性概念描述传播链：

- `R_ext`：外部可恢复性，专门解码器能否恢复隐藏信号；
- `R_sem`：语义可访问性，VLM 能否识别和理解该信号；
- `R_plan`：规划采纳，模型是否把该信号纳入后续计划；
- `R_act`：行动执行，Agent 是否按照该计划调用工具并完成动作。

这四个指标不应被视为同一个指标。隐藏信号可能仍能被外部解码器恢复，但 VLM 无法理解；VLM 也可能理解了信号，却没有把它写入计划或执行动作。

## 后续研究计划

后续将按照以下阶段推进：

1. 视觉指令传播的可见基线实验；
2. message swap 和 payload destruction 因果对照；
3. 传统隐写信息注入；
4. 神经网络隐写与面向 VLM 的对抗视觉注入；
5. JPEG 压缩、缩放、截图和频域变换实验；
6. 多模型、多任务和跨条件验证；
7. Agent 规划与工具行动传播实验；
8. 针对不同传播阶段的检测和防御方法。

## 可复现性记录

每次实验应尽量记录：

- 模型名称和版本；
- prompt 模板；
- 图片编号；
- 图像处理条件；
- 随机种子；
- 模型原始输出；
- 解析后的决策结果；
- 实验运行时间；
- 实验配置。

## 研究状态说明

本项目仍在持续开发中。当前实验主要用于发现稳定、可解释和可复现的传播现象。随着对照实验、多模型实验和隐写实验的加入，研究问题和实验结论可能会进一步调整。
