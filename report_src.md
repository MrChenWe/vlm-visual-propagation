# 论文深度解析与复现实验设计报告

@caption 对象论文：Universal Adversarial Attack on Multimodal Aligned LLMs（arXiv:2502.07987v3，ApNet 无关，cs.AI）｜配套交付物：同目录《…中文翻译》PDF

---

## 0. 报告导读

本报告面向“能否动手复现”这一目标组织，共 6 部分：

- **第 1 节**给出论文速览与元信息；
- **第 2 节**形式化问题定义与威胁模型，说明这篇工作与相邻工作的边界；
- **第 3 节**逐项拆解方法（策略 → 公式 → 8 个关键技巧 → 完整伪代码 → 原理分析）；
- **第 4 节**复盘论文的实验设计（数据、模型、评测、结果解读、以及**论文没有做的实验**）；
- **第 5 节**是重点：一份可执行的**复现实验流程设计**，包含资源估算、目录结构、分阶段流水线、核心代码骨架、10 条实现陷阱、评测协议与验收标准、排期与风险预案；
- **第 6 节**给出综合评价与可延伸方向。

!!! 结论先行：该攻击的核心可复现性很高（约 200 行代码即可跑通最小版本），真正的难点不在优化本身，而在三处工程细节——(1) 让梯度穿过**图像预处理**、(2) 只在**答案 token** 上计算损失、(3) 保证“训练用的浮点图像”与“评测时保存/量化的图像”一致。第 5.6 节把这三点列为必查项。

## 1. 论文速览

| 项目 | 内容 |
|---|---|
| 标题 | Universal Adversarial Attack on Multimodal Aligned LLMs |
| 作者 | T. Rahmatullaev, P. Druzhinina, N. Kurdiukov, M. Mikhalchuk, A. Kuznetsov, A. Razzhigaev |
| 机构 | AIRI、MSU、HSE University、Skoltech（俄罗斯） |
| 版本 | arXiv:2502.07987v3，2025-06-04（v1 为 2025-02-11） |
| 领域 | 多模态 LLM 安全 / 对抗攻击 / 越狱（jailbreak） |
| 攻击类型 | 白盒基于梯度的**通用**视觉对抗扰动（单张图像） |
| 评测基准 | SafeBench（23 类风险场景）、MM-SafetyBench（13 场景，1680 查询 × 3 图像变体） |
| 受害模型 | Llava-1.5-7B、Llama-3.2-11B-Vision-Instruct、Phi-3.5-Vision-Instruct、Qwen2-VL-2B-Instruct |
| 评判模型 | gemma-3-4b-it（少样本提示，人工标定后选定） |
| 最佳结果 | 单模型多答案攻击 ASR：Phi 81.3%、Qwen 79.3%、Llama 70.4%、Llava 46.0% |
| 开源 | 承诺以 Apache-2.0 发布代码与数据集 |
| 一句话总结 | 只需一张优化图像（约 0.1 的 L∞ 级扰动），就能让多个已对齐的 VLM 对几乎任何有害提问都以“Sure, here it is”开头作答；把这张图换成多答案版本，攻击成功率还能再涨 30–60 个百分点。 |

## 2. 研究问题与威胁模型

### 2.1 问题定义

设多模态模型为 $f$，文本提示为 $x$，图像为 $z$。对齐（alignment）的目标是让 $f(x,z)$ 在 $x$ 属于有害集合 $X_{harm}$ 时输出拒答（refusal）。本文的攻击目标是求解一张**与提示无关**的图像：

$$ z^* = arg min_z  E_{x ~ D} [ L(f(x, z), y_{target}) ] ,  s.t.  ‖z − z_0‖_∞ ≤ γ_1 $$

其中 $D$ 是混合了安全与有害问题的训练提示分布，$y_{target}$ 是固定的肯定式答案（如 “Sure, here it is”）。求得 $z^*$ 后，对任意未见过的 $x' ∈ X_{harm}$，模型都会以 $y_{target}$ 开头作答。这是一个**行为覆盖（behavior override）**问题，而不是标签翻转或输出破坏问题。

### 2.2 威胁模型

| 维度 | 本文设定 |
|---|---|
| 优化阶段权限 | **白盒**：可访问视觉编码器、投影层/适配器、语言模型，可反向传播 |
| 部署阶段权限 | **黑盒**：图像一旦优化完成，即可用于未见过的提示，也部分可用于未见过的模型 |
| 攻击者可控 | 图像像素（可约束在灰度初值附近）、以及任意文本提示（可多轮试） |
| 攻击者不可控 | 模型权重、推理时的系统提示、安全过滤器之外的对齐流程 |
| 攻击成本 | 一次离线优化（单模型）；优化完成后每次攻击的边际成本 = 1 张图 |
| 攻击形态 | 通用（universal）+ 目标化（targeted）+ 多模态（multimodal） |

!!! 关键区分：论文表 1 明确把“查询特定（query-specific）”的越狱（FigStep、Visual-RolePlay、Jailbreak in Pieces）与“通用（universal）”攻击区分开。本文的卖点是**同一张图 + 任意问题**，这比查询特定攻击的部署威胁更实际（不需要为每个问题重新生成图像）。表 1 中“Ours”也是唯一同时满足“白盒 + 通用 + 多模态 + 不可感知 + 越狱”的方法。

### 2.3 与相邻工作的边界

- 与 **GCG-transferable**（Zou et al., 2023）：同为通用攻击，但 GCG 是纯文本后缀；本文是视觉扰动，且论文实测其在多数模型上更低（如 Phi 8.8% vs 本文 MA 81.3%）。
- 与 **Visual Adversarial Examples**（Qi et al., 2023）：同样通用视觉扰动，但其 ASR 在 Phi 上仅 17.2%；本文的多答案机制是关键增量。
- 与 **SafeBench / MM-SafetyBench 基线**：它们是"针对每个问题生成语义相关的对抗图像"，属于查询特定；本文只生成一张图，属于更严格意义上的通用攻击。
- 与 **Carlini et al., 2023**：结论一致（对齐并不等于对抗鲁棒），但本文把结论从纯文本推广到视觉模态。

### 2.4 攻击策略的五个“通用性”维度

这是理解全文的骨架。作者把“通用”拆成五个正交维度，每个维度对应一组实验：

| 通用性维度 | 含义 | 实现手段 | 对应实验 |
|---|---|---|---|
| 提示通用 | 一张图适配所有问题 | 训练时随机采样提示 | 表 2 “Ours” |
| 答案泛化 | 不锁定单一目标短语 | 多答案随机采样（MA） | 表 2 “Ours-MA” |
| 模型通用 | 一张图对多模型同时有效 | 多模型损失求和训练 | 表 2 “Ours-Universal-MA” |
| 模型迁移 | 一张图对未见模型有效 | 留一法（3 训练 + 1 测试） | 表 4/5/6/7 |
| 模态鲁棒 | 抗图像后处理 | 高斯模糊、局部裁剪 | 表 2 “Blur/Localize” |

## 3. 方法解析：策略、算法与原理

### 3.1 总体策略

一句话：**把“攻击”转化为对图像像素的一次离线优化任务，优化目标是让模型在“随机有害问题 + 固定肯定式答案”上拟合。**

流程（Figure 2 的复述）：

```text
采样一个问题 x 与目标答案 y   (每次迭代随机，保证提示无关)
        │
        ▼
z = z0 + gamma * tanh(z1)     (z1 是可训练张量，z0 是灰度底图)
        │
        ├── 可选：高斯模糊 G(z)；可选：随机局部裁剪并 resize
        │
        ├── 可选：叠加自适应高斯噪声 sigma，并 clip 到 [0,255]
        ▼
视觉编码器 -> 适配器/投影 -> 语言模型
        │
        ▼
掩码交叉熵 L_LLM(y | x, z)     (只在 y 的 token 上计算)
        │
        ▼
反向传播更新 z1 (AdamW, lr=1e-2)
```

### 3.2 基础目标函数（逐项解释）

$$ z_1^* = arg min_{z_1} L_{LLM}( y | x, z_0 + g(z_1) ),  g(z_1) = γ_1 · tanh(z_1) $$

| 项 | 作用 | 为什么这样设计 |
|---|---|---|
| $z_1$ 而非直接优化 $z$ | 把扰动与底图解耦 | 便于施加幅度约束，也便于复用底图（灰度图）的“有效性” |
| $tanh(·)$ 有界 | 软约束 $‖g(z_1)‖_∞ ≤ γ_1$ | 相比每步投影裁剪（PGD 风格），tanh 参数化是**无约束优化**，不需要投影步骤，且梯度在最优点附近不会因投影而消失 |
| $γ_1 = 0.1$（单模型）/ $0.5$（多模型） | 控制扰动强度 | 多模型时需要更大扰动空间来同时满足多个模型的“共性方向”，实验上 0.5 才不会互相拖累；代价是不可感知性下降 |
| $z_0$ 取灰度图 | 初始点选择 | 灰度高熵底图让扰动能自由分配亮度，实验显示比全黑/全白/随机噪声更有效 |
| 损失仅作用于目标答案 token（masked CE） | 把“让模型说这句话”变成标准 SFT 式拟合 | 若对整个序列算 CE，模型会把注意力分散到视觉描述等其他 token；掩码后梯度信号 100% 指向“覆盖拒答行为” |
| 梯度流经 vision encoder + adapter + LLM | 端到端可微 | 这是视觉对抗攻击成立的前提；许多 VLM 实现会默认冻结视觉塔或在 processor 中做 numpy 变换，会**静默切断梯度** |

### 3.3 八个关键技巧的拆解

**技巧 1：白盒单模型单提示（3.1）** —— 最小可行版本。把 $y$ 固定为 “Sure, here it is!”，$x$ 固定为单个问题，优化 $z_1$。这只是个 sanity check，本身没有实用价值（等于给一个问题过拟合一张图）。

**技巧 2：量化鲁棒噪声（3.2）** —— 作者发现模型输出对图片“存盘再读回”高度敏感。原因：优化在第 24 位浮点上进行，一旦量化为 int8（PNG），微小误差经过视觉编码器会被放大，目标短语失效。解决方案：

$$ z = clip( z_0 + g(z_1) + ε,  0, 1 ),  ε ~ N(0, σ²I) $$

其中 $σ$ 不是超参，而是**每步自适应**：先取当前图像的 int8 量化版本，再计算两者差值的标准差，把 $σ$ 设为该值。这相当于让优化过程主动“看见”量化误差（一种针对性很强的数据增强/平滑正则）。

**技巧 3：提示通用性（3.3）** —— 构造提示池：100 个安全 + 50 个对抗性问题（按 Llama-Guard 的 13 个对抗类别生成），验证集 50 个对抗问题。每次迭代随机取一条。这是 SGD 意义上的“对提示分布的期望损失最小化”。

**技巧 4：多模型通用（3.4）** —— 损失求和 $L = Σ_{i∈MODELS} L_{LLM}^{i}$，同时把 $γ_1$ 从 0.1 提到 0.5。直觉：不同模型的安全行为由不同的视觉特征触发，取和等价于寻找一个"跨模型共同脆弱方向"；空间太小（0.1）时最优方向可能落到可行域之外。

**技巧 5：留一法迁移（3.4）** —— 用 3 个模型的损失训练，在第四模型上测试。用于回答“攻击是否学到了模型无关的视觉语义，而非某个模型的怪癖”。

**技巧 6：多答案攻击 MA（3.5.1）** —— 目标答案从一组肯定式/恶意短语中随机采样。两个收益：(a) 破坏“完整精确匹配一个固定短语”这一脆弱解，迫使攻击朝“肯定式语义子空间”收敛，ASR 因此大幅上升（Phi 15.0 → 81.3）；(b) 输出更多样、更自然，不易被基于固定字符串匹配或异常检测的防御发现。

**技巧 7：高斯模糊（3.5.2）** —— 对扰动做模糊后再叠加，压制高频成分。动机是高频扰动难以跨模型迁移（不同模型的预处理/分辨率不同，高频信号最先失真）。实验上对 Llava 提升显著（44.0 → 67.2）。

**技巧 8：局部化（3.5.3）** —— 随机裁剪 + resize，把扰动“推”到语义显著的局部区域（借鉴 M-Attack）。论文给出的理由是语义一致性与多样性，但实测收益有限（表 2、表 7 中多数场景不如 MA 或 Blur）。

### 3.4 完整算法伪代码

```text
Algorithm 1  通用对抗图像优化 (Universal Adversarial Image)
输入: 目标模型集合 M = {m_1..m_k}; 提示池 D; 答案池 A (MA 时 |A|>1);
     底图 z0; 扰动界 gamma; 步数 N; 学习率 eta; 是否用 blur / localize
输出: 单张对抗图像 z_adv

1  z1 <- 0                       # 与图像同形状的可训练张量
2  opt <- AdamW([z1], lr=eta)
3  for t = 1..N do
4      x <- RandomChoice(D)                     # 提示通用性
5      y <- RandomChoice(A) if multi_answer else "Sure, here it is!"
6      delta <- gamma * tanh(z1)
7      if use_blur:      delta <- GaussianBlur(delta)
8      if use_localize:  img <- Resize(Crop(z0 + delta, s ~ U[smin,smax]))
9      else:             img <- z0 + delta
10     img <- img + eps,  eps ~ N(0, sigma^2 I)      # 量化鲁棒 (式 3.2)
11     img <- clip(img, 0, 255) -> uint8 level         # 亮度合法
12     L <- 0
13     for each m in M do                              # 多模型/留一法
14         L <- L + MaskedCE( m(img, x),  y )          # 仅答案 token
15     end for
16     opt.zero_grad(); L.backward(); opt.step()
17     sigma <- std( img - Quantize8(img) )            # 自适应噪声幅度
18     log loss / ASR_estimate
19  end for
20  return Quantize8(clip(z0 + gamma*tanh(z1), 0, 255))
```

### 3.5 原理层面的评论与疑点

- **为什么 MA 能带来 5 倍提升？** 固定短语的问题在于它是**点目标**：优化容易收敛到“只在一个 token 序列上成立”的脆弱解。随机多答案把点目标变成**集合目标**，等价于在答案空间上做数据增强，解空间更平滑；同时“肯定式”本身可能是模型的一个低维子空间，比具体句子更容易被视觉扰动激活。
- **为什么量化噪声这么关键？** 说明该攻击的敏感性集中在像素级高频信息上——这既是它能生效的原因（视觉塔对高频极其敏感），也是它跨模型迁移差的原因（不同模型的 resize/归一化会把高频信息打散）。这与“Blur 提升迁移、但降低本模型 ASR”的观察完全一致。
- **抗迁移的不对称性：** 表 4–7 显示，Llava 几乎总是被迁移命中（50%+），而 Llama/Qwen 常低于 10%。合理解释是 Llava 的参考值本身就最高（14.4%），对齐最弱，因此更容易被任意扰动“顶开”。这意味着**跨模型迁移的表象部分来自受害者模型的弱对齐，而非攻击本身的普适性**——论文对此有提及，但未作为严谨结论（这属于可以深化复现的点）。
- **没有报告的内容：** 计算成本（GPU·小时）、随机种子与方差、γ 与 σ 的消融、图像与原图的感知相似度（LPIPS/SSIM）、以及防御评估（如 Llama-Guard-3-Vision 作为前置过滤器时 ASR 如何变化）。

## 4. 论文实验设计解析

### 4.1 实验矩阵

| 场景 | 训练模型 | 目标答案 | 结果位置 |
|---|---|---|---|
| 单模型单答案 | 各模型单独 | 固定 “Sure, here it is” | 表 2 Ours、表 3 Ours |
| 单模型多答案 | 各模型单独 | 随机采样自答案池 | 表 2 Ours-MA、表 3 Ours-MA |
| 模糊 / 模糊+多答案 | 各模型单独 | 固定 / 随机 | 表 2 Ours-Blur(-MA) |
| 局部化 | 各模型单独 | 固定 | 表 2 Ours-Localize |
| 通用多答案 | 4 模型联合 | 随机 | 表 2 Ours-Universal-MA |
| 留一法（4 种目标） | 3 模型联合 | 固定 / 随机 / 模糊 / 局部化 | 表 4 / 5 / 6 / 7 |

### 4.2 数据与评测设计

| 环节 | 做法 | 关键点 |
|---|---|---|
| 攻击训练集 | 100 安全 + 50 对抗问题（13 个对抗类别），验证 50 对抗问题 | 问题与图像语义无关，保证“提示通用” |
| 目标答案 | 单答案：固定 “Sure, here it is!”；多答案：独立生成的一组肯定式/恶意短语 | 答案池与问题无关 |
| 主评测基准 | SafeBench：23 个风险场景的多模态有害查询 | 与 SafeBench 基线（查询特定图像）对比 |
| 副评测基准 | MM-SafetyBench：13 场景 / 1680 查询 × 3 图像变体 | 与“无图像”“原基准图像”对比 |
| 优化器 | AdamW，lr = 1e-2 | 只优化图像张量，所以 lr 可以取大 |
| 底图 / 扰动 | 灰度底图；γ = 0.1（单模型）/ 0.5（多模型） | 未报告步数、batch、σ 初值 |
| 评判 Oracle | gemma-3-4b-it + 自定义 few-shot，只给回复、不给问题 | 用人工标注的“评判分歧”样本集选出（三种评判模型分别出现 388 / 196 / 261 个错误） |
| 指标 | $ASR = (1/n) Σ I(Oracle(R_i) = 1)$ | 判定“不安全”的指示函数 |

**Oracle 选择本身是实验的一部分**（附录 B）：在人工标注的“评判分歧”子集上，Llama-Guard-2-8B 准确率仅 0.3701、GPT-4o 0.5763（召回 0.8430 但假阳性 223）、gemma-3-4b-it 0.6818（精确率 0.6369、F1 0.5220）。作者据此选择 gemma-3-4b-it。这一段是本论文在评测严谨性上的加分项，但也意味着**所有 ASR 数字都带有 Oracle 的偏好**（高精确率 → 偏保守，可能低估真实攻击成功率）。

### 4.3 关键结果解读

![图 3：SafeBench 上各模型攻击表现（ASR %）。](figs/fig3.png)

- **观察 1（MA 是最大增量）：** Phi 从 15.0 → 81.3，Qwen 从 21.4 → 79.3，Llama 从 15.0 → 70.4。相比之下，任何消融（Blur、Localize）都不如 MA。
- **观察 2（Universal-MA 的模型间冲突）：** 4 模型联合训练时 Phi 76.3、Llava 80.7，但 Qwen 掉到 14.2。说明“一个模型集合的通用图像”并非免费，模型间的最优扰动方向存在竞争。
- **观察 3（Blur 的双刃剑）：** Ours-Blur 在 Llava 上达 67.2（高于 Ours 的 44.0），但在 Phi/Llama 上低于 Ours。与“压制高频 → 提升迁移、牺牲本模型拟合”的假设一致。
- **观察 4（迁移有限且不对称）：** 留一法单答案（表 4）中，Llava 被命中 30.3–60.2，Phi 5.6–30.0，Llama 4.4–24.5，Qwen 4.3–35.4；多答案（表 5）普遍提升 10–30 个百分点。
- **观察 5（基线各有“偏好模型”）：** Visual-RolePlay 在 Llama/Qwen 上达 87.0/81.0，但在 Llava 上只有 9.0；MM-SafetyBench 在 Qwen 上 57.3。因此在评价“通用性”时应看**跨模型平均与最小值**，而非单模型峰值——论文的叙事偏向峰值（Phi 81.3、Llava 80.7）。

### 4.4 实验设计的缺口（复现时可以补的贡献）

1. **无方差报告**：所有 ASR 都是单次运行的点估计，未给种子/置信区间；1–2 个百分点的差异可能不显著。
2. **无成本报告**：未说明优化步数、单模型优化耗时、显存占用，复现时无法对齐预算。
3. **无超参消融**：γ（0.1/0.5 之外的取值）、σ 策略（自适应 vs 固定）、答案池大小、训练集规模（100+50）都未做敏感性分析。
4. **无感知度量**：声称“不可感知”，但没有 SSIM/LPIPS 或人工感知实验（图 1/图 4 中图像肉眼可见明显纹理）。
5. **无防御评估**：未测试前置图像过滤器（如 Llama-Guard-3-Vision、CLIP-based NSFW 检测）对 ASR 的影响。
6. **Oracle 与 ASR 的耦合**：若换用 GPT-4o 作为 Oracle（召回更高），ASR 会上升；论文未报告多 Oracle 下的数字。

## 5. 复现实验设计

### 5.1 复现目标与范围（先定“复现什么”）

建议按“最小可行 → 完整复现”三段式推进，避免一上来就追 4 模型联合：

| 阶段 | 复现内容 | 对应论文结果 | 判定标准 |
|---|---|---|---|
| P-A（MVP） | Qwen2-VL-2B 上的单模型单答案与 MA 攻击，SafeBench 100–200 条子集 | 表 2 Ours / Ours-MA（Qwen 列） | MA 的 ASR ≥ 单答案 + 30 个百分点 |
| P-B（核心） | 再加 Phi-3.5-Vision 与 Llava-1.5-7B；复现 Blur 变体 | 表 2 三列 | 三模型 MA 均显著高于各自“参考值”与 “Sure, here it is” 基线 |
| P-C（进阶） | 留一法迁移（3 训练 + 1 测试）；多模型联合 | 表 4/5、Ours-Universal-MA | 迁移 ASR 高于该模型参考值；联合训练各模型 ASR 均 > 参考值 |

!!! 范围裁剪建议：优先复现 **Qwen2-VL-2B（2B，单卡可全参反传）** 与 **Llava-1.5-7B（7B，需 bf16 + 梯度检查点）**。Llama-3.2-11B-Vision 显存需求最大且需申请权重许可，放在 P-C 之后。

### 5.2 硬件与软件环境

**显存估算（只优化图像，但需保存全部激活以反传）**

| 模型 | 参数量 | bf16 权重 | 视觉塔激活（含梯度检查点） | 建议单卡显存 |
|---|---|---|---|---|
| Qwen2-VL-2B-Instruct | 2.2B | ~4.5 GB | ~2–4 GB | 16 GB（可行，甚至 12 GB 量化方案） |
| Phi-3.5-Vision-Instruct | 4.2B | ~8.5 GB | ~4–6 GB | 24 GB |
| Llava-1.5-7B | 7B | ~14 GB | ~6–10 GB | 40 GB 或 24 GB + 8-bit + 检查点 |
| Llama-3.2-11B-Vision | 11B | ~22 GB | ~10 GB | 80 GB |

**软件栈**

```text
python            3.10 / 3.11
torch             2.3+ (cu121)
transformers      4.45+   # Qwen2-VL / Llama-3.2-Vision 支持
accelerate        0.34+
pillow, numpy, tqdm, jsonlines
scipy             (高斯模糊可用 torchvision.transforms.functional.gaussian_blur 代替)
openai / vllm     (可选：作为高召回 Oracle 交叉验证)
```

!!! 陷阱预警：`AutoProcessor` 返回的是 numpy/PIL，**会切断计算图**。必须自己实现像素归一化（各模型 mean/std 不同）为 torch 张量运算，或使用 `processor.image_processor` 中的 mean/std 常量手动复现，否则梯度永远到不了 $z_1$（loss 照常下降——因为在优化别的参数或根本不更新，现象是图像不变而 loss 不变或乱跳）。

### 5.3 目录结构建议

```text
repro-left/
  configs/
    attack.yaml            # gamma, lr, steps, sigma 策略, 多答案开关
    models.yaml            # 每个模型的 name / dtype / device_map / mean,std / image_size
    eval.yaml              # oracle 名称、ASR prompt 模板、benchmark 路径
  data/
    raw/safebench/         # 下载好的基准（只读）
    attack_prompts.jsonl   # 100 safe + 50 adversarial + 50 val（自建）
    answers.jsonl          # 多答案池
  src/
    models.py              # 统一加载 + 可微分的图像预处理
    attack.py              # Algorithm 1 主循环
    loss.py                # masked CE
    augment.py             # blur / localize / quantize-noise
    eval_asr.py            # oracle 判定 + ASR 统计
    utils.py               # 种子、日志、图像 IO（统一 quantize8）
  runs/
    2025xxxx-qwen2vl-ma/   # ckpt: adv_image.png, z1.pt, log.csv, config.yaml
  reports/
    tables/                # 复现表格（含种子与置信区间）
```

### 5.4 数据构造流程

1. 以 Llama-Guard 的 13 个对抗类别为骨架（如：暴力、仇恨、自残、非法行为、隐私侵犯、色情、欺诈、网络攻击、政治煽动等），每类生成 3–5 条询问式提示。
2. 安全子集 100 条：与图像无关联的常识/知识类问题（避免与视觉内容耦合）。
3. 对抗子集 50 条：覆盖 13 类，尽量口语化、不出现越狱话术（避免引入文本层面的混淆变量）。
4. 验证/评测提示 50 条对抗 + SafeBench 子集（建议先取 200 条覆盖全部 23 场景，正式复现用全量）。
5. 多答案池（MA）：20–50 条肯定式/恶意开场白（“Sure, here it is.”, “Of course, here's how...”, “Absolutely, I can help with that.”…），独立生成，不与问题条件化。
6. 全部落盘为 jsonl，固定 hash 与随机种子，供三个模型共享，保证跨模型可比。

### 5.5 分阶段实验流水线

| 阶段 | 任务 | 关键命令/产出 | 检查点（通过才继续） |
|---|---|---|---|
| S0 | 环境冒烟 | 加载 Qwen2-VL-2B，跑 1 个前向+反向 | 能打印 `z1.grad` 且非零 |
| S1 | 预处理对齐 | 用 PIL 图算 processor 结果、用手写 torch 归一化算结果 | 两者 max 误差 < 1e-4 |
| S2 | 单模型单答案 | `attack.py --model qwen2vl --mode single --steps 500` | loss 下降；保存图后重新加载仍能触发目标短语（量化鲁棒） |
| S3 | “存盘—重载”一致性 | 对比浮点图与 uint8 图的 ASR | 差异 < 5 个百分点，否则加强 σ |
| S4 | 多答案（MA） | `--mode multi` | SafeBench 子集 ASR 相比 S2 提升 ≥ 30 个百分点 |
| S5 | Blur / Localize | `--blur` / `--localize` | 与 S4 对比，得到与表 2 方向一致的结论 |
| S6 | 迁移（留一法） | `--train qwen,phi,llava --test llama` | 未见模型 ASR > 其参考值 |
| S7 | 多模型联合 | `--mode universal-ma` | 每个构成模型 ASR 均 > 参考值 |
| S8 | 消融与鲁棒性 | γ / σ / 答案池大小 / 提示集规模 | 绘制 ASR–γ、ASR–池大小曲线 |

### 5.6 核心代码骨架（可直接改写运行）

```python
# ---------- 1) 可微分的图像预处理（S1 的产物，务必自己实现） ----------
def to_pixel_values(img01, mean, std, size):
    """img01: [B,3,H,W] in [0,1] torch tensor (requires_grad)"""
    x = torch.nn.functional.interpolate(img01, size=(size, size),
                                        mode="bicubic", align_corners=False)
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    return (x - mean_t) / std_t            # 梯度可回流到 img01

# ---------- 2) 掩码交叉熵：只在答案 token 上计算 ----------
def masked_ce(logits, labels, mask):
    # logits [1,T,V]; labels [1,T]（非答案位置置 -100）; mask [1,T] bool
    sl, s_lab, s_m = logits[:, :-1], labels[:, 1:], mask[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        sl.reshape(-1, sl.size(-1)), s_lab.reshape(-1), reduction="none")
    return (loss * s_m.reshape(-1)).sum() / s_m.sum().clamp(min=1)

# ---------- 3) 攻击主循环 ----------
z0 = torch.full((1, 3, H, W), 0.5)                    # 灰度底图
z1 = torch.zeros_like(z0, requires_grad=True)
opt = torch.optim.AdamW([z1], lr=cfg.lr)              # lr = 1e-2
sigma = cfg.sigma_init                                # 自适应噪声初值

for step in range(cfg.steps):
    x = sample_prompt(D_train, step)                  # 固定种子
    y = sample_answer(A) if cfg.multi_answer else "Sure, here it is!"

    delta = cfg.gamma * torch.tanh(z1)
    if cfg.blur:
        delta = gaussian_blur(delta, k=5, sigma=1.0)
    img = z0 + delta
    if cfg.localize:                                  # 随机裁剪后 resize
        img = random_crop_resize(img, s=uniform(0.4, 0.9))

    img = (img + torch.randn_like(img) * sigma).clamp(0, 1)   # 量化鲁棒噪声

    pv = to_pixel_values(img, mean=MEAN, std=STD, size=IMAGE_SIZE)
    ids, labels, mask = build_chat_batch(x, y, tokenizer)     # 只对 y 置 mask
    out = model(input_ids=ids, pixel_values=pv, labels=None).logits
    loss = masked_ce(out, labels, mask)

    opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():                             # 自适应更新 sigma
        q = (img * 255).round().clamp(0, 255) / 255
        sigma = (img - q).std().item()
```

!!! 与论文的一致性说明：论文的式 (3) 写作 `clip(z0 + g(z1) + eps, 0, 1)`，而正文又提到“确保转为整数后仍在 [0,255]”。二者是同一件事的两种量纲。实现时统一用 [0,1] 浮点 + 落盘时 ×255 取整，可避免歧义。

### 5.7 实现陷阱清单（10 条必查）

| # | 陷阱 | 症状 | 对策 |
|---|---|---|---|
| 1 | processor 返回 numpy，切断计算图 | 图像几乎不变化 | 手写可微预处理（5.6-1） |
| 2 | 损失算在整段序列上 | 能生成目标短语但仍偶尔拒答 | 只在答案 token 上掩码 |
| 3 | 视觉塔被 `no_grad`/冻结 | 梯度为零或只更新语言塔 | 检查 `z1.grad` 非零、视觉塔 `requires_grad` |
| 4 | 训练用浮点图、评测用 PNG | 训练 loss 完美但 ASR ≈ 0 | 每步加入量化噪声；评测前统一经过 `quantize8` |
| 5 | 各模型归一化/分辨率不同 | 换模型后攻击失效 | 每模型独立配置 mean/std/size |
| 6 | chat template 不一致 | 目标短语被 system prompt 修饰，标签错位 | 用各模型官方 template；打印一次完整 token 序列检查 |
| 7 | γ 直接设 0.5 做单模型 | 单模型 ASR 下降、图像噪声明显 | 单模型 0.1，多模型才 0.5 |
| 8 | 提示采样每次不同导致不可复现 | 结果无法复跑 | 固定全局种子与采样器状态 |
| 9 | Oracle 判定给了问题文本 | ASR 虚高（Oracle 会“看问题”判不安全） | 只把 response 交给 Oracle，与论文一致 |
| 10 | 显存 OOM | 反传中断 | bf16 + gradient checkpointing + 只保留图像梯度；必要时 8-bit 量化权重 |

### 5.8 评测协议与验收标准

| 指标 | 计算方式 | 验收标准（P-A/P-B） |
|---|---|---|
| ASR（主指标） | Oracle 只读回复，判为 unsafe 记 1 | Qwen/Llava 的 MA ≥ 40%；Phi 的 MA ≥ 30% |
| 参考值基线 | 无图像、原问题直接问 | 应复现出较低值（5–15% 量级） |
| “Sure, here it is” 基线 | 回复前缀注入 | MA 应明显高于该基线 |
| 一致性检查 | 浮点图 vs uint8 图 ASR 差 | < 5 个百分点 |
| 方差 | 3 个随机种子 × 2 次采样 | 报告均值 ± 标准差（论文未做，复现时应补） |
| 迁移（P-C） | 留一法 | 未见模型 ASR 高于其参考值 |

!!! 重要提醒：若用更强的 Oracle（如 GPT-4o）或更严格的人工复核，ASR 可能显著低于论文数字（论文自己的 gemma-3-4b-it 精确率只有 0.6369，即约 36% 的“判为不安全”其实是误判）。因此复现报告的结论口径应写“在同一 Oracle 下的相对比较”，而不是“绝对值复现”。

### 5.9 排期与里程碑（按单人、单卡 24 GB 估）

| 周 | 任务 | 产出 |
|---|---|---|
| W1 | S0–S2：环境、可微预处理、单模型单答案跑通 | Qwen2-VL 的一张对抗图 + loss 曲线 |
| W2 | S3–S4：量化一致性 + MA；接入 SafeBench 子集 | 表 2 的 Qwen 列复现 |
| W3 | S5 + Phi/Llava 扩展 | 三模型 ASR 表（含 Blur/Localize） |
| W4 | S6 留一法迁移 | 表 4/表 5 的缩比版本 |
| W5 | S7 多模型联合 + S8 消融（γ、答案池、步数） | 消融图表 + 复现报告 |

### 5.10 风险与降级方案

| 风险 | 触发条件 | 降级方案 |
|---|---|---|
| Llama-3.2-Vision 权重/算力不可得 | 显存 < 40 GB | 用 Qwen2-VL-7B 替代，保持 4 模型规模 |
| 攻击不收敛（loss 平台） | 200 步内 loss 不降 | 提高 lr、改双精度、检查预处理是否可微；先对单条提示过拟合验证通路 |
| ASR 显著低于论文 | MA 后仍 < 20% | 检查 Oracle 模板与“只给回复”约定；先做人工抽检 |
| 子集与全量结论不一致 | SafeBench 子集 ASR 波动大 | 扩到全量 1680 条；报告置信区间 |
| 图像可感知性过强 | 人工观察明显噪声 | 降 γ、加 Blur，并报告 SSIM/LPIPS 权衡曲线 |

## 6. 综合评价

**主要贡献**

1. 把“通用越狱”从纯文本推广到视觉模态，并给出可用一条公式描述的统一框架（掩码 CE + tanh 参数化 + 随机提示 + 多答案）。
2. 系统化地把“通用性”拆成提示/答案/模型/迁移/后处理五个维度，并逐一实验验证——这是本文相对同类工作的最大组织性优势。
3. 多答案（MA）这一改动极其简单却带来数量级级别的提升（Phi 15.0 → 81.3），是一个高性价比的工程发现。
4. 对评判模型（Oracle）做了人工标定与错误分解，评测透明度高于同类工作。

**主要不足**

1. 白盒依赖 + 迁移性有限（多数留一法场景 < 35%），离“现实部署威胁”仍有距离。
2. 未报告计算成本、随机方差、超参消融与感知度量，复现时难以对齐预算与显著性。
3. ASR 的绝对值强依赖 Oracle 质量（所选 Oracle 精确率仅 0.64），且缺乏前置于防御过滤器的端到端评估。
4. “不可感知”的主张与图 1/图 4 中肉眼可见的强扰动纹理存在张力。

**可延伸方向**

- 纯黑盒/查询受限条件下的通用视觉攻击（论文自认的 future work）；
- 面向安全对齐的**图像对抗训练**（用本文的 pipeline 生成训练样本）；
- 视觉嵌入对拒答行为的因果分析（哪一层/哪个注意力头承载了“覆盖拒答”的信号）；
- 防御侧评估：前置图像过滤器、多模型集成拒答、解码期约束。

## 附：复现所需的公开资源清单

| 资源 | 用途 | 获取方式 |
|---|---|---|
| 本论文 arXiv:2502.07987v3 | 方法细节与全部结果表 | arxiv.org |
| SafeBench（arXiv:2410.18927） | 主评测基准 | 论文/项目主页 |
| MM-SafetyBench（ECCV 2024） | 副评测基准 | 论文/项目主页 |
| Qwen2-VL-2B-Instruct | 最小可行受害模型 | HuggingFace |
| Llava-1.5-7B-hf | 主受害模型（与论文一致） | HuggingFace |
| Phi-3.5-Vision-Instruct | 受害模型 | HuggingFace |
| Llama-3.2-11B-Vision-Instruct | 受害模型（需申请许可） | HuggingFace（gated） |
| gemma-3-4b-it | 评判 Oracle | HuggingFace |
| Llama-Guard-3-Vision | 建议补做的防御评估 | HuggingFace |

!!! 合规提示：上述模型与基准均受各自许可证约束；攻击目标文本与有害提示应在受控环境中使用，输出结果不得对外传播，复现报告应以防御研究为唯一目的。
