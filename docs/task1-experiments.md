# TASK1 实验与实现详解

本文从原始实验归档整理。指标为历史记录，未在本次重新训练。不同轮数、checkpoint、Oracle 条件和校准条件分别注明。

> 对照口径：Scheduler 修复的三轮记录为 73.174%，十轮为 73.522%；最终 74.261% 与十轮 baseline 比较。Top-k 与长尾表可能来自不同 checkpoint，应按表中条件解释。

## 2.1 任务定义与 baseline

输入：句子和已知 Target。输出：713 个 Frame 中的一个 Frame。

### Baseline 的 token 流

```text
[CLS] sentence_left [T_START] target [T_END] sentence_right [SEP]
        ↓
Chinese BERT-WWM-ext，输出 H ∈ R^{B×L×768}
        ↓
最后四层平均：H̄ = (H_{-4}+H_{-3}+H_{-2}+H_{-1})/4
        ↓
Dense(768 → 713×2×64)
        ↓
每个 token 得到 q_i,k_i ∈ R^{713×64}
        ↓
正弦位置编码 + 旋转式二维变换
        ↓
取 Target 起点 q_{t_s} 和终点 k_{t_e}
        ↓
s_f = q_{t_s,f}^{T}k_{t_e,f}/√64
        ↓
713 类 logits → Cross Entropy → Frame
```

### Baseline 的主要问题

1. **训练调度错误**：旧代码在一个 batch 内重复推进 `global_step`，学习率过早降到 0。
2. **分类头冗余**：Task1 的 Target 边界已知，却仍先生成每个 Token 的全部 q/k，再只取 Target 位置。
3. **长尾离散分类问题**：训练集覆盖 638/713 个 Frame，75 个 Frame 在训练中没有正样本。
4. **Target 只被分类头后处理使用**：BERT 自注意力阶段没有显式 target marker 语义，只是在输出后抽取 target 向量。
5. **普通离散分类不能利用 Frame 名称、定义和 FE 语义**。

### Task1 代码入口

| 作用 | 文件 |
|---|---|
| 数据、Target marker、标签 | `tasks/task1/dataset.py` |
| RoPE/Target-aware 模型 | `tasks/task1/model.py` |
| 训练、Scheduler、best checkpoint | `tasks/task1/train.py` |
| 单模型预测 | `tasks/task1/predict.py` |
| Prototype 文本构造 | `tasks/task1/prototype/utils.py` |
| Prototype 预计算 | `tasks/task1/prototype/build.py` |
| Prototype 模型 | `tasks/task1/prototype/model.py` |
| Baseline/aware 融合分析 | `tasks/task1/tools/analyze_ensemble.py` |
| 最终 Prototype hybrid 预测 | `tasks/task1/prototype/predict_hybrid.py` |

## 2.2 Task1 的实验主线

```text
E0：原始 baseline / 发现 Scheduler bug
  ↓
E1：Scheduler fix，建立可信的训练基线
  ↓
E2/E3：Target-aware last / last4，降低计算和显存
  ↓
E4：10 epoch + Baseline/Target-aware soft voting
  ↓
E5：Target Marker，验证显式输入标记是否带来额外语义
  ↓
E6：Frame Definition Prototype，把 Frame 当作语义原型而非离散 ID
  ↓
Final：Prototype calibration + Baseline 0.5/0.5 hybrid
```

## 2.3 Task1 主要实验性能表

Task1 为单标签分类，主指标是 Dev Accuracy；P/R/F1 不作为该任务的独立主评价。

| 编号 | 实验 | 主要改动 | Best Dev Acc | 相对原始 baseline | 结论 |
|---|---|---|---:|---:|---|
| E0 | 原始 baseline | RoPE + 旧 Scheduler | 69.435% | — | 学习率过早归零 |
| E1 | Scheduler fix | 修正 `global_step` 双计数 | **73.174%（3ep）；73.522%（10ep）** | **+3.739 pp** | 最重要的工程修复 |
| E2 | Target-aware Last | BERT 最后一层 + `[CLS; target_start; target_end; target_mean]` | 72.261%（3ep）/73.478%（10ep） | 10ep 与 E1 近持平 | 约半显存、约半时间 |
| E3 | Target-aware Last4 | 最后四层平均 + target-aware head | 72.217%（3ep）/73.348%（10ep） | 未超过 Last | loss 更低但 Accuracy 无收益 |
| E4 | 10ep Baseline + Last 集成 | Dev 网格搜索 soft voting | **73.913%** | +0.391 pp（相对10ep baseline） | 需要重新搜索权重 |
| E5 | Target Marker | 在输入中加入显式 Target marker | 72.739% | -0.739 pp（相对 target-aware） | 负向消融 |
| E6 | Frame Prototype MVP | Frame name + FE names + definition，余弦匹配 | 73.522% | 与10ep baseline持平 | unseen top-k 非零 |
| Final | Prototype calibration + Baseline hybrid | seen/unseen calibration，0.5/0.5 logits | **74.261%** | **+0.739 pp** | 当前 Dev 最优 Task1 |

### 资源对比：同一台机器的 3 epoch 对照

| 模型 | 每轮时间 | 峰值显存 | Checkpoint |
|---|---:|---:|---:|
| RoPE baseline | 约602.7 s | 3934 MB | 704 MB |
| Target-aware Last | 约306.3 s | 2013 MB | 418 MB |
| 相对变化 | **-49.2% 时间** | **-48.8% 显存** | **-40.6% 文件体积** |

注意：baseline-scheduler-fix-v2 和 target-aware-last4 的服务器/本地机器不同，不能直接做跨机器速度结论；严格效率结论只使用上述同机对照。

## 2.4 各实验的动机、组件与结论

### E1——Scheduler Fix

- **动机**：确认性能差是否来自模型，还是来自学习率在第二轮就已经归零。
- **改动位置**：训练循环，不改模型前向。
- **实现**：一个 batch 只执行一次 `optimizer.step()` 和一次 scheduler/global-step 更新。
- **结果**：69.435%→73.174%，提升 3.739 pp。
- **意义**：先把训练过程修正确，后续结构实验才有可比性。
- **文献**：不需要硬套论文，这是工程正确性修复。

### E2/E3——Target-aware Last / Last4

- **动机**：Baseline 先生成全部 Token 的 q/k，最后只取 Target 起止位置，存在明显无效计算。
- **改动位置**：BERT 输出之后、原始 Dense/RoPE 分类头位置。
- **新结构**：

  ```text
  [CLS] + target_start + target_end + target_mean
  → LayerNorm/Dropout/Linear(4×768 → 713)
  → Frame logits
  ```

- **Last**：使用 BERT 最后一层；**Last4**：使用最后四层平均。
- **结果**：10ep 后 Last 73.478%，几乎追平 baseline 73.522%；Last4 73.348%。
- **分析**：Target-aware 的收敛速度较慢，但最终 Accuracy 追平；主要价值是效率而不是立即提高精度。
- **文献依据**：Target-conditioned/predicate-aware BERT 的方向可参考 [Shi & Lin 2019](https://arxiv.org/abs/1904.05255)，但当前代码是针对 Frame Identification 的轻量自定义 head。

### E4——Baseline + Target-aware 集成

- **动机**：两种分类头的错误不完全重合。
- **实现**：对两个模型的 Frame logits 做加权和，在 Dev 上搜索 baseline weight：
  $s(f)=w s_{base}(f)+(1-w)s_{aware}(f)$
  
- **10ep 最优权重**：baseline 0.375，aware 0.625。
- **结果**：73.522%/73.478%→73.913%。
- **关键回答**：权重不是拍脑袋，也不是固定超参；是在 Dev 上遍历网格、以 Accuracy 最大为准则确定。模型重新训练后必须重新搜索。

### E5——Target Marker

- **动机**：验证 BERT 在输入阶段明确知道 Target 边界是否有帮助。
- **改动位置**：Tokenizer 输入序列，在 Target 左右插入 `[unused1]`、`[unused2]`。
- **结果**：72.739%，低于纯 target-aware 73.478%。
- **原因**：当前任务 Target 边界已经已知，后续 pooling 已提供精确位置信息；新 marker embedding 需要从小数据中学习，增加噪声但没有提供新的标签语义。
- **启发**：Task1 的瓶颈不是“模型不知道 Target 在哪里”，而是 Frame 的长尾和标签语义。

### E6——Frame Definition Prototype

- **动机**：普通 713 类分类头无法给 75 个 unseen Frame 学习正样本；需要利用 Frame 名称、定义和 FE 名称。
- **改动位置**：替代离散分类头，不改变句子 BERT 的主要编码流程。
- **实现**：

  ```text
  Frame 文本：frame_name + FE names + frame_definition
  → BERT 编码 → 固定 prototype [713,768]

  句子 + Target → BERT → context vector [768]
  → cosine(context, prototype_f)
  → 713 个 Frame logits
  ```

- **训练策略**：训练只在 638 个 seen Frame 上计算 CE；验证/测试恢复全部 713 个 prototype。
- **结果**：overall 73.522%，与 baseline 持平；unseen top-1 最佳 6.25%，unseen top-5 最佳 28.125%。
- **意义**：没有解决 unseen top-1 的排序偏置，但证明 Frame Definition prototype 能让正确 unseen Frame 进入候选集合。
- **文献**：AGED 明确提出使用 Frame/FE definition 学习标签语义；我们的 prototype 是其低成本、冻结文本原型的简化版本。

### Final——Prototype Calibration + Baseline Hybrid

- **问题**：Prototype 对 seen Frame 存在优势，导致 unseen prototype 的分数被整体压低。
- **校准**：对 seen prototype logits 减去 gamma：
  $s'_f=s_f-γ\cdot I(f\in Seen)
  $
  
- **最优设置**：gamma=2.7，baseline/prototype=0.5/0.5。
- **结果**：73.522%单模型→74.261%，Dev 增加 17 条正确样本。
- **分桶变化**：head +15、rare +2、few +1、medium -2、unseen +1；主要收益来自 head，但 prototype 也提供少量长尾互补。

## 2.5 Task1 特殊长尾表

| Frame 频次桶 | 数量 | Baseline | Prototype | Calibrated Prototype | Final Hybrid |
|---|---:|---:|---:|---:|---:|
| unseen=0 | 64 | 0/64 = 0.00% | 1/64 = 1.56% | 3/64 = 4.69% | 1/64 = 1.56% |
| rare 1–5 | 107 | 54/107 = 50.47% | 52/107 = 48.60% | 52/107 = 48.60% | 56/107 = 52.34% |
| few 6–15 | 255 | 171/255 = 67.06% | 175/255 = 68.63% | 175/255 = 68.63% | 172/255 = 67.45% |
| medium 16–50 | 684 | 515/684 = 75.29% | 502/684 = 73.39% | 502/684 = 73.39% | 513/684 = 75.00% |
| head >50 | 1190 | 951/1190 = 79.92% | 961/1190 = 80.76% | 961/1190 = 80.76% | 966/1190 = 81.18% |

