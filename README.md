# CCL25 中文框架语义解析

基于 CCL2025-Eval 汉语框架语义解析任务的实验项目：围绕 Frame 识别、论元边界识别和语义角色识别，探索标签语义、边界监督和流水线误差控制。代码基于[赛事官方基线](https://github.com/SXUNLP/The-3nd-Chinese-Frame-Semantic-Parsing)扩展；这些实验是对赛事任务的研究，不代表官方榜单成绩或获奖情况。

```text
句子 + 目标词 → Task 1：Frame → Task 2：论元 Span → Task 3：Frame Element Role
```

| 任务 | 方法 | 实验记录中的结果 |
| --- | --- | --- |
| Task 1 Frame 识别 | 修复学习率调度；Target-aware；定义原型和校准集成 | Dev Accuracy 74.26% |
| Task 2 论元识别 | GlobalPointer、合法 Span 掩码、边界辅助头、校准集成 | Dev Token-overlap F1 87.98%；记录的线上 F1 85.85% |
| Task 3 角色识别 | Target–Argument 交互、Frame 软先验、候选验证 | Dev 完整 Span–Role Tuple F1 61.35% |

以上数字来自原实验记录，不是本次重新运行的结果。模型选择、阈值和集成权重在 Dev 上确定；不同任务的指标不能直接比较。更多模型结构、负实验、指标口径见[架构与实验说明](docs/architecture.md)。

## 实验工作与阅读入口

项目从三个任务的误差分析出发，完成训练调度修复、合法候选与评价审计、Target 和标签语义建模、结构消融、阈值校准、专家集成及候选验证。当前归档可核查 **31 份逐轮指标文件、148 行指标记录**；这反映保留下来的实验记录规模，不等同于独立训练次数。

| 文档 | 内容 |
|---|---|
| [Task 1 实验](docs/task1-experiments.md) | Scheduler 修复、Last/Last4、Marker、Prototype、长尾分桶与融合 |
| [Task 2 实验](docs/task2-experiments.md) | A/B/C 消融、Target Gate、Content、G1/G2、H1/H1-R、校准及线上验证 |
| [Task 3 实验](docs/task3-experiments.md) | Frame prior、C/C2、D/D2、C3、D3/D4、I1B-R 与完整流水线 |
| [模型架构导航](docs/model-architectures.md) | 十张架构图嵌入上述各 Task 的对应实验章节，附张量维度与代码映射 |
| [实验规模与证据](docs/experiment-scale.md) | 实验目录清单和逐轮 CSV |
| [系统概览](docs/architecture.md) | 三任务连接、指标与复现条件 |

Task 1 的主要改进是修复训练并引入 Frame 定义原型；Target-aware 重点降低计算和显存。Task 2 通过训练/解码候选空间对齐建立基准，验证 Target、内容、语义和边界分支，再以校准集成利用互补误差。Task 3 用显式 Predicate–Argument 交互减少角色头参数，以定义边缘化和 Frame 软先验保留不确定性，并针对上游错误候选训练验证器。

关键对照包括：Task 1 十轮 baseline 73.5217% → hybrid 74.2609%；Task 2 校准 B 87.2391% → B/D/E/H 87.9759%；Task 3 完整流水线 baseline 59.7602% → 主角色集成 60.9740% → 保守拒识 61.3454%。这些提升分别使用各任务指标，不能相加为统一系统分数。

## 目录

```text
tasks/common/          公共参数、实验记录与检查工具
tasks/task1/           Frame 分类、语义原型与集成
tasks/task2/           Span 识别、边界和 Frame 引导实验
tasks/task3/           Role 分类、语义定义与候选验证
docs/                  结构、结果、数据准备与图示
```

## 数据、模型和运行

数据集和模型权重未包含在仓库中。请从[赛事页面](https://tianchi.aliyun.com/competition/entrance/532338)取得符合使用条件的 CFN 数据，在仓库根目录创建 `dataset/` 并放置 `cfn-train.json`、`cfn-dev.json`、`frame_info.json` 及所需测试集（例如 `cfn-test-B.json`）。代码默认从这些路径读取，使用数据前请遵守赛事数据条款。

预训练编码器是 [hfl/chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext)。将其 `config.json`、`vocab.txt` 和 `pytorch_model.bin` 放入根目录 `chinese_bert_wwm_ext/`。训练得到的任务 checkpoint、预计算原型、缓存和预测结果也不随仓库分发；最终集成需要相应模型和校准产物，不能仅凭基模型权重直接复现。

建议使用 Python 3.9 和与 CUDA 环境匹配的 PyTorch 1.13.0。安装 `requirements.txt` 后，**从仓库根目录**运行：

```bash
python -m pip install -r requirements.txt
python -m tasks.task1.train --experiment_name task1-baseline
python -m tasks.task2.train --experiment_name task2-valid --task2_valid_loss_mask --task2_safe_decode --with_adv_train
python -m tasks.task3.train --experiment_name task3-baseline
```

这些命令是各任务训练入口示例，并非论文式完整复现脚本；运行前请检查数据、GPU、依赖和参数。不同实验的模型、预训练初始化、输出目录及预测文件由脚本参数指定，使用 `python -m <模块名> --help` 查看。Task 3 流水线推理依赖 Task 2 预测，Frame 条件实验还需要 Task 1 结果。源代码中部分实验默认引用旧实验目录或缓存，需根据实际路径传参调整。

## 许可与致谢

本项目保留原项目 Apache-2.0 许可文件。赛事数据、中文 BERT 模型及其他第三方资源分别遵守其发布方的许可。感谢赛事组织方和官方基线作者。
