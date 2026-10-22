# 方法与复现范围 / Method and Reproduction Scope

[项目首页 / Home](../README.md) · [实验报告 / Experiments](experiment.md) · [正式配置 / Configuration](../configs/paper.yaml)

## 论文与项目 / Paper and Project

WHALE 在每层结合 Wukong 特征交互、HSTU 历史建模和 Attention Fusion（注意力融合）。
本项目在公开 RecFlow 数据上实现该结构，并比较单独分支、浅层融合和均值读取。

WHALE combines Wukong feature interaction, HSTU history modeling, and attention fusion at each layer.
This project implements the architecture on public RecFlow data and compares individual branches, shallow fusion, and mean readout.

| 方面 / Aspect | 原论文 / Paper | 本项目 / This project |
| --- | --- | --- |
| 数据 / Data | 工业推荐数据 / Industrial recommendation data | RecFlow `realshow` |
| 核心问题 / Question | 融合结构与规模扩展 / Fusion and scaling | 当前规模的结构比较 / Architecture comparison at the current scale |
| 序列扩展 / History scaling | 3,000～15,000 条 / Events | 最多 128 条 / Up to 128 events |
| 比较条件 / Comparison | 调整规模，使 FLOPs 接近 / Adjust model sizes for comparable FLOPs | 相同宽度、层数和历史数量，计算量不同 / Same width, depth, and event count; different compute |
| 输入映射 / Input mapping | 工业特征 / Industrial features | 公开字段适配 / Adaptation to released fields |

## 输入表示 / Input Representation

监督目标为 RecFlow 发布的业务 `long_view`，当前候选反馈只参与损失计算。
十个类别字段和三个数值组形成 13 个非序列 Token（表示单元），Embedding（嵌入）维度为 64。

The target is RecFlow's released business label `long_view`; current candidate feedback is used only by the loss.
Ten categorical fields and three numerical groups produce 13 non-sequence tokens with embedding dimension 64.

| 输入 / Input | 字段 / Fields |
| --- | --- |
| 用户与上下文 / User and context | `user_id`, `device_id`, `age`, `gender`, `province` |
| 视频属性 / Video attributes | `video_id`, `author_id`, `category_level_one`, `category_level_two`, `upload_type` |
| 数值组 / Numerical groups | 视频时长、请求时间周期、视频年龄 / Duration, cyclic request time, video age |
| 历史 / History | 五个视频属性、`effective_view`、`long_view` 和请求年龄 / Five video attributes, both feedback labels, and request age |

历史视频属性与候选共享 Embedding 表。类别字典拟合训练目标及其允许使用的历史，数值参数只拟合训练目标。
PAD（填充）为 0，UNK（未知类别）为 1，历史二值反馈编码为 2 和 3。

Historical and candidate video attributes share embedding tables. Categorical vocabularies use training targets and their eligible histories; numerical preprocessing uses training targets only.
PAD is 0, UNK is 1, and binary historical feedback uses codes 2 and 3.

## 历史与时间 / History and Time

每条目标最多使用 128 条曝光历史，要求同用户、不同请求且请求时间严格更早。
历史按 `(request_timestamp, request_id, event_id)` 排列，右侧填充 PAD。
项目以请求内累计观看时间估计反馈可用时间，单位为毫秒：

Each target uses up to 128 exposed events from the same user, with a different request ID and a strictly earlier request timestamp.
Events are ordered by `(request_timestamp, request_id, event_id)` and right-padded.
The project estimates feedback availability from cumulative playing time within a request, in milliseconds:

$$
a_r=t_r+\sum_{e\in realshow(r)}playing\_time_e+60{,}000.
$$

历史还要求估计的 `a_r` 早于目标请求。`playing_time` 的毫秒解释和额外 60 秒均属于项目假设。
公开资料没有提供真实反馈完成时刻，额外等待时间未经延迟测量。
这些条件限定了当前离线协议，无法证明真实线上反馈时序。

The estimated `a_r` must also precede the target request. Interpreting `playing_time` as milliseconds and adding 60 seconds are project assumptions.
Public materials do not provide actual feedback completion times, and the additional delay has not been measured.
These conditions define the offline protocol without establishing actual online feedback timing.

验证和测试采用滚动历史，允许更早且满足上述条件的事件进入后续目标历史。评价期间模型参数保持固定。

Validation and testing use rolling histories: earlier events meeting these conditions may enter later histories. Model parameters remain fixed during evaluation.

## 模型结构 / Architecture

WHALE、Shallow-hybrid 和 MeanPool 将视频属性、两项反馈和年龄共同编码。
输入维度为 449，经过 `449 → 128 → 64` 的 MLP（多层感知机），每个事件形成一个 Token。
HSTU-only 分别编码物品和反馈，输入顺序为：

WHALE, Shallow-hybrid, and MeanPool jointly encode video attributes, two feedback fields, and age.
A `449 → 128 → 64` MLP produces one token per event.
HSTU-only encodes items and actions separately, using:

$$
item_1,action_1,\ldots,item_L,action_L,candidate.
$$

候选位置为 `2L`，最多 257 个 Token。其余历史分支最多为 128 个 Token。
候选 Token 来自当前 13 个非序列 Token，当前标签不进入输入表示。

The candidate occupies position `2L`, giving at most 257 tokens. Other history branches use up to 128 tokens.
The candidate token comes from the 13 current non-sequence tokens; its label is excluded from the input representation.

| 模块 / Module | 实现 / Implementation |
| --- | --- |
| Wukong | FMB 使用完整内积，LCB 执行线性组合，输出 8 个 Token / Full-dot-product FMB and linear-combination LCB; 8 output tokens |
| HSTU | 4 个 Head，QK 与 V 每个 Head 各 16 维 / 4 heads; 16 QK and V dimensions per head |
| Fusion | 单个 Head 的 Attention 和 SwiGLU；每层独立参数 / Single-head attention and SwiGLU; separate parameters per layer |
| 预测头 / Prediction head | SiLU MLP，输出一个 logit / SiLU MLP producing one logit |

每个 WHALE 层更新两个分支，并使用非序列表示查询历史：

Each WHALE layer updates both branches and uses non-sequence representations to query the history:

$$
N'=Wukong(N),\qquad H'=HSTU(H),
$$

$$
A=Softmax\left(\frac{Q(N')K(H')^\top}{\sqrt D}+mask\right)V(H'),
$$

$$
Z=N'+[N';A]W_f,\qquad N_{next}=Z+SwiGLU(LN(Z)).
$$

Shallow-hybrid 将 HSTU 的历史均值作为额外 Token 输入 Wukong。
MeanPool 保留逐层结构，以投影后的有效历史均值替代 Attention 读取。

Shallow-hybrid passes the mean HSTU history representation to Wukong as an extra token.
MeanPool preserves the layer-wise structure and replaces attention readout with the mean of projected valid history states.

## 实现约定 / Implementation Details

HSTU 使用 SiLU Attention，并按模块最大长度归一化：历史分支为 128，交替输入为 257。
相对时间使用 `floor(log(max(abs(delta_ms),1))/0.301)`，结果限制在 0～128。
LayerNorm 的 `eps=1e-6`，HSTU 归一化层没有可学习的仿射参数。

HSTU uses SiLU attention normalized by the module's maximum length: 128 for history branches and 257 for interleaved input.
Relative time uses `floor(log(max(abs(delta_ms),1))/0.301)`, clipped to 0–128.
LayerNorm uses `eps=1e-6`, and HSTU normalization has no learnable affine parameters.

共同历史分支允许读取严格更早的时间或自身位置。
交替输入同时要求 Token 因果顺序，允许反馈读取同一事件的物品表示，时间分桶采用下一事件时间。
PAD 每层清零；空历史在 Fusion 中产生零读取结果。

Joint history branches attend to strictly earlier timestamps or the same position.
Interleaved input also enforces causal token order, allows an action to read its own item, and uses next-event time alignment for bucketing.
PAD positions are zeroed after every layer; empty histories produce zero readout in Fusion.

原始 `long_view` 总正例数与论文统计相差 58 条，原因尚未确认。实验保留发布标签。
训练参数、单个种子和计算量限制见[实验报告](experiment.md)。

The raw `long_view` positive count differs from the paper's statistic by 58 examples; the cause remains unconfirmed. Experiments retain the released labels.
See the [experiment report](experiment.md) for training settings, the single-seed scope, and compute limitations.

## 参考文献 / References

| 来源 / Source | 项目采用的内容 / Used in this project |
| --- | --- |
| [WHALE, RecSys 2026](https://arxiv.org/abs/2607.17017v2) | 逐层结构、Fusion 和消融 / Layer-wise architecture, fusion, and ablations |
| [Wukong, ICML 2024](https://proceedings.mlr.press/v235/zhang24ao.html) | FMB、LCB 和残差 / FMB, LCB, and residual connections |
| [HSTU, ICML 2024](https://proceedings.mlr.press/v235/zhai24a.html) | SiLU Attention、时间与交替输入 / SiLU attention, time encoding, and interleaved input |
| [RecFlow, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/647904f8340763f26e62010872751f76-Paper-Conference.pdf) | 曝光数据、业务反馈与字段 / Exposure data, business feedback, and fields |
| [Meta HSTU](https://github.com/meta-recsys/generative-recommenders) | 投影、时间分桶和相对位置的参考实现 / Reference projections, time bucketing, and relative positions |

WHALE 发表 DOI 为 `10.1145/3773078.3831927`，本项目公式核验使用作者 arXiv v2。
Wukong、HSTU 和 RecFlow 使用正式会议版本。字段映射及具体尺寸属于本项目适配。

WHALE's publication DOI is `10.1145/3773078.3831927`; this project checks its formulas against the authors' arXiv v2.
Wukong, HSTU, and RecFlow are checked against conference publications. Field mappings and concrete dimensions are project adaptations.
