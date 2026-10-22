# 方法与复现范围

[项目首页](../README.md) · [实验报告](experiment.md) · [正式配置](../configs/paper.yaml)

## 论文与项目

WHALE 在每层结合 Wukong 特征交互、HSTU 历史建模和 Attention Fusion（注意力融合）。
本项目在公开 RecFlow 数据上实现该结构，并比较单独分支、浅层融合和均值读取。

| 方面 | 原论文 | 本项目 |
| --- | --- | --- |
| 数据 | 工业推荐数据 | RecFlow `realshow` |
| 核心问题 | 融合结构与规模扩展 | 当前规模的结构比较 |
| 序列扩展 | 3,000～15,000 条 | 最多 128 条 |
| 比较条件 | 调整规模，使 FLOPs 接近 | 相同宽度、层数和历史数量，计算量不同 |
| 输入映射 | 工业特征 | 公开字段适配 |

## 输入表示

监督目标为 RecFlow 发布的业务 `long_view`，当前候选反馈只参与损失计算。
十个类别字段和三个数值组形成 13 个非序列 Token（表示单元），Embedding（嵌入）维度为 64。

| 输入 | 字段 |
| --- | --- |
| 用户与上下文 | `user_id`、`device_id`、`age`、`gender` 和 `province` |
| 视频属性 | `video_id`、`author_id`、`category_level_one`、`category_level_two` 和 `upload_type` |
| 数值组 | 视频时长、请求时间周期和视频年龄 |
| 历史 | 五个视频属性、`effective_view`、`long_view` 和请求年龄 |

历史视频属性与候选共享 Embedding 表。类别字典拟合训练目标及其允许使用的历史，数值参数只拟合训练目标。
PAD（填充）为 0，UNK（未知类别）为 1，历史二值反馈编码为 2 和 3。

## 历史与时间

每条目标最多使用 128 条曝光历史，要求同用户、不同请求且请求时间严格更早。
历史按 `(request_timestamp, request_id, event_id)` 排列，右侧填充 PAD。
项目以请求内累计观看时间估计反馈可用时间，单位为毫秒：

$$
a_r=t_r+\sum_{e\in realshow(r)}playing\_time_e+60{,}000.
$$

历史还要求估计的 `a_r` 早于目标请求。`playing_time` 的毫秒解释和额外 60 秒均属于项目假设。
公开资料没有提供真实反馈完成时刻，额外等待时间未经延迟测量。
这些条件限定了当前离线协议，无法证明真实线上反馈时序。

验证和测试采用滚动历史，允许更早且满足上述条件的事件进入后续目标历史。评价期间模型参数保持固定。

## 模型结构

WHALE、Shallow-hybrid 和 MeanPool 将视频属性、两项反馈和年龄共同编码。
输入维度为 449，经过 `449 → 128 → 64` 的 MLP（多层感知机），每个事件形成一个 Token。
HSTU-only 分别编码物品和反馈，输入顺序为：

$$
item_1,action_1,\ldots,item_L,action_L,candidate.
$$

候选位置为 `2L`，最多 257 个 Token。其余历史分支最多为 128 个 Token。
候选 Token 来自当前 13 个非序列 Token，当前标签不进入输入表示。

| 模块 | 实现 |
| --- | --- |
| Wukong | FMB 使用完整内积，LCB 执行线性组合，输出 8 个 Token |
| HSTU | 4 个 Head，QK 与 V 每个 Head 各 16 维 |
| Fusion | 单个 Head 的 Attention 和 SwiGLU；每层独立参数 |
| 预测头 | SiLU MLP，输出一个 logit |

每个 WHALE 层更新两个分支，并使用非序列表示查询历史：

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

## 实现约定

HSTU 使用 SiLU Attention，并按模块最大长度归一化：历史分支为 128，交替输入为 257。
相对时间使用 `floor(log(max(abs(delta_ms),1))/0.301)`，结果限制在 0～128。
LayerNorm 的 `eps=1e-6`，HSTU 归一化层没有可学习的仿射参数。

共同历史分支允许读取严格更早的时间或自身位置。
交替输入同时要求 Token 因果顺序，允许反馈读取同一事件的物品表示，时间分桶采用下一事件时间。
PAD 每层清零；空历史在 Fusion 中产生零读取结果。

原始 `long_view` 总正例数与论文统计相差 58 条，原因尚未确认。实验保留发布标签。
训练参数、单个种子和计算量限制见[实验报告](experiment.md)。

## 参考文献

| 来源 | 项目采用的内容 |
| --- | --- |
| [WHALE，RecSys 2026](https://arxiv.org/abs/2607.17017v2) | 逐层结构、Fusion 和消融 |
| [Wukong，ICML 2024](https://proceedings.mlr.press/v235/zhang24ao.html) | FMB、LCB 和残差 |
| [HSTU，ICML 2024](https://proceedings.mlr.press/v235/zhai24a.html) | SiLU Attention、时间与交替输入 |
| [RecFlow，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/647904f8340763f26e62010872751f76-Paper-Conference.pdf) | 曝光数据、业务反馈与字段 |
| [Meta HSTU](https://github.com/meta-recsys/generative-recommenders) | 投影、时间分桶和相对位置的参考实现 |

WHALE 发表 DOI 为 `10.1145/3773078.3831927`，本项目公式核验使用作者 arXiv v2。
Wukong、HSTU 和 RecFlow 使用正式会议版本。字段映射及具体尺寸属于本项目适配。
