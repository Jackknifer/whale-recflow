# 模型设计与论文依据

## 研究范围

项目使用 RecFlow 曝光日志，比较特征交互与历史序列的逐层融合。
主模型为 WHALE，参照包括 Wukong、HSTU、Shallow-hybrid 和均值读取消融。
工业论文中的数据、模型规模和计算预算，与本项目条件分别说明。

| 来源 | 本项目使用的内容 |
| --- | --- |
| [WHALE，RecSys 2026](https://arxiv.org/abs/2607.17017v2) | 第 3 节的逐层结构与 Fusion，第 5 节的单轮训练和消融定义 |
| [Wukong，ICML 2024](https://proceedings.mlr.press/v235/zhang24ao.html) | FMB、LCB、残差和 Token 数量变换 |
| [HSTU，ICML 2024](https://proceedings.mlr.press/v235/zhai24a.html) | SiLU Attention、相对位置与时间、物品和反馈交替输入 |
| [RecFlow，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/647904f8340763f26e62010872751f76-Paper-Conference.pdf) | 真实曝光、业务反馈、字段含义和原数据评价方式 |
| [Meta HSTU 参考实现](https://github.com/meta-recsys/generative-recommenders) | UVQK 投影、时间分桶与相对位置方向；采用 Apache-2.0 |

WHALE 的发表 DOI 为 `10.1145/3773078.3831927`。
公式核验使用作者 arXiv v2，出版 PDF 尚未取得。
Wukong、HSTU 和 RecFlow 使用正式会议版本核对。

## 数据与时间条件

### 监督目标与特征

来源为 RecFlow `realshow` 的 37 个日文件。
`long_view` 使用发布的业务反馈，当前候选反馈只参与监督损失。
视频时长、观看时间和严格完播属于不同概念。

十个类别字段分别生成一个 Token：

- 用户与上下文：`user_id`、`device_id`、`age`、`gender`、`province`。
- 视频：`video_id`、`author_id`、`category_level_one`、`category_level_two`、`upload_type`。

`age` 使用发布的类别编码。
另外三个数值组为视频时长、请求时间和视频年龄，输入维度为 2、4 和 2。
每组通过 `Linear → SiLU → Linear` 生成一个 Token，共有 13 个非序列 Token。
当前观看反馈和阶段排名不进入候选特征。

历史使用五个视频属性、`effective_view`、`long_view` 和相对年龄。
五个视频字段与候选共享对应的 Embedding 表。
类别字典拟合实际训练目标及其合法历史，数值参数只拟合训练目标。
PAD 为 0，UNK 为 1，历史二值反馈编码为 2 和 3。

### 历史构造

每条目标最多使用 128 条曝光历史，按时间升序排列，右侧填充 PAD。
历史要求同用户、不同请求，且请求时间与可用时间均严格更早。
请求可用时间使用以下项目假设，单位为毫秒：

$$
a_r=t_r+\sum_{e\in realshow(r)}playing\_time_e+60{,}000.
$$

该公式根据累计观看时间估计反馈可用时间。
原始数据没有提供每项反馈完成时刻，公式无法证明真实反馈的完成时间。
`playing_time` 的毫秒解释保留为数据适配假设。
历史顺序按 `(request_timestamp, request_id, event_id)` 确定。
同次请求的不同历史事件不相互读取反馈。

验证和测试采用滚动历史，较早且已经可用的反馈可以进入后续目标历史。
评价期间模型参数保持固定。

## 输入表示

Embedding 维度为 64，层数为 2，LayerNorm 的 `eps=1e-6`。
历史的七个 Embedding 与 `log1p(age_seconds)` 拼接，维度为 449。
共同历史编码使用 `449 → 128 → 64` 的 MLP，中间使用 SiLU。
WHALE、Shallow-hybrid 和 MeanPool 的历史分支采用该表示。

HSTU-only 使用以下顺序：

$$
item_1,action_1,\ldots,item_L,action_L,candidate.
$$

物品 Token 使用历史视频属性与年龄，反馈 Token 使用历史反馈。
候选 Token 来自 13 个当前非序列 Token 的投影。
长度为 `L` 的有效历史，对应候选位置 `2L`，最多 257 个 Token。
当前标签独立传入损失函数，候选表示不读取当前反馈。

## Wukong 交互

对于输入 `X [B,N,D]`，FMB 使用完整内积矩阵：

$$
G=XX^\top,\qquad F=reshape\bigl(MLP(LN(flatten(G)))\bigr).
$$

FMB 输出 4 个 Token，MLP 隐藏维度为 128。
LCB 使用线性组合生成另外 4 个 Token：

$$
C=W_LX,\qquad Y=LN([F;C]+R(X)).
$$

输出包含 8 个 Token。
输入数量为 8 时，`R` 使用恒等残差；其他数量使用沿 Token 维度的线性投影。
WHALE 首层接收 13 个 Token，Shallow-hybrid 首层接收 14 个 Token。
本实现使用基础完整内积，Wukong 原论文默认实验还涉及优化 FM。

## HSTU 编码

每层使用 4 个 Head，QK 和 V 的每个 Head 维度均为 16。
输入归一化和聚合归一化采用无可学习参数的 LayerNorm。

$$
[U,V,Q,K]=Split\bigl(SiLU(LN(X)W_{uvqk})\bigr).
$$

$$
S_{bhij}=\frac{SiLU(\langle Q_{bhi},K_{bhj}\rangle+r^p_{ij}+r^t_{bij})}{L_{max}}.
$$

位置参数索引为 `j-i+L_max-1`。
相对时间使用以下分桶公式，时间单位为毫秒：

$$
bucket=clip\left(\left\lfloor\frac{\log(\max(|\Delta t|,1))}{0.301}\right\rfloor,0,128\right).
$$

$$
A=LN(ConcatHeads((S\odot M)V)),\qquad
Y=X+Linear(A\odot ConcatHeads(U)).
$$

分母使用模块最大长度：共同历史分支为 128，HSTU-only 为 257。
UVQK 投影没有 bias，输出投影包含 bias。
位置和时间参数在同一层的全部 Head 间共享。

共同历史分支要求两个位置有效，并满足 `time_j < time_i` 或 `i == j`。
HSTU-only 还限制 Token 因果顺序，允许同一事件的反馈读取其物品表示。
交替输入的时间分桶采用下一事件时间方式，候选读取合法历史。
每层完成计算后，将 PAD 位置清零。

## WHALE 逐层融合

每层依次更新非序列表示、历史表示，并从历史中读取信息：

$$
N'=Wukong(N),\qquad H'=HSTU(H).
$$

$$
Q=LN(N')W_Q,\quad K=LN(H')W_K,\quad V=LN(H')W_V.
$$

$$
A=Softmax\left(\frac{QK^\top}{\sqrt D}+mask\right)V.
$$

$$
Z=N'+[N';A]W_f,\qquad N_{next}=Z+SwiGLU(LN(Z)).
$$

Fusion 使用一个 Head，投影没有 bias，SwiGLU 隐藏维度为 128。
两层各自拥有参数，更新后的 `N` 和 `H` 分别进入下一层。
空历史产生零读取结果。
最终将 8 个非序列 Token 展平，经预测头输出一个 logit。

Shallow-hybrid 对 HSTU 编码结果求有效均值，再作为额外 Token 输入 Wukong。
MeanPool 保留逐层 Fusion 的投影和 SwiGLU，对归一化及 V 投影后的历史求有效均值。
该消融使用均值替代根据当前特征进行的 Attention 读取。

## 复现条件与项目设置

| 内容 | 当前实现 |
| --- | --- |
| 核心结构 | 依据 WHALE 的逐层交互和均值读取消融 |
| 数据 | 公开 RecFlow 曝光；原论文采用工业数据 |
| 模型规模 | 64 维、2 层、128 条历史行为 |
| 输入与预测头 | 字段映射、数值 MLP、SiLU 和尺寸为项目设置 |
| 历史反馈时间 | 使用明确记录的项目可用时间假设 |
| 训练 | 3,000,000 条目标、单轮、AdamW、固定 ID 遮盖 |
| 比较条件 | 相同目标、历史行为数量和遍历次数，计算量分别报告 |

原论文没有披露完整字段映射和所有优化器数值。
补充设置均由配置明确记录。
相同计算预算、多个种子及更长历史的效果尚未验证。

## 论文与训练依据

[单轮次过拟合研究，CIKM 2022](https://arxiv.org/abs/2209.06053)
记录了重复训练中的性能退化，并研究特征稀疏与优化过程。
[自适应正则化，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/491bbea0db340e2d0bc1feea9059909a-Paper-Conference.pdf)
研究 Embedding 更新间隔与正则化。
[MEDA](https://arxiv.org/abs/2407.01607)研究 Embedding 与重复训练的关系。
这些工作为训练规定提供研究背景，当前项目只执行已记录的单轮配置。

RecFlow 原项目采用 [CC BY-SA 4.0](https://github.com/RecFlow-ICLR/RecFlow/blob/main/LICENSE)。
Meta HSTU 参考代码采用 [Apache-2.0](https://github.com/meta-recsys/generative-recommenders/blob/main/LICENSE)。
本仓库提供论文链接和署名，论文 PDF 由读者从原来源取得。
