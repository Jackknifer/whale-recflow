# WHALE-RecFlow

使用 PyTorch 复现论文 [WHALE: A Scalable Unified Model for Recommendation with Wukong-HSTU Architecture](https://arxiv.org/abs/2607.17017v2) 中的 WHALE 模型。
模型逐层结合 Wukong、HSTU 和 Attention Fusion，并比较 Wukong-only 和 HSTU-only。

## 数据集

使用 [RecFlow](https://github.com/RecFlow-ICLR/RecFlow) 的 `realshow` 数据，日期为 2024-01-13 至 2024-02-18。
预测 `long_view` 标签。
训练、验证和测试样本分别为 3,000,000、100,000 和 200,000 条，按时间划分。

数据下载：[作者提供的下载入口](https://rec.ustc.edu.cn/share/f8e5adc0-2e57-11ef-bea5-3b4cac9d110e)。

## 训练配置

| 设置 | 取值 |
| --- | --- |
| Embedding／模型层数 | 64 维／2 层 |
| Batch Size／训练轮数／种子 | 256／1／42 |
| 优化器／学习率／weight decay | AdamW／0.001／0.00001 |
| 损失／梯度范数上限 | BCEWithLogitsLoss／1.0 |
| 训练 ID 遮盖概率 | 5% |
| 环境 | Python 3.13、PyTorch 2.13、CUDA、FP32 |

完整配置见 [configs/official.yaml](configs/official.yaml)。

## 结果

实验设备：NVIDIA GeForce RTX 4090。

| 模型 | AUC ↑ | LogLoss ↓ | NE ↓ |
| --- | ---: | ---: | ---: |
| Wukong | 0.727188 | 0.477716 | 0.892711 |
| HSTU | 0.733265 | 0.474901 | 0.887451 |
| WHALE | 0.729998 | 0.478255 | 0.893718 |

## 开源说明

- 源码：[Apache-2.0](LICENSE)，Copyright 2026 Jackknifer。
- HSTU：参考 [Meta generative-recommenders](https://github.com/meta-recsys/generative-recommenders)（Apache-2.0），已适配 RecFlow。
  Copyright (c) Meta Platforms, Inc. and affiliates.
- RecFlow 数据（Qi Liu 等）及实验汇总：[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en)。

## 参考文献

1. Renqin Cai et al. [WHALE: A Scalable Unified Model for Recommendation with Wukong-HSTU Architecture](https://arxiv.org/abs/2607.17017v2). arXiv, 2026.
2. Buyun Zhang et al. [Wukong: Towards a Scaling Law for Large-Scale Recommendation](https://proceedings.mlr.press/v235/zhang24ao.html). ICML, 2024.
3. Jiaqi Zhai et al. [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://proceedings.mlr.press/v235/zhai24a.html). ICML, 2024.
4. Qi Liu et al. [RecFlow: An Industrial Full Flow Recommendation Dataset](https://arxiv.org/abs/2410.20868). arXiv, 2024.
