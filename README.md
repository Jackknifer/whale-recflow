<h1 align="center">WHALE-RecFlow</h1>

<p align="center">
  <em>基于 RecFlow 数据集的 WHALE 模型复现</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/%E6%A1%86%E6%9E%B6-PyTorch-C94B29?style=for-the-badge" alt="框架：PyTorch">
  <img src="https://img.shields.io/badge/%E6%95%B0%E6%8D%AE%E9%9B%86-RecFlow-0675BB?style=for-the-badge" alt="数据集：RecFlow">
  <a href="LICENSE"><img src="https://img.shields.io/badge/%E6%BA%90%E7%A0%81%E8%AE%B8%E5%8F%AF-Apache--2.0-318329?style=for-the-badge" alt="源码许可：Apache-2.0"></a>
</p>

<p align="center">
  <a href="#项目简介">项目简介</a> ·
  <a href="#快速链接">快速链接</a> ·
  <a href="#数据集">数据集</a> ·
  <a href="#训练配置">训练配置</a> ·
  <a href="#实验结果">实验结果</a> ·
  <a href="#参考文献">参考文献</a>
</p>

## 项目简介

使用 PyTorch 复现论文 [WHALE: A Scalable Unified Model for Recommendation with Wukong-HSTU Architecture](https://arxiv.org/abs/2607.17017v2) 中的 WHALE 模型。
模型逐层结合 Wukong、HSTU 和 Attention Fusion（注意力融合）。
实验同时包含 Wukong-only 和 HSTU-only。

## 快速链接

| 资源 | 入口 |
| --- | --- |
| 论文与数据来源 | [WHALE](https://arxiv.org/abs/2607.17017v2)、[Wukong](https://proceedings.mlr.press/v235/zhang24ao.html)、[HSTU](https://proceedings.mlr.press/v235/zhai24a.html) 和 [RecFlow](https://github.com/RecFlow-ICLR/RecFlow) |
| 完整实验配置 | [configs/official.yaml](configs/official.yaml) |
| 实验结果文件 | [results/comparison.csv](results/comparison.csv) |

## 数据集

| 项目 | 说明 |
| --- | --- |
| 数据来源 | [RecFlow](https://github.com/RecFlow-ICLR/RecFlow) 的 `realshow` 数据 |
| 日期范围 | 2024-01-13 至 2024-02-18 |
| 预测标签 | `long_view` |
| 训练样本 | 3,000,000 条 |
| 验证样本 | 100,000 条 |
| 测试样本 | 200,000 条 |
| 划分方式 | 按时间划分 |

数据下载：[作者提供的下载入口](https://rec.ustc.edu.cn/share/f8e5adc0-2e57-11ef-bea5-3b4cac9d110e)。

## 训练配置

| 设置 | 取值 |
| --- | --- |
| Embedding（嵌入）维度／模型层数 | 64 维／2 层 |
| Batch Size（批次大小）／训练轮数／随机种子 | 256／1／42 |
| 优化器／学习率／weight decay（权重衰减） | AdamW／0.001／0.00001 |
| 损失函数／梯度范数上限 | BCEWithLogitsLoss／1.0 |
| 训练 ID 遮盖概率 | 5% |
| 运行环境 | Python 3.13、PyTorch 2.13、CUDA、FP32 |

完整配置见 [configs/official.yaml](configs/official.yaml)。

## 实验结果

> 训练 1 轮，随机种子为 42。实验设备为 NVIDIA GeForce RTX 4090。

| 模型 | AUC ↑ | LogLoss ↓ | NE ↓ |
| --- | ---: | ---: | ---: |
| Wukong | 0.727188 | 0.477716 | 0.892711 |
| HSTU | 0.733265 | 0.474901 | 0.887451 |
| WHALE | 0.729998 | 0.478255 | 0.893718 |

## 参考文献

1. Renqin Cai et al. [WHALE: A Scalable Unified Model for Recommendation with Wukong-HSTU Architecture](https://arxiv.org/abs/2607.17017v2). arXiv, 2026.
2. Buyun Zhang et al. [Wukong: Towards a Scaling Law for Large-Scale Recommendation](https://proceedings.mlr.press/v235/zhang24ao.html). ICML, 2024.
3. Jiaqi Zhai et al. [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://proceedings.mlr.press/v235/zhai24a.html). ICML, 2024.
4. Qi Liu et al. [RecFlow: An Industrial Full Flow Recommendation Dataset](https://arxiv.org/abs/2410.20868). arXiv, 2024.
