# WHALE-RecFlow

基于 PyTorch，在 RecFlow 上复现 [WHALE: A Scalable Unified Model for Recommendation with Wukong-HSTU Architecture](https://arxiv.org/abs/2607.17017v2) 的逐层 Wukong、HSTU 和 Attention Fusion 结构，同时比较 Wukong-only 和 HSTU-only。

## 数据与训练

使用 [RecFlow](https://github.com/RecFlow-ICLR/RecFlow) 的 `realshow` 数据，日期为 2024-01-13 至 2024-02-18。
预测发布的 `long_view` 标签，历史采用作者提供的最多 50 条有效观看，保留请求时间严格早于目标的记录。

| 设置 | 取值 |
| --- | --- |
| 训练／验证／测试样本 | 3,000,000／100,000／200,000 |
| Embedding／模型层数 | 64 维／2 层 |
| Batch Size／训练轮数／种子 | 256／1／42 |
| 优化器／学习率／weight decay | AdamW／0.001／0.00001 |
| 损失／梯度范数上限 | BCEWithLogitsLoss／1.0 |
| 训练 ID 遮盖概率 | 5% |
| 环境 | Python 3.13、PyTorch 2.13、CUDA、FP32 |

完整配置见 [configs/official.yaml](configs/official.yaml)。

## 结果

单轮实验在 RTX 4090 上完成，三模型使用相同测试成员。

| 模型 | AUC ↑ | LogLoss ↓ | NE ↓ |
| --- | ---: | ---: | ---: |
| Wukong | 0.727188 | 0.477716 | 0.892711 |
| HSTU | 0.733265 | 0.474901 | 0.887451 |
| WHALE | 0.729998 | 0.478255 | 0.893718 |

单轮及最多三轮的验证选权结果保存在 [results/comparison.csv](results/comparison.csv)，实验信息见 [results/experiment.json](results/experiment.json)。
结果来自单个种子的公开数据离线实验，论文的工业规模结论需要相应实验条件。

## 运行

将 37 个 `.feather` 日文件和 `seq_effective_50_dict.tar.gz` 放入 `data/raw/RecFlow/`。
数据、权重和运行输出由本机管理，通过 `.gitignore` 排除。

```bash
mkdir -p .work/runtime
export TMPDIR="$PWD/.work/runtime"
uv sync --frozen
uv run python scripts/prepare_data.py
uv run python scripts/run_experiment.py --phase check
uv run python scripts/run_experiment.py --phase train
uv run python scripts/export_results.py
```

源码采用 [Apache-2.0](LICENSE)，RecFlow 结果采用 [CC BY-SA 4.0](LICENSES/CC-BY-SA-4.0.txt)。第三方署名见 [NOTICE](NOTICE)。
