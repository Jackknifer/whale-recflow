# WHALE-RecFlow

[论文](https://arxiv.org/abs/2607.17017v2) · [方法](docs/research.md) · [实验](docs/experiment.md) · [结果](results/recflow)

基于 PyTorch，在 RecFlow 上独立复现 WHALE 的逐层融合结构。
研究特征交互与历史序列的联合建模，包含五模型比较、训练恢复和分组评价。

## 复现范围

| 项目 | 设置 |
| --- | --- |
| 数据 | RecFlow `realshow` |
| 监督目标 | 发布的业务反馈 `long_view` |
| 训练、验证、测试 | 3,000,000 / 100,000 / 200,000 |
| Embedding 维度、层数 | 64 / 2 |
| 历史行为上限 | 128 |
| 训练轮数、种子 | 1 / 42 |

本项目评价公开数据、小规模条件下的结构表现。原论文的工业数据和规模扩展结论另有实验条件。
`long_view` 使用数据集的业务定义，不能直接解释为严格完播。

## 实验结果

| 模型 | 测试 AUC ↑ | 测试 LogLoss ↓ | 测试 NE ↓ |
| --- | ---: | ---: | ---: |
| Wukong | 0.727525 | 0.477824 | 0.892913 |
| HSTU | 0.756088 | 0.462144 | 0.863612 |
| Shallow-hybrid | 0.754102 | 0.463354 | 0.865872 |
| **WHALE** | **0.757566** | **0.459887** | **0.859394** |
| WHALE MeanPool | 0.754784 | 0.462278 | 0.863863 |

五模型使用相同测试成员。此次运行中，WHALE 相对 HSTU 的 AUC 增加 0.001478，NE 相对降低 0.4884%。
NE 表示归一化交叉熵，定义与分组结果见[实验报告](docs/experiment.md)。

## 模型

| 标识符 | 说明 |
| --- | --- |
| `wukong` | 用户、候选与上下文的特征交互 |
| `hstu` | 交替输入历史物品与反馈，在候选位置预测 |
| `shallow_hybrid` | 历史摘要输入 Wukong |
| `whale` | 每层结合 Wukong、HSTU 和 Attention Fusion |
| `whale_mean_pool` | 用历史均值替代 Attention 读取 |

HSTU 最多处理 257 个 Token，其余历史分支最多处理 128 个 Token。
比较采用相同历史行为数量，各模型计算量分别报告。

## 复现入口

配置见 [configs/paper.yaml](configs/paper.yaml)，依赖版本由 [uv.lock](uv.lock) 固定。
已验证环境为 Python 3.13、PyTorch 2.13.0+cu130、RTX 5090 32 GB 和 FP32。
从 [RecFlow 官方仓库](https://github.com/RecFlow-ICLR/RecFlow)取得 37 个 `realshow` 日文件，放入 `data/raw/RecFlow/`。
日期范围为 2024-01-13 至 2024-02-18，文件扩展名为 `.feather`。

<details>
<summary>最小运行流程</summary>

```bash
mkdir -p .work/runtime .work/uv-cache
export TMPDIR="$PWD/.work/runtime" UV_CACHE_DIR="$PWD/.work/uv-cache"
uv sync --frozen --no-dev
uv run python scripts/prepare_data.py --config configs/paper.yaml --baseline-config configs/local.yaml
uv run python scripts/run_experiment.py --config configs/paper.yaml --phase train
uv run python scripts/export_results.py --config configs/paper.yaml
```

数据加工入口从原始文件重建目标成员。训练输出保存至 `outputs/paper/`。
完整参数见各脚本的 `--help`，公开汇总导出入口为 [export_public_results.py](scripts/export_public_results.py)。

</details>

## 研究限制

- 单个种子，测试成员曾用于早期实验；结论限定于当前配置。
- 历史反馈时间使用项目估计，真实反馈回传时刻未知。
- 模型计算量不同，长序列扩展和跨种子稳定性尚未验证。

## 引用与许可

使用本项目时，请引用 [CITATION.cff](CITATION.cff) 和[原模型与数据论文](docs/research.md#参考文献)。
源码与文档采用 [Apache-2.0](LICENSE)，RecFlow 汇总结果采用 [CC BY-SA 4.0](LICENSES/CC-BY-SA-4.0.txt)。
数据、逐条预测、权重和论文 PDF 由使用者在本地管理。第三方署名见 [NOTICE](NOTICE)。

代码与文档使用 AI 辅助编写和复核。模型与数据贡献归属于原作者。
