# WHALE-RecFlow

[论文 / Paper](https://arxiv.org/abs/2607.17017v2) · [方法 / Method](docs/research.md) · [实验 / Experiments](docs/experiment.md) · [结果 / Results](results/recflow)

基于 PyTorch，在 RecFlow 上独立复现 WHALE 的逐层融合结构。
研究特征交互与历史序列的联合建模，包含五模型比较、训练恢复和分组评价。

An independent PyTorch reproduction of WHALE's layer-wise fusion on RecFlow.
This project studies joint feature interaction and history modeling, with five-model comparisons, training recovery, and subgroup evaluation.

## 复现范围 / Scope

| 项目 / Item | 设置 / Setting |
| --- | --- |
| 数据 / Data | RecFlow `realshow` |
| 监督目标 / Target | 发布的业务反馈 / Released business label: `long_view` |
| 训练、验证、测试 / Train, validation, test | 3,000,000 / 100,000 / 200,000 |
| Embedding 维度、层数 / Embedding dimension, layers | 64 / 2 |
| 历史行为上限 / Maximum history events | 128 |
| 训练轮数、种子 / Epochs, seed | 1 / 42 |

本项目评价公开数据、小规模条件下的结构表现。原论文的工业数据和规模扩展结论另有实验条件。
`long_view` 使用数据集的业务定义，不能直接解释为严格完播。

The experiments evaluate the architecture at a small scale on public data. The paper's industrial data and scaling conclusions involve different experimental conditions.
The released `long_view` label follows a business definition and does not directly mean full video completion.

## 实验结果 / Results

| 模型 / Model | Test AUC ↑ | Test LogLoss ↓ | Test NE ↓ |
| --- | ---: | ---: | ---: |
| Wukong | 0.727525 | 0.477824 | 0.892913 |
| HSTU | 0.756088 | 0.462144 | 0.863612 |
| Shallow-hybrid | 0.754102 | 0.463354 | 0.865872 |
| **WHALE** | **0.757566** | **0.459887** | **0.859394** |
| WHALE MeanPool | 0.754784 | 0.462278 | 0.863863 |

五模型使用相同测试成员。此次运行中，WHALE 相对 HSTU 的 AUC 增加 0.001478，NE 相对降低 0.4884%。
NE 表示归一化交叉熵，定义与分组结果见[实验报告](docs/experiment.md)。

All five models use the same test examples. In this run, WHALE improves AUC over HSTU by 0.001478 and reduces normalized entropy (NE) by 0.4884% relatively.
See the [experiment report](docs/experiment.md) for metric definitions and subgroup results.

## 模型 / Models

| 标识符 / Identifier | 中文说明 | Description |
| --- | --- | --- |
| `wukong` | 用户、候选与上下文的特征交互 | User, candidate, and context feature interaction |
| `hstu` | 交替输入历史物品与反馈，在候选位置预测 | Interleaved items and actions; prediction at the candidate position |
| `shallow_hybrid` | 历史摘要输入 Wukong | Pooled HSTU history passed to Wukong |
| `whale` | 每层结合 Wukong、HSTU 和 Attention Fusion | Wukong, HSTU, and attention fusion at each layer |
| `whale_mean_pool` | 用历史均值替代 Attention 读取 | Mean history readout in place of attention readout |

HSTU 最多处理 257 个 Token，其余历史分支最多处理 128 个 Token。
比较采用相同历史行为数量，各模型计算量分别报告。

HSTU processes up to 257 tokens; the other history branches process up to 128.
The models receive the same number of history events, with compute reported separately.

## 复现入口 / Reproduction

配置见 [configs/paper.yaml](configs/paper.yaml)，依赖版本由 [uv.lock](uv.lock) 固定。
已验证环境为 Python 3.13、PyTorch 2.13.0+cu130、RTX 5090 32 GB 和 FP32。
从 [RecFlow 官方仓库](https://github.com/RecFlow-ICLR/RecFlow)取得 37 个 `realshow` 日文件，放入 `data/raw/RecFlow/`。
日期范围为 2024-01-13 至 2024-02-18，文件扩展名为 `.feather`。

Settings are in [configs/paper.yaml](configs/paper.yaml), with dependencies pinned by [uv.lock](uv.lock).
The verified environment uses Python 3.13, PyTorch 2.13.0+cu130, an RTX 5090 32 GB, and FP32.
Obtain the 37 daily `realshow` files from the [official RecFlow repository](https://github.com/RecFlow-ICLR/RecFlow).
Place the `.feather` files dated 2024-01-13 through 2024-02-18 in `data/raw/RecFlow/`.

<details>
<summary>最小运行流程 / Minimal workflow</summary>

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

The preparation script reconstructs target membership from raw files. Training outputs go to `outputs/paper/`.
Each script provides `--help`; [export_public_results.py](scripts/export_public_results.py) exports public aggregates.

</details>

## 研究限制 / Limitations

- 单个种子，测试成员曾用于早期实验；结论限定于当前配置。<br>One seed, with test examples previously used in earlier experiments; conclusions apply to this configuration.
- 历史反馈时间使用项目估计，真实反馈回传时刻未知。<br>History availability uses a project-defined estimate; actual feedback arrival times are unavailable.
- 模型计算量不同，长序列扩展和跨种子稳定性尚未验证。<br>Compute differs across models; long-history scaling and stability across seeds remain unverified.

## 引用与许可 / Citation and License

使用本项目时，请引用 [CITATION.cff](CITATION.cff) 和[原模型与数据论文](docs/research.md#参考文献--references)。
源码与文档采用 [Apache-2.0](LICENSE)，RecFlow 汇总结果采用 [CC BY-SA 4.0](LICENSES/CC-BY-SA-4.0.txt)。
数据、逐条预测、权重和论文 PDF 由使用者在本地管理。第三方署名见 [NOTICE](NOTICE)。

Please cite [CITATION.cff](CITATION.cff) and the [original model and dataset papers](docs/research.md#参考文献--references).
Code and documentation use [Apache-2.0](LICENSE); RecFlow result aggregates use [CC BY-SA 4.0](LICENSES/CC-BY-SA-4.0.txt).
Users manage data, individual predictions, weights, and paper PDFs locally. See [NOTICE](NOTICE) for attribution.

代码与文档使用 AI 辅助编写和复核。模型与数据贡献归属于原作者。

AI assisted code and documentation development and review. Model and dataset contributions belong to their original authors.
