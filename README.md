# WHALE-RecFlow

基于 PyTorch，在 RecFlow `realshow` 上独立复现 WHALE 的逐层融合结构。
研究非序列特征交互与用户历史序列的联合建模，监督目标为发布的 `long_view` 业务反馈。
项目包含数据加工、五模型比较、训练恢复、最终评价和汇总结果。

| 模型 | 测试 AUC ↑ | 测试 LogLoss ↓ | 测试 NE ↓ |
| --- | ---: | ---: | ---: |
| Wukong | 0.727525 | 0.477824 | 0.892913 |
| HSTU | 0.756088 | 0.462144 | 0.863612 |
| Shallow-hybrid | 0.754102 | 0.463354 | 0.865872 |
| **WHALE** | **0.757566** | **0.459887** | **0.859394** |
| WHALE MeanPool | 0.754784 | 0.462278 | 0.863863 |

结果来自种子 42 的一次正式运行，五模型使用相同测试成员。
WHALE 相对 HSTU 的 AUC 增加 0.001478，NE 相对降低 0.4884%。
完整结果与限制见 [实验报告](docs/experiment.md)，机器表格见 [results/recflow](results/recflow)。

## 模型与比较

| 模型标识符 | 计算方式 |
| --- | --- |
| `wukong` | 对候选、用户与上下文特征执行 Wukong 交互 |
| `hstu` | 将历史物品、反馈交替编码，并在候选位置预测 |
| `shallow_hybrid` | HSTU 编码历史，均值摘要作为额外 Token 输入 Wukong |
| `whale` | 每层执行 Wukong、HSTU 和 Attention Fusion |
| `whale_mean_pool` | 保留逐层结构，使用投影后历史均值替代 Attention 读取 |

HSTU 最多处理 257 个 Token，其余历史分支最多处理 128 个 Token。
模型使用相同历史行为数量，计算量存在差异。
公式、输入映射和论文适配见 [模型设计](docs/research.md)。

## 环境安装

需要 Python 3.13 和 [uv](https://docs.astral.sh/uv/)。
正式训练使用 NVIDIA GPU、CUDA 和 FP32。
已验证环境为 RTX 5090 32 GB、PyTorch 2.13.0+cu130 和 CUDA Runtime 13.0。
本轮峰值显存约 10.43 GiB；其他设备的完整训练时间尚未测量。

在仓库根目录执行：

```bash
mkdir -p .work/runtime .work/uv-cache
export TMPDIR="$PWD/.work/runtime"
export UV_CACHE_DIR="$PWD/.work/uv-cache"
uv sync --frozen --no-dev
```

`uv.lock` 固定依赖解析结果。
项目通过 `src/whale_reproduction/` 安装 Python 包，导入名称为 `whale_reproduction`。

## 数据准备

从 [RecFlow 官方仓库](https://github.com/RecFlow-ICLR/RecFlow)取得 `realshow` 数据。
[作者确认的 Hugging Face 来源](https://github.com/RecFlow-ICLR/RecFlow/issues/7)也提供对应文件。
数据按原发布条款使用，仓库中保留来源署名。

将 37 个日文件直接放入 `data/raw/RecFlow/`。
文件名范围为 `2024-01-13.feather` 至 `2024-02-18.feather`。
加工入口检查日期、字段、反馈取值和原始行数。
`long_view` 表示业务反馈，不能直接解释为严格完播。

首次从原始数据复现：

```bash
uv run python scripts/prepare_data.py --config configs/paper.yaml --baseline-config configs/local.yaml
```

该命令按版本 2 的规则重建原成员，再扩充训练目标。
无需取得既有实验的索引，也无需训练原基线。
重建成员已经在完整真实数据上核验，与保存的两轮成员逐项一致。
目标数量为训练 3,000,000 条、验证 100,000 条、测试 200,000 条。
加工结果写入 `data/processed/recflow-paper/`，已有目录含有内容时立即终止。

持有原基线索引的使用者可以省去 `--baseline-config`。
此时程序读取配置中的 `baseline_indices`，并核对来源清单、行数和事件成员。

## 训练与评价

```bash
uv run python scripts/run_experiment.py --config configs/paper.yaml --phase check --hourly-rate 2.99
uv run python -u scripts/run_experiment.py --config configs/paper.yaml --phase train
uv run python scripts/export_results.py --config configs/paper.yaml
```

`--hourly-rate` 按实际服务器单价填写，仅用于检查阶段估算。
检查覆盖五模型的真实批次训练、梯度、显存和测速。
正式阶段顺序训练五模型，各遍历训练目标一次，并执行验证和最终测试。
日期升序、日内打乱；Batch Size 为 256，每模型更新 11,719 次。
最后一批包含 192 条目标。

运行日志每 100 次更新打印平均训练损失。
中断后使用同一命令并增加 `--resume`，恢复边界为已经完成的 Epoch。
已完成模型核对配置后复用记录。
同次最终测试预测用于整体、时长和自然 UNK 分组评价。

训练输出保存在 `outputs/paper/`。
将整体和分组汇总导出至公开目录：

```bash
uv run python scripts/export_public_results.py --config configs/paper.yaml --output results/reproduced
```

目标目录已有文件时，导出程序立即终止。
仓库自带的 `results/recflow/` 属于 2026-10-22 实验，新汇总保存在所指定的独立目录。

## 目录用途

| 路径 | 内容 |
| --- | --- |
| `src/whale_reproduction/data/` | 数据读取、成员抽样、训练字典、历史索引和批次 |
| `src/whale_reproduction/models/` | 输入编码、Wukong、HSTU、Fusion 和五模型 |
| `src/whale_reproduction/training/` | 检查、训练、恢复、评价和结果导出 |
| `scripts/` | 数据准备、正式运行与导出入口 |
| `configs/paper.yaml` | 本轮单轮训练协议，版本 3 |
| `configs/local.yaml` | 原基线协议，版本 2；同时用于重建基线成员 |
| `results/recflow/` | 可公开的整体指标、分组、比较和实验元数据 |
| `docs/research.md` | 模型公式、数据假设、适配说明和论文引用 |
| `docs/experiment.md` | 实验结果、计算成本、分组分析和研究限制 |

`data/`、`outputs/`、`references/` 和 `.work/` 用于本机资料管理，已从 Git 排除。
逐条预测、用户与事件索引、字典、训练权重和论文 PDF 均留在本机。
公开结果只包含汇总数值。

## 结果的适用范围

当前实验支持当前数据和模型规模下的比较。
只运行一个种子，测试成员此前已经用于原基线评价。
本项目没有确认跨种子稳定性和统计显著性。
统一宽度、深度与历史数量仍会产生不同计算量。
公开数据适配及原论文工业实验条件分别记录。

代码与文档使用 AI 辅助编写和复核。
模型设计与数据资源的贡献归属于对应论文作者。
研究与个人经历的表述应以已完成代码、真实运行和具体参与内容为依据。

## 许可证与引用

源码和文档采用 [Apache-2.0](LICENSE)。
`results/` 的 RecFlow 汇总结果采用 [CC BY-SA 4.0](LICENSES/CC-BY-SA-4.0.txt)。
原始数据遵循 RecFlow 的发布条款，第三方来源见 [NOTICE](NOTICE)。

软件引用信息见 [CITATION.cff](CITATION.cff)。
使用本项目开展研究时，请同时引用 WHALE、Wukong、HSTU 和 RecFlow 原论文。
