import argparse
import shutil
from pathlib import Path

from whale_reproduction.config import PROJECT_ROOT, load_config
from whale_reproduction.training.reporting import collect_records
from whale_reproduction.utils import read_json, write_json


PUBLIC_FILES = ("comparison.csv", "duration-groups.csv", "id-groups.csv", "deltas.json")
ANALYSIS_FILES = ("baseline-comparison.csv", "group-deltas.csv", "analysis.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出适合公开的汇总结果，不包含逐条预测与成员索引")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="公开汇总目录；默认按配置版本区分")
    parser.add_argument("--include-analysis", action="store_true", help="同时复制已有的新旧比较与派生分析")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if config.schema_version not in (3, 4):
        raise ValueError("公开结果导出需要版本 3 或 4 配置")
    records = collect_records(config)
    public_files = PUBLIC_FILES + (ANALYSIS_FILES if arguments.include_analysis else ())
    source = config.results_directory()
    missing = [name for name in public_files if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"汇总结果缺少文件：{missing}")
    output = arguments.output or Path("results/recflow-official" if config.schema_version == 4 else "results/recflow")
    destination = PROJECT_ROOT / output
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"公开结果目录已有内容：{destination}")
    # 只复制明确列出的汇总文件，运行目录中的用户记录保留在本机。
    destination.mkdir(parents=True, exist_ok=True)
    for name in public_files:
        shutil.copy2(source / name, destination / name)
    run = read_json(config.run_directory("whale") / "run.json")
    context = read_json(config.run_directory("whale") / "config.json")["data_context"]
    environment_fields = ("python", "torch", "torch_cuda", "device", "device_memory_gib", "numpy")
    metadata = {
        "title": "WHALE-RecFlow",
        "experiment_date": run["finished_at"][:10],
        "config": config.snapshot(),
        "environment": {key: run["environment"][key] for key in environment_fields},
        "model_count": len(records),
        "test_count_per_model": config.data.target_samples["test"],
        "source_event_count": context["event_count"],
        "license": "CC-BY-SA-4.0",
        "data_source": "https://github.com/RecFlow-ICLR/RecFlow",
        "scope": "保存的真实实验汇总；逐条预测、成员索引和权重未纳入公开文件。",
    }
    if config.data.uses_official_history:
        metadata["history_protocol"] = context["history_conditions"]
        audit = context["official_history"]
        metadata["official_history"] = {key: audit[key] for key in (
            "source", "archive_name", "archive_bytes", "request_timestamp_unit",
            "raw_pair_count", "raw_duplicate_pair_rows", "official_request_count", "splits", "complete",
        )}
    write_json(destination / "experiment.json", metadata)
    print(f"已导出 {len(public_files) + 1} 份汇总文件至 {destination}")


if __name__ == "__main__":
    main()
