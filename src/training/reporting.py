from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import current_timestamp, ensure_directory, read_json, write_json
from .grouping import build_deltas

COMPARISON_FILE = "comparison.csv"
DURATION_FILE = "duration-groups.csv"
SUMMARY_FILE = "summary.json"
PRIMARY_MODEL = "whale"
COMPARISON_COLUMNS = (
    "model",
    "auc",
    "logloss",
    "ne",
    "best_epoch",
    "epochs_completed",
    "stopped_early",
    "train_seconds",
    "validation_seconds",
    "total_parameters",
    "embedding_parameters",
    "trainable_parameters",
    "gradient_parameter_count",
    "peak_memory_gib",
    "test_count",
)
DURATION_COLUMNS = (
    "model",
    "group",
    "lower_seconds",
    "upper_seconds",
    "count",
    "positive_count",
    "positive_rate",
    "auc",
    "logloss",
    "ne",
    "notes",
)


def _record_path(config: Config, model_name: str) -> Path:
    return config.run_directory(model_name) / "run.json"


def collect_records(config: Config) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    reference_context = None
    reference_indices = None
    for model_name in config.model_names:
        path = _record_path(config, model_name)
        if not path.is_file():
            raise FileNotFoundError(
                f"模型 {model_name} 缺少运行记录 {path}，无法导出结果"
            )
        record = read_json(path)
        if record.get("status") != "completed":
            raise ValueError(
                f"模型 {model_name} 的运行记录状态为 {record.get('status')}"
            )
        context = read_json(path.parent / "config.json")
        if context["model"] != model_name or context["config"] != config.snapshot():
            raise ValueError(f"模型 {model_name} 的配置与当前比较设置不一致")
        if record["model"] != model_name or record["seed"] != config.seed:
            raise ValueError(f"模型 {model_name} 的运行身份与配置不一致")
        if reference_context is None:
            reference_context = context["data_context"]
        elif reference_context != context["data_context"]:
            raise ValueError("比较模型的数据来源或预处理参数不一致")
        with np.load(path.parent / "target_indices.npz", allow_pickle=False) as indices:
            if reference_indices is None:
                reference_indices = {name: indices[name].copy() for name in indices.files}
            elif set(indices.files) != set(reference_indices) or any(
                not np.array_equal(indices[name], reference_indices[name]) for name in indices.files
            ):
                raise ValueError("比较模型的完整目标索引或事件编号不一致")
        if record["final_test_count"] != config.data.target_samples["test"]:
            raise ValueError(f"模型 {model_name} 的最终测试数量与配置不一致")
        final_test = record["final_test"]
        parameters = record["parameters"]
        records.append(
            {
                "model": model_name,
                "auc": float(final_test["auc"]),
                "logloss": float(final_test["logloss"]),
                "ne": float(final_test["ne"]),
                "best_epoch": int(record["best_epoch"]),
                "epochs_completed": int(record["epochs_completed"]),
                "stopped_early": bool(record["stopped_early"]),
                "train_seconds": float(record["train_seconds"]),
                "validation_seconds": float(record["validation_seconds"]),
                "total_parameters": int(parameters["total_parameters"]),
                "embedding_parameters": int(parameters["embedding_parameters"]),
                "trainable_parameters": int(parameters["trainable_parameters"]),
                "gradient_parameter_count": int(parameters["gradient_parameter_count"]),
                "peak_memory_gib": float(record["peak_memory_gib"]),
                "test_count": int(record["final_test_count"]),
                "duration_groups": record["duration_groups"],
                "id_groups": record["id_groups"],
                "compute": record["compute"],
            }
        )
    return records


def build_duration_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        for group in record["duration_groups"]:
            rows.append(
                {
                    "model": record["model"],
                    "group": group["group"],
                    "lower_seconds": group["lower_seconds"],
                    "upper_seconds": group["upper_seconds"],
                    "count": group["count"],
                    "positive_count": group["positive_count"],
                    "positive_rate": group["positive_rate"],
                    "auc": group["auc"],
                    "logloss": group["logloss"],
                    "ne": group["ne"],
                    "notes": "; ".join(group["notes"]),
                }
            )
    if not rows:
        raise ValueError("运行记录中没有任何时长分组结果")
    return pd.DataFrame(rows)[list(DURATION_COLUMNS)]


def build_comparison(config: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    records = collect_records(config)
    frame = pd.DataFrame(records)[list(COMPARISON_COLUMNS)]
    frame["matrix_forward_flops_per_sample"] = [record["compute"]["matrix_forward_flops_per_sample"] for record in records]
    frame["history_behaviors"] = config.data.max_history_length
    frame["max_sequence_tokens"] = [record["compute"]["max_sequence_tokens"] for record in records]
    directory = config.results_directory()
    ensure_directory(directory)
    frame.to_csv(directory / COMPARISON_FILE, index=False, float_format="%.17g")

    duration_frame = build_duration_frame(records)
    duration_frame.to_csv(
        directory / DURATION_FILE, index=False, float_format="%.17g"
    )

    metrics_by_model = {
        record["model"]: {
            "auc": record["auc"],
            "logloss": record["logloss"],
            "ne": record["ne"],
        }
        for record in records
    }
    baselines = ("wukong", "hstu")
    deltas = build_deltas(metrics_by_model, PRIMARY_MODEL, baselines)

    summary = {
        "created_at": current_timestamp(),
        "models": [record["model"] for record in records],
        "reference": PRIMARY_MODEL,
        "baselines": list(baselines),
        "deltas": deltas,
        "rows": records,
    }
    id_rows = [{"model": record["model"], **group, "notes": "; ".join(group["notes"])} for record in records for group in record["id_groups"]]
    pd.DataFrame(id_rows).to_csv(directory / "id-groups.csv", index=False, float_format="%.17g")
    context = read_json(config.run_directory(PRIMARY_MODEL) / "config.json")
    summary["data_statistics"] = context["data_context"]["split_statistics"]
    summary["config"] = config.snapshot()
    summary["history_protocol"] = context["data_context"]["history_conditions"]
    write_json(directory / SUMMARY_FILE, summary)
    return frame, summary
