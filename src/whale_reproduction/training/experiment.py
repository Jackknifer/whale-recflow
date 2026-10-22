import shutil
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..config import Config
from ..data.batch import BatchReader
from ..utils import (
    current_timestamp,
    ensure_directory,
    environment_record,
    read_json,
    write_json,
)
from .evaluate import collect_predictions
from .grouping import duration_groups, id_groups, sigmoid
from .metrics import Metrics, compute_metrics
from .checks import verify_checks
from .trainer import (
    BEST_MODEL_NAME,
    RESUME_NAME,
    LogWriter,
    best_epoch_path,
    run_training,
)

FINAL_TEST_PREDICTIONS = "final_test_predictions.parquet"
METRICS_FILE = "metrics.json"
RUN_RECORD_FILE = "run.json"
CONFIG_RECORD_FILE = "config.json"
DURATION_GROUPS_FILE = "duration_groups.json"
TARGET_INDICES_FILE = "target_indices.npz"


def verify_run_context(config: Config, reader: BatchReader, model_name: str) -> dict[str, Any]:
    directory = config.run_directory(model_name)
    context = read_json(directory / CONFIG_RECORD_FILE)
    if context["model"] != model_name or context["config"] != config.snapshot():
        raise ValueError(f"模型 {model_name} 的运行配置与当前配置不一致")
    if context["data_context"] != reader.data_context:
        raise ValueError(f"模型 {model_name} 的数据来源或预处理参数与当前数据不一致")
    reader.verify_target_indices(directory / TARGET_INDICES_FILE)
    return context


def load_model_for_evaluation(
    config: Config, model_name: str, reader: BatchReader, device: torch.device, best_epoch: int
):
    weights = best_epoch_path(config.run_directory(model_name), best_epoch)
    if not weights.is_file():
        raise FileNotFoundError(f"最佳权重不存在 {weights}")
    model = reader.build_model(model_name, config).to(device)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model


def run_final_test(
    config: Config,
    model_name: str,
    reader: BatchReader,
    device: torch.device,
    best_epoch: int,
) -> tuple[Metrics, int, list[dict[str, Any]], list[dict[str, Any]]]:
    run_directory = config.run_directory(model_name)
    prediction_path = run_directory / FINAL_TEST_PREDICTIONS
    rows = reader.target_rows("test")
    if prediction_path.is_file():
        predictions = pd.read_parquet(prediction_path)
        exposure = reader.exposure
        expected_columns = {
            "event_id": exposure.event_id[rows],
            "request_id": exposure.request_id[rows],
            "duration_ms": exposure.duration[rows],
            "label": exposure.long_view[rows],
        }
        for column, values in expected_columns.items():
            if not np.array_equal(predictions[column].to_numpy(), values):
                raise ValueError(f"保存的最终测试预测字段 {column} 与当前目标不一致")
        logits = predictions["logit"].to_numpy(dtype=np.float64)
        labels = predictions["label"].to_numpy(dtype=np.float64)
        event_id = predictions["event_id"].to_numpy(dtype=np.int64)
        request_id = predictions["request_id"].to_numpy(dtype=np.int64)
        duration_ms = predictions["duration_ms"].to_numpy(dtype=np.int64)
        if config.schema_version == 3:
            video_unknown = predictions["video_is_unknown"].to_numpy(dtype=bool)
            author_unknown = predictions["author_is_unknown"].to_numpy(dtype=bool)
            for field, actual in (("video_id", video_unknown), ("author_id", author_unknown)):
                if not np.array_equal(actual, reader.target_unknown("test", field)):
                    raise ValueError(f"保存的 {field} 自然 UNK 标记与当前字典不一致")
    else:
        model = load_model_for_evaluation(config, model_name, reader, device, best_epoch)
        logits, labels, event_id, request_id, duration_ms, video_unknown, author_unknown = collect_predictions(
            model, reader, "test", config.training.batch_size, device
        )
        del model
    expected = int(reader.target_rows("test").size)
    if logits.size != expected:
        raise ValueError(f"测试集预测数量为 {logits.size}，期望 {expected}")
    metrics = compute_metrics(
        logits,
        labels,
        config.evaluation.probability_dtype,
        config.evaluation.single_class_policy,
    )
    probabilities = sigmoid(logits)
    if not prediction_path.is_file():
        temporary = prediction_path.with_name(f".{prediction_path.name}.partial")
        columns = {
                "event_id": event_id.astype(np.int64),
                "request_id": request_id.astype(np.int64),
                "label": labels.astype(np.int8),
                "logit": logits.astype(np.float64),
                "probability": probabilities.astype(np.float64),
                "duration_ms": duration_ms.astype(np.int64),
            }
        if config.schema_version == 3:
            columns.update(video_is_unknown=video_unknown, author_is_unknown=author_unknown)
        pd.DataFrame(columns).to_parquet(temporary, index=False)
        temporary.replace(prediction_path)
    groups = duration_groups(config, labels, logits, duration_ms)
    identifiers = id_groups(labels, logits, video_unknown, author_unknown) if config.schema_version == 3 else []
    if config.schema_version == 3:
        write_json(run_directory / "id_groups.json", {"model": model_name, "groups": identifiers})
    write_json(
        run_directory / DURATION_GROUPS_FILE,
        {"model": model_name, "groups": groups},
    )
    with LogWriter(run_directory / "final_test.log") as log:
        log.write(
            f"最终测试完成：样本 {metrics.count} 条，AUC {metrics.auc:.6f}，"
            f"LogLoss {metrics.logloss:.6f}，NE {metrics.ne:.6f}，"
            f"时长分组 {len(groups)} 组"
        )
    return metrics, expected, groups, identifiers


def _should_resume(run_directory, resume: bool) -> bool:
    if not resume:
        return False
    return (run_directory / RESUME_NAME).is_file()


def run_single_model(
    config: Config,
    model_name: str,
    reader: BatchReader,
    device: torch.device,
    resume: bool,
    check_results: dict[str, Any],
) -> dict[str, Any]:
    run_directory = config.run_directory(model_name)
    ensure_directory(run_directory)
    record_path = run_directory / RUN_RECORD_FILE
    if record_path.is_file():
        existing = read_json(record_path)
        if existing.get("status") == "completed":
            raise RuntimeError(
                f"模型 {model_name} 已完成正式训练，记录位于 {record_path}；"
                "当前阶段不重复训练"
            )
    config_path = run_directory / CONFIG_RECORD_FILE
    if config_path.is_file():
        started_at = verify_run_context(config, reader, model_name)["started_at"]
        if not resume:
            raise RuntimeError("运行目录已经存在配置记录，继续执行需要显式指定 --resume")
    else:
        if any((run_directory / name).exists() for name in (BEST_MODEL_NAME, RESUME_NAME, FINAL_TEST_PREDICTIONS)):
            raise RuntimeError("运行目录存在权重或预测，但缺少对应配置记录")
        started_at = current_timestamp()
        reader.save_target_indices(run_directory / TARGET_INDICES_FILE)
        write_json(
            config_path,
            {
                "model": model_name,
                "config": config.snapshot(),
                "data_context": reader.data_context,
                "environment": environment_record(device),
                "started_at": started_at,
            },
        )
    if (run_directory / FINAL_TEST_PREDICTIONS).is_file() and not (run_directory / RESUME_NAME).is_file():
        raise FileNotFoundError("已有最终测试预测但缺少训练恢复状态，不能重新训练")
    outcome = run_training(
        config,
        model_name,
        reader,
        device,
        run_directory,
        resume=_should_resume(run_directory, resume),
    )
    test_metrics, test_count, groups, identifiers = run_final_test(
        config, model_name, reader, device, outcome.best_epoch
    )
    step = check_results["models"][model_name]["training_step"]
    parameters = {
        key: int(step[key]) for key in (
            "total_parameters", "embedding_parameters", "trainable_parameters", "gradient_parameter_count"
        )
    }
    record = {
        "schema_version": config.schema_version,
        "model": model_name,
        "seed": config.seed,
        "status": "completed",
        "started_at": started_at,
        "finished_at": current_timestamp(),
        "environment": environment_record(device),
        "batch_size": config.training.batch_size,
        "max_epochs": config.training.max_epochs,
        "resumed_from_epoch": outcome.resumed_from_epoch,
        "best_epoch": outcome.best_epoch,
        "stopped_early": outcome.stopped_early,
        "epochs_completed": len(outcome.history),
        "best_validation": outcome.best_validation.to_json(),
        "final_test": test_metrics.to_json(),
        "final_test_count": test_count,
        "train_seconds": outcome.train_seconds,
        "validation_seconds": outcome.validation_seconds,
        "peak_memory_gib": outcome.peak_memory_bytes / 1024**3,
        "history": [item.to_json() for item in outcome.history],
        "parameters": parameters,
        "duration_groups": groups,
    }
    if config.schema_version == 3:
        record["id_groups"] = identifiers
        record["compute"] = check_results["models"][model_name]["compute"]
        record["training_protocol"] = config.snapshot()["training"]
    write_json(
        run_directory / METRICS_FILE,
        {
            "model": model_name,
            "best_epoch": outcome.best_epoch,
            "validation": outcome.best_validation.to_json(),
            "test": test_metrics.to_json(),
            "test_count": test_count,
        },
    )
    best_path = run_directory / BEST_MODEL_NAME
    temporary = best_path.with_name(f".{best_path.name}.partial")
    shutil.copyfile(best_epoch_path(run_directory, outcome.best_epoch), temporary)
    temporary.replace(best_path)
    write_json(record_path, record)
    resume_path = run_directory / RESUME_NAME
    if resume_path.is_file():
        resume_path.unlink()
    for epoch in range(config.training.max_epochs):
        path = best_epoch_path(run_directory, epoch)
        if path.is_file():
            path.unlink()
    return record


def run_all_models(
    config: Config, reader: BatchReader, device: torch.device, resume: bool
) -> list[dict[str, Any]]:
    if config.experiment.parallel_runs != 1:
        raise ValueError("正式实验要求顺序执行，parallel_runs 必须为 1")
    checks = verify_checks(config, reader, device)
    records: list[dict[str, Any]] = []
    for model_name in config.model_names:
        record_path = config.run_directory(model_name) / RUN_RECORD_FILE
        if record_path.is_file() and read_json(record_path).get("status") == "completed":
            verify_run_context(config, reader, model_name)
            records.append(read_json(record_path))
            continue
        records.append(run_single_model(config, model_name, reader, device, resume, checks))
    return records
