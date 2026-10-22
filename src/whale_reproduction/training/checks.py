import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import Config
from ..data.batch import BatchReader
from ..utils import (
    current_timestamp,
    ensure_directory,
    environment_record,
    read_json,
    save_checkpoint,
    seed_everything,
    write_json,
)
from .trainer import (
    BEST_MODEL_NAME,
    LogWriter,
    build_loss,
    build_optimizer,
    parameter_report,
)
from .compute import forward_compute

WARMUP_STEPS = 3
TIMED_STEPS = 8
EVALUATION_WARMUP_BATCHES = 3
EVALUATION_TIMED_BATCHES = 8
DEFAULT_OVERHEAD_HOURS = 1.0
MAX_TOTAL_COST_CNY = 100.0
TARGET_INDICES_FILE = "target_indices.npz"


def _representative_batch(config: Config, reader: BatchReader, split: str):
    rows = reader.target_rows(split)
    all_positions = np.arange(rows.size, dtype=np.int64)
    counts = reader.history_counts(split, all_positions)
    positions = all_positions[: config.training.batch_size].copy()
    for index, condition in enumerate((counts == 0, (counts > 0) & (counts < reader.max_history_length), counts == reader.max_history_length)):
        matched = np.flatnonzero(condition)
        if matched.size and index < positions.size:
            positions[index] = matched[0]
    positions = np.unique(positions)
    return positions, reader.build_batch(rows[positions], reader.history_rows(split, positions))


def check_data(
    config: Config, reader: BatchReader, log: LogWriter
) -> dict[str, Any]:
    exposure = reader.exposure
    positions, batch = _representative_batch(config, reader, "train")
    rows = reader.target_rows("train")[positions]
    history_slot = reader.history_rows("train", positions)
    counts = reader.history_counts("train", positions)
    valid = history_slot >= 0
    safe = np.maximum(history_slot, 0)
    target_user = exposure.user_id[rows].astype(np.int64)
    target_time = exposure.request_timestamp[rows]
    target_request = exposure.request_id[rows].astype(np.int64)
    history_user = np.where(valid, exposure.user_id[safe].astype(np.int64), -1)
    history_time = np.where(valid, exposure.request_timestamp[safe], 0)
    history_request = np.where(valid, exposure.request_id[safe].astype(np.int64), -1)
    history_available = np.where(
        valid, reader.event_available_ms[safe], 0
    )
    user_broadcast = np.broadcast_to(target_user[:, None], valid.shape)
    time_broadcast = np.broadcast_to(target_time[:, None], valid.shape)
    request_broadcast = np.broadcast_to(target_request[:, None], valid.shape)
    if not (history_user[valid] == user_broadcast[valid]).all():
        raise ValueError("历史检查失败：存在属于其他用户的历史事件")
    if not (history_time[valid] < time_broadcast[valid]).all():
        raise ValueError("历史检查失败：存在请求时间不早于目标请求时间的历史事件")
    if not (history_available[valid] < time_broadcast[valid]).all():
        raise ValueError("历史检查失败：存在可用时间不早于目标请求时间的历史事件")
    if not (history_request[valid] != request_broadcast[valid]).all():
        raise ValueError("历史检查失败：存在与目标同请求的历史事件")
    if counts.max() > reader.max_history_length:
        raise ValueError(
            f"历史长度 {int(counts.max())} 超过上限 {reader.max_history_length}"
        )
    if counts.min() < 0:
        raise ValueError("历史长度出现负值")

    if not np.array_equal(valid.sum(axis=1), counts):
        raise ValueError("历史有效位置数量与保存的长度不一致")
    if not np.array_equal(valid, np.arange(reader.max_history_length)[None, :] < counts[:, None]):
        raise ValueError("历史有效位置没有位于左侧或 PAD 没有位于右侧")
    expected_shape = (batch.size, reader.max_history_length, reader.max_history_length)
    if batch.causal_mask.shape != expected_shape:
        raise ValueError(f"因果掩码形状为 {batch.causal_mask.shape}，与设计不一致")
    if batch.time_bucket.shape != expected_shape:
        raise ValueError(f"时间分桶形状为 {batch.time_bucket.shape}，与设计不一致")
    if batch.causal_mask.dtype != np.bool_:
        raise TypeError("因果掩码需要 bool 类型")
    if batch.time_bucket.min() < 0 or batch.time_bucket.max() > 128:
        raise ValueError("时间分桶超出 0 到 128 的范围")
    for name, values in (
        ("duration", batch.duration),
        ("request_time", batch.request_time),
        ("video_age", batch.video_age),
        ("history_age", batch.history_age),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} 数值组含有非有限值")
    if batch.history_age.shape != (batch.size, reader.max_history_length):
        raise ValueError(f"历史年龄形状为 {batch.history_age.shape}，与设计不一致")
    padding = ~batch.history_mask
    if padding.any():
        for name, values in batch.history_video.items():
            if (values[padding] != 0).any():
                raise ValueError(f"PAD 位置的历史视频属性 {name} 不是 0")
        for name, values in batch.history_feedback.items():
            if (values[padding] != 0).any():
                raise ValueError(f"PAD 位置的历史反馈 {name} 不是 0")
        if (batch.history_age[padding] != 0).any():
            raise ValueError("PAD 位置的历史年龄不是 0")
    if batch.causal_mask.sum() == 0:
        raise ValueError("因果掩码全部为 False")
    for name, values in batch.categorical.items():
        if values.ndim != 1 or values.size != batch.size:
            raise ValueError(f"类别字段 {name} 的形状为 {values.shape}，与批次不一致")
        if values.min() < 0:
            raise ValueError(f"类别字段 {name} 出现负的编码")
    if batch.labels.dtype != np.float32:
        raise TypeError(f"标签类型为 {batch.labels.dtype}，需要 float32")
    if not np.isin(np.unique(batch.labels), (0.0, 1.0)).all():
        raise ValueError("标签出现非二值取值")

    log.write(
        f"数据检查通过：批次 {batch.size} 条，历史长度 {int(counts.min())} 到 "
        f"{int(counts.max())}，空历史 {int((counts == 0).sum())} 条"
    )
    return {
        "batch_size": int(batch.size),
        "history_min": int(counts.min()),
        "history_max": int(counts.max()),
        "empty_history": int((counts == 0).sum()),
        "full_history": int((counts >= reader.max_history_length).sum()),
        "causal_mask_true": int(batch.causal_mask.sum()),
        "time_bucket_max": int(batch.time_bucket.max()),
        "padding_positions": int(padding.sum()),
        "same_user_verified": True,
        "different_request_verified": True,
        "strict_request_time_verified": True,
        "strict_availability_verified": True,
    }


def check_training_step(
    config: Config,
    reader: BatchReader,
    model_name: str,
    device: torch.device,
    log: LogWriter,
) -> dict[str, Any]:
    seed_everything(config.seed)
    model = reader.build_model(model_name, config).to(device)
    optimizer = build_optimizer(config, model)
    loss_function = build_loss(config)
    _, batch = _representative_batch(config, reader, "train")
    if config.schema_version == 3:
        batch = reader.mask_training_batch(batch, 0, 0)
    tensors = batch.as_tensors(device)
    ranking_report = check_ranking_path(model, tensors) if model_name == "hstu" and config.schema_version == 3 else None
    labels = torch.from_numpy(batch.labels).to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(tensors)
    if logits.shape != (batch.size,):
        raise ValueError(f"{model_name} 输出形状为 {tuple(logits.shape)}")
    loss = loss_function(logits, labels)
    if not torch.isfinite(loss):
        raise ValueError(f"{model_name} 损失不是有限值")
    loss.backward()
    unknown_gradients = {}
    if config.schema_version == 3:
        for field in config.training.id_mask_fields:
            gradient = model._embedding._categorical[field].weight.grad
            norm = float(gradient[1].norm()) if gradient is not None else 0.0
            if norm <= 0.0 or not np.isfinite(norm):
                raise ValueError(f"{model_name} 的 {field} UNK 行没有有效梯度")
            unknown_gradients[field] = norm
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config.training.gradient_clip_norm
    )
    if not torch.isfinite(gradient_norm):
        raise ValueError(f"{model_name} 梯度范数不是有限值")
    snapshot = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith(("_head.", "_ranking_input.")) and parameter.grad is not None
    }
    optimizer.step()
    updated = 0
    unchanged = 0
    for name, parameter in model.named_parameters():
        if name not in snapshot:
            continue
        if torch.equal(snapshot[name], parameter.detach()):
            unchanged += 1
        else:
            updated += 1
    if updated == 0:
        raise ValueError(f"{model_name} 没有任何参数完成更新")
    report = parameter_report(model)
    with_gradient = sum(
        1 for parameter in model.parameters() if parameter.grad is not None
    )
    gradient_parameters = sum(
        int(parameter.grad.numel())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    report.update(
        {
            "model": model_name,
            "loss": float(loss.detach()),
            "gradient_norm": float(gradient_norm),
            "updated_parameter_tensors": int(updated),
            "unchanged_parameter_tensors": int(unchanged),
            "parameters_with_gradient": int(with_gradient),
            "gradient_parameter_count": int(gradient_parameters),
            "update_scope": "prediction_head_and_ranking_inputs" if ranking_report is not None else "prediction_head",
            "ranking_input": ranking_report,
            "unknown_row_gradient_norm": unknown_gradients,
            "compute": forward_compute(model, tensors, reader.max_history_length),
        }
    )
    log.write(
        f"{model_name} 训练步骤检查通过：损失 {float(loss.detach()):.6f}，"
        f"梯度范数 {float(gradient_norm):.6f}，更新 {updated} 组，未变 {unchanged} 组，"
        f"获得梯度的参数张量 {with_gradient} 个"
    )
    return report


def check_ranking_path(model, tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    with torch.no_grad():
        ranking = model._ranking_input(tensors, model._embedding)
    counts = tensors["history_mask"].sum(dim=1)
    if not torch.equal(ranking.candidate_positions, 2 * counts):
        raise ValueError("HSTU 候选位置与历史数量不一致")
    if not torch.equal(ranking.mask.sum(dim=1), 2 * counts + 1):
        raise ValueError("HSTU 有效 Token 数量不符")
    length = ranking.mask.size(1)
    invalid = ~ranking.mask
    if ranking.causal_mask[invalid].any() or (ranking.causal_mask & invalid[:, None, :]).any():
        raise ValueError("HSTU 的 PAD 位置参与历史读取")
    if torch.triu(ranking.causal_mask, diagonal=1).any():
        raise ValueError("HSTU 历史位置读取了后续 Token")
    rows = torch.arange(counts.size(0), device=counts.device)
    candidate_mask = ranking.causal_mask[rows, ranking.candidate_positions]
    if not torch.equal(candidate_mask, ranking.mask):
        raise ValueError("HSTU 候选没有读取全部有效历史及自身")
    history_time = tensors["history_time_ms"].cpu().numpy()
    target_time = tensors["target_time_ms"].cpu().numpy()
    actual = ranking.time_bucket.cpu().numpy()
    for sample in range(counts.size(0)):
        size = int(counts[sample])
        sequence_time = np.full(length + 1, target_time[sample], dtype=np.int64)
        sequence_time[:2 * size] = np.repeat(history_time[sample, :size], 2)
        difference = np.abs(sequence_time[1:, None] - sequence_time[None, :-1])
        expected = np.clip((np.log(np.maximum(difference, 1)) / 0.301).astype(np.int64), 0, 128)
        if not np.array_equal(actual[sample], expected):
            raise ValueError("HSTU 下一事件时间分桶与原始毫秒时间不一致")
    return {"max_sequence_tokens": length, "empty_history": int((counts == 0).sum()), "candidate_positions_verified": True, "next_event_time_verified": True, "causality_and_padding_verified": True}


def _training_step(
    model,
    optimizer,
    loss_function,
    batch,
    device: torch.device,
    config: Config,
) -> None:
    tensors = batch.as_tensors(device)
    labels = torch.from_numpy(batch.labels).to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(tensors)
    loss = loss_function(logits, labels)
    if not torch.isfinite(loss):
        raise ValueError("测速训练出现非有限损失")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm, error_if_nonfinite=True)
    optimizer.step()


def measure_training_speed(
    config: Config,
    reader: BatchReader,
    model_name: str,
    device: torch.device,
    log: LogWriter,
) -> dict[str, Any]:
    seed_everything(config.seed)
    model = reader.build_model(model_name, config).to(device)
    optimizer = build_optimizer(config, model)
    loss_function = build_loss(config)
    if reader.num_batches("train", config.training.batch_size) < WARMUP_STEPS + TIMED_STEPS:
        raise ValueError("训练集合批次数量不足，无法完成训练测速")
    batches = iter(reader.iter_batches("train", config.training.batch_size, shuffle=config.schema_version == 3, generator=np.random.Generator(np.random.PCG64(config.seed)), training=True))
    model.train()
    for index in range(WARMUP_STEPS):
        _training_step(model, optimizer, loss_function, next(batches), device, config)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(TIMED_STEPS):
        _training_step(model, optimizer, loss_function, next(batches), device, config)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated())
    seconds_per_batch = elapsed / TIMED_STEPS
    samples_per_second = config.training.batch_size / seconds_per_batch
    log.write(
        f"{model_name} 训练测速：每批 {seconds_per_batch:.4f} 秒，"
        f"{samples_per_second:.2f} 样本/秒，峰值显存 {peak / 1024**3:.2f} GiB"
    )
    return {
        "warmup_steps": WARMUP_STEPS,
        "timed_steps": TIMED_STEPS,
        "seconds_per_batch": seconds_per_batch,
        "samples_per_second": samples_per_second,
        "peak_memory_bytes": peak,
        "peak_memory_gib": peak / 1024**3,
        "timing_scope": "batch_preparation_transfer_forward_backward_optimizer",
    }


def measure_evaluation_speed(
    config: Config,
    reader: BatchReader,
    model_name: str,
    device: torch.device,
    log: LogWriter,
) -> dict[str, Any]:
    model = reader.build_model(model_name, config).to(device)
    model.eval()
    if reader.num_batches("validation", config.training.batch_size) < EVALUATION_WARMUP_BATCHES + EVALUATION_TIMED_BATCHES:
        raise ValueError("验证集合批次数量不足，无法完成评估测速")
    batches = iter(reader.iter_batches("validation", config.training.batch_size, shuffle=False, generator=None))
    with torch.no_grad():
        for index in range(EVALUATION_WARMUP_BATCHES):
            model(next(batches).as_tensors(device))
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(EVALUATION_TIMED_BATCHES):
            logits = model(next(batches).as_tensors(device))
            if not torch.isfinite(logits).all():
                raise ValueError("评估测速出现非有限预测")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    seconds_per_batch = elapsed / EVALUATION_TIMED_BATCHES
    samples_per_second = config.training.batch_size / seconds_per_batch
    log.write(
        f"{model_name} 评估测速：每批 {seconds_per_batch:.4f} 秒，"
        f"{samples_per_second:.2f} 样本/秒"
    )
    return {
        "timed_batches": EVALUATION_TIMED_BATCHES,
        "seconds_per_batch": seconds_per_batch,
        "samples_per_second": samples_per_second,
        "timing_scope": "batch_preparation_transfer_forward",
    }


def check_weight_roundtrip(
    config: Config,
    reader: BatchReader,
    device: torch.device,
    directory: Path,
    log: LogWriter,
    model_name: str = "whale",
) -> dict[str, Any]:
    seed_everything(config.seed)
    model = reader.build_model(model_name, config).to(device)
    path = directory / f"{model_name}-{BEST_MODEL_NAME}"
    _, batch = _representative_batch(config, reader, "validation")
    model.eval()
    with torch.no_grad():
        logits_before = model(batch.as_tensors(device)).to("cpu", torch.float64).numpy()
    save_checkpoint(model.state_dict(), path)
    del model
    reloaded = reader.build_model(model_name, config).to(device)
    reloaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    reloaded.eval()
    with torch.no_grad():
        logits_after = reloaded(batch.as_tensors(device)).to("cpu", torch.float64).numpy()
    if not np.isfinite(logits_before).all() or not np.isfinite(logits_after).all():
        raise ValueError("权重保存加载检查出现非有限预测")
    difference = float(np.max(np.abs(logits_before - logits_after)))
    if difference > 1e-6:
        raise ValueError(f"权重保存与加载前后的预测最大差异为 {difference}，超出容差")
    path.unlink()
    log.write(
        f"{model_name} 权重保存加载检查通过：真实样本 {batch.size} 条，最大差异 {difference:.3e}"
    )
    return {
        "max_logit_difference": difference,
        "sample_count": batch.size,
    }


def estimate_memory(config: Config, reader: BatchReader, model_name: str) -> dict[str, Any]:
    model = reader.build_model(model_name, config)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    weights_bytes = total * 4
    gradients_bytes = trainable * 4
    optimizer_bytes = trainable * 8
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "weights_bytes": int(weights_bytes),
        "gradients_bytes": int(gradients_bytes),
        "optimizer_state_bytes": int(optimizer_bytes),
        "base_bytes": int(weights_bytes + gradients_bytes + optimizer_bytes),
        "base_gib": (weights_bytes + gradients_bytes + optimizer_bytes) / 1024**3,
    }


def estimate_cost(config: Config, results: dict[str, Any]) -> dict[str, Any]:
    batch_size = config.training.batch_size
    epochs = config.training.max_epochs
    train_batches = -(-int(config.data.target_samples["train"]) // batch_size)
    validation_batches = -(-int(config.data.target_samples["validation"]) // batch_size)
    test_batches = -(-int(config.data.target_samples["test"]) // batch_size)
    models: dict[str, Any] = {}
    total_seconds = 0.0
    for name, entry in results["models"].items():
        train_time = entry["training_speed"]["seconds_per_batch"]
        eval_time = entry["evaluation_speed"]["seconds_per_batch"]
        seconds = (
            epochs * train_batches * train_time
            + (epochs * validation_batches + test_batches) * eval_time
        )
        models[name] = {
            "training_seconds": seconds,
            "training_hours": seconds / 3600.0,
            "peak_memory_gib": entry["training_speed"]["peak_memory_gib"],
            "train_batches_per_epoch": train_batches,
            "validation_batches_per_epoch": validation_batches,
            "test_batches": test_batches,
            "training_updates": epochs * train_batches,
            "evaluation_batches": epochs * validation_batches + test_batches,
        }
        total_seconds += seconds
    hourly_rate = float(results["hourly_rate_cny"])
    overhead_hours = float(results.get("overhead_hours", DEFAULT_OVERHEAD_HOURS))
    total_hours = total_seconds / 3600.0 + overhead_hours
    return {
        "batch_size": batch_size,
        "epochs": epochs,
        "train_batches_per_epoch": train_batches,
        "validation_batches_per_epoch": validation_batches,
        "test_batches": test_batches,
        "models": models,
        "total_training_seconds": total_seconds,
        "total_training_hours": total_seconds / 3600.0,
        "overhead_hours": overhead_hours,
        "total_hours_with_overhead": total_hours,
        "hourly_rate_cny": hourly_rate,
        "estimated_cost_cny": total_hours * hourly_rate,
    }


def run_checks(
    config: Config,
    reader: BatchReader,
    device: torch.device,
    hourly_rate: float | None = None,
) -> dict[str, Any]:
    if hourly_rate is None or not np.isfinite(hourly_rate) or hourly_rate <= 0:
        raise ValueError("必要检查需要通过 --hourly-rate 提供实际正数单价")
    directory = config.checks_directory()
    ensure_directory(directory)
    results: dict[str, Any] = {
        "schema_version": config.schema_version,
        "status": "running",
        "created_at": current_timestamp(),
        "environment": environment_record(device),
        "config": config.snapshot(),
        "data_context": reader.data_context,
        "hourly_rate_cny": float(hourly_rate),
        "overhead_hours": DEFAULT_OVERHEAD_HOURS,
        "models": {},
    }
    write_json(directory / "check.json", results)
    reader.save_target_indices(directory / TARGET_INDICES_FILE)
    with LogWriter(directory / "check.log") as log:
        log.write("集中必要检查开始")
        results["data"] = check_data(config, reader, log)
        for model_name in config.model_names:
            entry: dict[str, Any] = {
                "memory_estimate": estimate_memory(config, reader, model_name),
                "training_step": check_training_step(
                    config, reader, model_name, device, log
                ),
            }
            entry["compute"] = entry["training_step"].pop("compute")
            if model_name == "whale" or (config.schema_version == 3 and model_name == "hstu"):
                entry["weight_roundtrip"] = check_weight_roundtrip(
                    config, reader, device, directory, log, model_name
                )
            entry["training_speed"] = measure_training_speed(
                config, reader, model_name, device, log
            )
            entry["evaluation_speed"] = measure_evaluation_speed(
                config, reader, model_name, device, log
            )
            results["models"][model_name] = entry
        results["estimate"] = estimate_cost(config, results)
        results["status"] = "passed"
        write_json(directory / "check.json", results)
        log.write("集中必要检查结束")
    return results


def verify_checks(config: Config, reader: BatchReader, device: torch.device) -> dict[str, Any]:
    directory = config.checks_directory()
    results = read_json(directory / "check.json")
    if results.get("schema_version") != config.schema_version or results.get("status") != "passed":
        raise ValueError("当前数据与配置尚未完成必要运行检查")
    if results["config"] != config.snapshot() or results["data_context"] != reader.data_context:
        raise ValueError("必要检查的数据或配置与当前正式运行不一致")
    current_environment = environment_record(device)
    for field in ("device", "device_memory_bytes", "torch", "torch_cuda", "cpu_count"):
        if results["environment"][field] != current_environment[field]:
            raise ValueError(f"当前环境的 {field} 与必要检查记录不一致")
    reader.verify_target_indices(directory / TARGET_INDICES_FILE)
    if tuple(results["models"]) != config.model_names:
        raise ValueError("必要检查没有覆盖全部五种模型")
    cost = float(results["estimate"]["estimated_cost_cny"])
    if not np.isfinite(cost) or cost <= 0.0 or cost > MAX_TOTAL_COST_CNY:
        raise ValueError("预计费用不在已授权预算内，需要用户决定后续运行")
    return results
