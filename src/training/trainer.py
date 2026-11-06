import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..config import Config
from ..data.batch import BatchReader
from ..utils import (
    current_timestamp,
    ensure_directory,
    format_seconds,
    restore_rng_states,
    rng_states,
    save_checkpoint,
    seed_everything,
)
from .evaluate import collect_logits
from .metrics import Metrics, compute_metrics

BEST_MODEL_NAME = "best_model.pt"
RESUME_NAME = "resume.pt"
LOG_NAME = "train.log"
STATE_SCHEMA_VERSION = 2


def best_epoch_path(directory: Path, epoch: int) -> Path:
    return directory / f"best_epoch_{epoch}.pt"


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_loss: float
    train_steps: int
    train_seconds: float
    validation: Metrics
    validation_seconds: float
    selected: bool
    stale_epochs: int

    def to_json(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_steps": self.train_steps,
            "train_seconds": self.train_seconds,
            "validation": self.validation.to_json(),
            "validation_seconds": self.validation_seconds,
            "selected": self.selected,
            "stale_epochs": self.stale_epochs,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "EpochResult":
        return cls(
            epoch=int(payload["epoch"]),
            train_loss=float(payload["train_loss"]),
            train_steps=int(payload["train_steps"]),
            train_seconds=float(payload["train_seconds"]),
            validation=metrics_from_json(payload["validation"]),
            validation_seconds=float(payload["validation_seconds"]),
            selected=bool(payload["selected"]),
            stale_epochs=int(payload["stale_epochs"]),
        )


@dataclass(frozen=True)
class TrainingOutcome:
    model_name: str
    best_epoch: int
    best_validation: Metrics
    history: list[EpochResult]
    stopped_early: bool
    train_seconds: float
    validation_seconds: float
    resumed_from_epoch: int
    peak_memory_bytes: int


class LogWriter:
    def __init__(self, path: Path, echo: bool = False) -> None:
        ensure_directory(path.parent)
        self._handle = path.open("a", encoding="utf-8")
        self._echo = echo

    def write(self, message: str) -> None:
        line = f"[{current_timestamp()}] {message}"
        self._handle.write(line + "\n")
        self._handle.flush()
        if self._echo:
            print(line, flush=True)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "LogWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def metrics_from_json(payload: dict[str, Any]) -> Metrics:
    return Metrics(
        auc=float(payload["auc"]),
        logloss=float(payload["logloss"]),
        ne=float(payload["ne"]),
        count=int(payload["count"]),
        positive_count=int(payload["positive_count"]),
        positive_rate=float(payload["positive_rate"]),
    )


def parameter_report(model: nn.Module) -> dict[str, int]:
    embedding = sum(
        module.weight.numel()
        for module in model.modules()
        if isinstance(module, nn.Embedding)
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": int(total),
        "embedding_parameters": int(embedding),
        "trainable_parameters": int(trainable),
    }


def build_optimizer(config: Config, model: nn.Module) -> torch.optim.Optimizer:
    if config.training.optimizer != "adamw":
        raise ValueError(f"不支持的优化器 {config.training.optimizer}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )


def build_loss(config: Config) -> nn.Module:
    if config.training.loss != "bce_with_logits":
        raise ValueError(f"不支持的损失函数 {config.training.loss}")
    return nn.BCEWithLogitsLoss()


def _train_epoch(
    config: Config,
    model: nn.Module,
    reader: BatchReader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    generator: np.random.Generator,
    log: LogWriter,
    epoch: int,
) -> tuple[float, int, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    steps = 0
    started = time.perf_counter()
    batch_size = config.training.batch_size
    for batch in reader.iter_batches("train", batch_size, shuffle=True, generator=generator, training=True, epoch=epoch):
        tensors = batch.as_tensors(device)
        labels = torch.from_numpy(batch.labels).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(tensors)
        if logits.shape != (batch.size,):
            raise ValueError(f"模型输出形状为 {tuple(logits.shape)}，期望 ({batch.size},)")
        loss = loss_function(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError(f"Epoch {epoch} 第 {steps + 1} 步出现非有限损失")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.training.gradient_clip_norm, error_if_nonfinite=True
        )
        optimizer.step()
        total_loss += float(loss.detach()) * batch.size
        total_samples += batch.size
        steps += 1
        if steps % config.training.log_every_steps == 0:
            log.write(f"Epoch {epoch} 步 {steps} 平均损失 {total_loss / total_samples:.6f}")
    if total_samples == 0:
        raise ValueError("训练集合没有可用样本")
    return total_loss / total_samples, steps, time.perf_counter() - started


def run_training(
    config: Config,
    model_name: str,
    reader: BatchReader,
    device: torch.device,
    run_directory: Path,
    resume: bool,
) -> TrainingOutcome:
    ensure_directory(run_directory)
    seed_everything(config.seed)
    model = reader.build_model(model_name, config).to(device)
    optimizer = build_optimizer(config, model)
    loss_function = build_loss(config)

    resume_path = run_directory / RESUME_NAME
    start_epoch = 0
    best_logloss = float("inf")
    best_epoch = -1
    best_validation: Metrics | None = None
    stale_epochs = 0
    history: list[EpochResult] = []
    train_seconds_total = 0.0
    validation_seconds_total = 0.0
    previous_peak_memory_bytes = 0

    if not config.training.checkpoint_on_epoch_end:
        raise ValueError("正式实验要求在 Epoch 结束时保存检查点")

    if resume:
        if not resume_path.is_file():
            raise FileNotFoundError(f"恢复文件不存在 {resume_path}")
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        verify_resume_state(config, state, reader, model_name)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng_states(state["rng"])
        start_epoch = int(state["next_epoch"])
        best_logloss = float(state["best_logloss"])
        best_epoch = int(state["best_epoch"])
        best_validation = (
            metrics_from_json(state["best_validation"])
            if state["best_validation"] is not None
            else None
        )
        stale_epochs = int(state["stale_epochs"])
        history = [EpochResult.from_json(item) for item in state["history"]]
        train_seconds_total = float(state["train_seconds_total"])
        validation_seconds_total = float(state["validation_seconds_total"])
        previous_peak_memory_bytes = int(state["peak_memory_bytes"])
        del state
    elif resume_path.is_file():
        raise RuntimeError(
            f"运行目录已存在恢复文件 {resume_path}，恢复训练需要显式指定 --resume"
        )

    batch_size = config.training.batch_size
    stopped_early = stale_epochs >= config.training.early_stopping_patience
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with LogWriter(run_directory / LOG_NAME, echo=True) as log:
        log.write(f"模型 {model_name} 开始，起始 Epoch {start_epoch}")
        epochs = () if stopped_early else range(start_epoch, config.training.max_epochs)
        for epoch in epochs:
            generator = np.random.Generator(np.random.PCG64([config.seed, epoch]))
            train_loss, steps, train_seconds = _train_epoch(
                config, model, reader, optimizer, loss_function, device, generator, log, epoch
            )
            validation_started = time.perf_counter()
            validation_logits, validation_labels, _ = collect_logits(
                model, reader, "validation", batch_size, device
            )
            validation = compute_metrics(
                validation_logits,
                validation_labels,
                config.evaluation.probability_dtype,
                config.evaluation.single_class_policy,
            )
            validation_seconds = time.perf_counter() - validation_started

            previous_best = best_logloss
            selected = validation.logloss < previous_best
            if selected:
                best_logloss = validation.logloss
                best_epoch = epoch
                best_validation = validation
                save_checkpoint(model.state_dict(), best_epoch_path(run_directory, epoch))
            improvement = previous_best - validation.logloss
            if improvement > config.training.early_stopping_min_delta:
                stale_epochs = 0
            else:
                stale_epochs += 1

            train_seconds_total += train_seconds
            validation_seconds_total += validation_seconds
            history.append(
                EpochResult(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_steps=steps,
                    train_seconds=train_seconds,
                    validation=validation,
                    validation_seconds=validation_seconds,
                    selected=selected,
                    stale_epochs=stale_epochs,
                )
            )
            log.write(
                f"Epoch {epoch} 训练损失 {train_loss:.6f} 步数 {steps} "
                f"训练用时 {format_seconds(train_seconds)} "
                f"验证 LogLoss {validation.logloss:.6f} AUC {validation.auc:.6f} "
                f"NE {validation.ne:.6f} 选择 {selected} 停止计数 {stale_epochs}"
            )
            save_resume_state(
                config,
                model_name,
                run_directory,
                model,
                optimizer,
                epoch,
                best_logloss,
                best_epoch,
                best_validation,
                stale_epochs,
                history,
                train_seconds_total,
                validation_seconds_total,
                reader,
                previous_peak_memory_bytes,
                device,
            )
            if stale_epochs >= config.training.early_stopping_patience:
                stopped_early = True
                log.write(f"Early Stopping 在 Epoch {epoch} 触发")
                break
        log.write(f"模型 {model_name} 训练结束")

    if best_validation is None or best_epoch < 0:
        raise RuntimeError("没有产生有效的验证结果，无法选择最佳模型")
    return TrainingOutcome(
        model_name=model_name,
        best_epoch=best_epoch,
        best_validation=best_validation,
        history=history,
        stopped_early=stopped_early,
        train_seconds=train_seconds_total,
        validation_seconds=validation_seconds_total,
        resumed_from_epoch=start_epoch,
        peak_memory_bytes=max(previous_peak_memory_bytes, int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0),
    )


def save_resume_state(
    config: Config,
    model_name: str,
    run_directory: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_logloss: float,
    best_epoch: int,
    best_validation: Metrics | None,
    stale_epochs: int,
    history: list[EpochResult],
    train_seconds_total: float,
    validation_seconds_total: float,
    reader: BatchReader,
    previous_peak_memory_bytes: int,
    device: torch.device,
) -> None:
    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "model_name": model_name,
        "seed": config.seed,
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": rng_states(),
        "best_logloss": best_logloss,
        "best_epoch": best_epoch,
        "best_validation": best_validation.to_json() if best_validation else None,
        "stale_epochs": stale_epochs,
        "history": [item.to_json() for item in history],
        "train_seconds_total": train_seconds_total,
        "validation_seconds_total": validation_seconds_total,
        "config": config.snapshot(),
        "data_context": reader.data_context,
        "peak_memory_bytes": max(previous_peak_memory_bytes, int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0),
        "target_counts": {
            split: int(reader.target_rows(split).size)
            for split in ("train", "validation", "test")
        },
        "preprocessing_root": str(config.processed_directory()),
        "source_files": list(config.data.file_names()),
        "event_count": int(reader.exposure.size),
        "target_rows": {
            split: np.array(reader.target_rows(split), copy=True)
            for split in ("train", "validation", "test")
        },
        "target_event_ids": {
            split: reader.exposure.event_id[reader.target_rows(split)].copy()
            for split in ("train", "validation", "test")
        },
        "saved_at": current_timestamp(),
    }
    save_checkpoint(payload, run_directory / RESUME_NAME)


def verify_resume_state(
    config: Config, state: dict[str, Any], reader: BatchReader, model_name: str
) -> None:
    if int(state.get("schema_version", -1)) != STATE_SCHEMA_VERSION:
        raise ValueError("恢复文件的版本与当前实现不一致")
    if state["config"] != config.snapshot():
        raise ValueError("恢复文件的有效配置与当前配置不一致")
    if state["data_context"] != reader.data_context:
        raise ValueError("恢复文件的数据来源或预处理参数与当前数据不一致")
    if state["model_name"] != model_name:
        raise ValueError(f"恢复文件的模型名称为 {state['model_name']}，当前为 {model_name}")
    if int(state["seed"]) != config.seed:
        raise ValueError(f"恢复文件的种子为 {state['seed']}，配置为 {config.seed}")
    counts = state["target_counts"]
    for split in ("train", "validation", "test"):
        expected = int(reader.target_rows(split).size)
        if int(counts[split]) != expected:
            raise ValueError(
                f"恢复文件 {split} 的样本数量为 {counts[split]}，"
                f"当前加工数据为 {expected}"
            )
    recorded = state["config"]["data"]["target_samples"]
    current = config.snapshot()["data"]["target_samples"]
    if recorded != current:
        raise ValueError("恢复文件的目标样本数量与当前配置不一致")
    if state["config"]["training"]["batch_size"] != config.training.batch_size:
        raise ValueError("恢复文件的 Batch Size 与当前配置不一致")
    if state["config"]["model"] != config.snapshot()["model"]:
        raise ValueError("恢复文件的模型结构与当前配置不一致")
    if state.get("source_files") != list(config.data.file_names()):
        raise ValueError("恢复文件的来源文件清单与当前配置不一致")
    if int(state.get("event_count", -1)) != int(reader.exposure.size):
        raise ValueError(
            f"恢复文件的曝光记录数为 {state.get('event_count')}，"
            f"当前加工数据为 {reader.exposure.size}"
        )
    for split in ("train", "validation", "test"):
        rows = reader.target_rows(split)
        if not np.array_equal(rows, state["target_rows"][split]):
            raise ValueError(f"恢复文件 {split} 的完整目标索引与当前数据不一致")
        if not np.array_equal(reader.exposure.event_id[rows], state["target_event_ids"][split]):
            raise ValueError(f"恢复文件 {split} 的完整事件编号与当前数据不一致")
