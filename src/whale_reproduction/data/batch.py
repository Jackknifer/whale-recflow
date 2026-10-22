from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator

import numpy as np

from ..config import Config, parse_time_ms
from ..models.embedding import VocabularySizes
from .preprocessing import (
    apply_numerical_parameters,
    history_age_seconds,
    numerical_group_slices,
    raw_numerical_values,
)
from .store import ProcessedData

MAX_TIME_BUCKET = 128
TIME_LOG_DIVISOR = 0.301
PADDING_INDEX = 0


@dataclass(frozen=True)
class Batch:
    categorical: dict[str, np.ndarray]
    duration: np.ndarray
    request_time: np.ndarray
    video_age: np.ndarray
    history_video: dict[str, np.ndarray]
    history_feedback: dict[str, np.ndarray]
    history_age: np.ndarray
    history_mask: np.ndarray
    time_bucket: np.ndarray
    causal_mask: np.ndarray
    labels: np.ndarray
    event_id: np.ndarray
    request_id: np.ndarray
    duration_ms: np.ndarray
    history_time_ms: np.ndarray
    target_time_ms: np.ndarray
    video_is_unknown: np.ndarray
    author_is_unknown: np.ndarray

    @property
    def size(self) -> int:
        return int(self.labels.size)

    def as_tensors(self, device) -> dict[str, object]:
        import torch

        tensors: dict[str, object] = {}
        for name, values in self.categorical.items():
            tensors[name] = torch.from_numpy(np.ascontiguousarray(values)).to(device)
        for name, values in self.history_video.items():
            tensors[f"history_{name}"] = torch.from_numpy(
                np.ascontiguousarray(values)
            ).to(device)
        for name, values in self.history_feedback.items():
            tensors[f"history_{name}"] = torch.from_numpy(
                np.ascontiguousarray(values)
            ).to(device)
        for name, values in (
            ("duration", self.duration),
            ("request_time", self.request_time),
            ("video_age", self.video_age),
            ("history_age", self.history_age),
            ("history_mask", self.history_mask),
            ("time_bucket", self.time_bucket),
            ("causal_mask", self.causal_mask),
            ("history_time_ms", self.history_time_ms),
            ("target_time_ms", self.target_time_ms),
        ):
            tensors[name] = torch.from_numpy(np.ascontiguousarray(values)).to(device)
        return tensors


class BatchReader:
    def __init__(self, processed: ProcessedData) -> None:
        self._data = processed
        self._config: Config = processed.config
        self._exposure = processed.exposure
        self._vocabulary = processed.vocabulary
        self._numerical = processed.numerical
        self._max_length = processed.max_history_length
        self._categorical_fields = tuple(processed.config.data.categorical_fields)
        self._history_video_fields = tuple(processed.config.data.history_video_fields)
        self._history_feedback_fields = tuple(
            processed.config.data.history_feedback_fields
        )

    @property
    def max_history_length(self) -> int:
        return self._max_length

    @property
    def exposure(self):
        return self._exposure

    @property
    def data_context(self) -> dict[str, object]:
        result = {
            "preprocessing_config": self._data.metadata["preprocessing_config"],
            "source_files": self._data.metadata["source_files"],
            "source_file_rows": self._data.metadata["source_file_rows"],
            "event_count": int(self._exposure.size),
            "request_count": int(self._data.request_index.size),
            "vocabulary": self._vocabulary.rules(),
            "numerical": self._numerical.to_json(),
        }
        if self._config.schema_version == 3:
            result["split_statistics"] = self._data.metadata["splits"]
        return result

    @property
    def event_available_ms(self) -> np.ndarray:
        return self._data.event_available_ms

    @property
    def categorical_fields(self) -> tuple[str, ...]:
        return self._categorical_fields

    @property
    def history_video_fields(self) -> tuple[str, ...]:
        return self._history_video_fields

    @property
    def history_feedback_fields(self) -> tuple[str, ...]:
        return self._history_feedback_fields

    @property
    def vocabulary_sizes(self) -> VocabularySizes:
        sizes = self._vocabulary.sizes
        return VocabularySizes(
            categorical={field: int(sizes[field]) for field in self._categorical_fields},
            feedback={
                field: int(self._vocabulary.feedback_sizes[field])
                for field in self._history_feedback_fields
            },
        )

    def build_model(self, model_name: str, config) -> "object":
        from ..models.factory import build_model as _build_model

        return _build_model(
            model_name,
            config.model,
            self.vocabulary_sizes,
            self._max_length,
            self._categorical_fields,
            self._history_video_fields,
            self._history_feedback_fields,
        )

    def target_rows(self, split: str) -> np.ndarray:
        if split not in self._data.target_rows:
            raise KeyError(f"未知数据集合 {split}")
        return self._data.target_rows[split]

    def target_unknown(self, split: str, field: str) -> np.ndarray:
        rows = self.target_rows(split)
        return self._vocabulary.encode(field, self._exposure.categorical(field)[rows]) == self._vocabulary.unknown_index

    def save_target_indices(self, path: Path) -> None:
        values = {}
        for split in ("train", "validation", "test"):
            rows = self.target_rows(split)
            values[f"{split}_rows"] = rows
            values[f"{split}_event_ids"] = self._exposure.event_id[rows]
        temporary = path.with_name(f".{path.name}.partial")
        with temporary.open("wb") as handle:
            np.savez(handle, **values)
        temporary.replace(path)

    def verify_target_indices(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as recorded:
            for split in ("train", "validation", "test"):
                rows = self.target_rows(split)
                if not np.array_equal(recorded[f"{split}_rows"], rows):
                    raise ValueError(f"{split} 完整目标索引与保存的记录不一致")
                if not np.array_equal(recorded[f"{split}_event_ids"], self._exposure.event_id[rows]):
                    raise ValueError(f"{split} 完整事件编号与保存的记录不一致")

    def history_rows(self, split: str, positions: np.ndarray) -> np.ndarray:
        if split not in self._data.history_rows:
            raise KeyError(f"未知数据集合 {split}")
        requests = self._data.history_target_positions[split][positions]
        return self._data.history_rows[split][requests]

    def history_counts(self, split: str, positions: np.ndarray) -> np.ndarray:
        if split not in self._data.history_counts:
            raise KeyError(f"未知数据集合 {split}")
        requests = self._data.history_target_positions[split][positions]
        return self._data.history_counts[split][requests]

    def num_batches(self, split: str, batch_size: int) -> int:
        total = int(self.target_rows(split).size)
        return (total + batch_size - 1) // batch_size

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        shuffle: bool,
        generator: np.random.Generator | None,
        training: bool = False,
        epoch: int = 0,
    ) -> Iterator[Batch]:
        rows = self.target_rows(split)
        total = int(rows.size)
        if total == 0:
            raise ValueError(f"数据集合 {split} 没有目标")
        if training and split != "train":
            raise ValueError("训练遮盖只能用于训练集合")
        if shuffle and self._config.training.sample_order == "daily_shuffle":
            if split != "train":
                raise ValueError("逐日训练顺序只用于训练集合")
            order = self.daily_order(epoch)
        elif shuffle:
            if generator is None:
                raise ValueError("打乱批次顺序时需要随机数生成器")
            order = generator.permutation(total).astype(np.int64)
        else:
            order = np.arange(total, dtype=np.int64)
        for batch_index, start in enumerate(range(0, total, batch_size)):
            positions = order[start : start + batch_size]
            batch = self.build_batch(rows[positions], self.history_rows(split, positions))
            yield self.mask_training_batch(batch, epoch, batch_index) if training else batch

    def daily_order(self, epoch: int) -> np.ndarray:
        if epoch != 0 and self._config.schema_version == 3:
            raise ValueError("版本 3 只允许单轮训练")
        rows = self.target_rows("train")
        start_ms = parse_time_ms(self._config.data.train_start, self._config.data.timezone)
        day = (self._exposure.request_timestamp[rows] - start_ms) // 86400000
        order = np.argsort(day, kind="stable")
        ordered_days = day[order]
        bounds = np.concatenate(([0], np.flatnonzero(np.diff(ordered_days)) + 1, [rows.size]))
        for begin, end in zip(bounds[:-1], bounds[1:]):
            day_index = int(ordered_days[begin])
            generator = np.random.Generator(np.random.PCG64([self._config.seed, day_index]))
            order[begin:end] = generator.permutation(order[begin:end])
        return order.astype(np.int64)

    def mask_training_batch(self, batch: Batch, epoch: int, batch_index: int) -> Batch:
        training = self._config.training
        if training.id_mask_probability == 0.0:
            return batch
        generator = np.random.Generator(np.random.PCG64([
            self._config.seed, epoch, batch_index, training.id_mask_seed_namespace,
        ]))
        categorical = dict(batch.categorical)
        history_video = dict(batch.history_video)
        for field in training.id_mask_fields:
            values = categorical[field]
            chosen = generator.random(values.shape, dtype=np.float32) < training.id_mask_probability
            categorical[field] = np.where(chosen & (values > 1), self._vocabulary.unknown_index, values)
        for field in ("video_id", "author_id"):
            values = history_video[field]
            chosen = generator.random(values.shape, dtype=np.float32) < training.id_mask_probability
            history_video[field] = np.where(
                chosen & batch.history_mask & (values > 1), self._vocabulary.unknown_index, values
            )
        return replace(batch, categorical=categorical, history_video=history_video)

    def build_batch(self, rows: np.ndarray, history_slot: np.ndarray) -> Batch:
        exposure = self._exposure
        rows = np.asarray(rows, dtype=np.int64)
        history_slot = np.asarray(history_slot, dtype=np.int64)
        if history_slot.ndim != 2 or history_slot.shape[1] != self._max_length:
            raise ValueError(
                f"历史行索引形状为 {history_slot.shape}，"
                f"需要 ({rows.size}, {self._max_length})"
            )
        if history_slot.shape[0] != rows.size:
            raise ValueError(
                f"历史行索引行数 {history_slot.shape[0]} 与目标数量 {rows.size} 不一致"
            )
        categorical = {
            field: self._vocabulary.encode(
                field, exposure.categorical(field)[rows]
            )
            for field in self._categorical_fields
        }
        numerical = raw_numerical_values(
            exposure.duration[rows],
            exposure.request_timestamp[rows],
            exposure.upload_timestamp[rows],
        )
        applied = apply_numerical_parameters(numerical, self._numerical)
        valid = history_slot >= 0
        safe = np.maximum(history_slot, 0)
        history_video = {}
        for field in self._history_video_fields:
            encoded = self._vocabulary.encode(
                field, exposure.video_attribute(field)[safe]
            )
            history_video[field] = np.where(valid, encoded, PADDING_INDEX).astype(
                np.int64
            )
        history_feedback = {}
        for field in self._history_feedback_fields:
            encoded = self._vocabulary.encode_feedback(field, exposure.feedback(field)[safe])
            history_feedback[field] = np.where(valid, encoded, PADDING_INDEX).astype(
                np.int64
            )
        history_time = np.where(valid, exposure.request_timestamp[safe], 0)
        target_time = exposure.request_timestamp[rows]
        age = history_age_seconds(target_time, history_time)
        age = np.where(valid, age, 0.0).astype(np.float32)
        time_bucket, causal_mask = self._temporal_structures(valid, history_time)
        slices = numerical_group_slices(self._config)
        duration = applied[:, slices[0]]
        request_time = applied[:, slices[1]]
        video_age = applied[:, slices[2]]
        return Batch(
            categorical=categorical,
            duration=np.ascontiguousarray(duration),
            request_time=np.ascontiguousarray(request_time),
            video_age=np.ascontiguousarray(video_age),
            history_video=history_video,
            history_feedback=history_feedback,
            history_age=age,
            history_mask=valid,
            time_bucket=time_bucket,
            causal_mask=causal_mask,
            labels=exposure.long_view[rows].astype(np.float32),
            event_id=exposure.event_id[rows].astype(np.int64),
            request_id=exposure.request_id[rows].astype(np.int64),
            duration_ms=exposure.duration[rows].astype(np.int64),
            history_time_ms=history_time.astype(np.int64),
            target_time_ms=target_time.astype(np.int64),
            video_is_unknown=categorical["video_id"] == self._vocabulary.unknown_index,
            author_is_unknown=categorical["author_id"] == self._vocabulary.unknown_index,
        )

    def _temporal_structures(
        self, history_mask: np.ndarray, history_time: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        row_time = history_time[:, :, None]
        column_time = history_time[:, None, :]
        difference = np.abs(row_time - column_time).astype(np.float64)
        magnitude = np.log(np.maximum(difference, 1.0)) / TIME_LOG_DIVISOR
        time_bucket = np.clip(np.floor(magnitude), 0, MAX_TIME_BUCKET).astype(np.int64)
        row_valid = history_mask[:, :, None]
        column_valid = history_mask[:, None, :]
        diagonal = np.arange(self._max_length)[None, :, None] == np.arange(
            self._max_length
        )[None, None, :]
        causal_mask = row_valid & column_valid & ((column_time < row_time) | diagonal)
        return time_bucket, causal_mask
