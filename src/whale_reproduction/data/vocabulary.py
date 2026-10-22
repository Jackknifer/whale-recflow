from dataclasses import dataclass

import numpy as np

from ..config import Config
from .raw import (
    CATEGORICAL_FIELDS,
    HISTORY_FEEDBACK_FIELDS,
    HISTORY_VIDEO_FIELDS,
    ExposureTable,
)

FEEDBACK_TABLE_SIZE = 4
HISTORY_FIT_CHUNK = 8192


@dataclass(frozen=True)
class Vocabulary:
    sizes: dict[str, int]
    raw_values: dict[str, np.ndarray]
    feedback_sizes: dict[str, int]
    padding_index: int
    unknown_index: int
    code_offset: int
    feedback_offset: int

    def rules(self) -> dict[str, object]:
        return {
            "sizes": dict(self.sizes),
            "feedback_sizes": dict(self.feedback_sizes),
            "padding_index": self.padding_index,
            "unknown_index": self.unknown_index,
            "code_offset": self.code_offset,
            "feedback_offset": self.feedback_offset,
            "categorical_fields": list(CATEGORICAL_FIELDS),
            "history_video_fields": list(HISTORY_VIDEO_FIELDS),
            "history_feedback_fields": list(HISTORY_FEEDBACK_FIELDS),
            "fit_scope": "training_inputs_and_their_histories",
        }

    def encode(self, field: str, raw: np.ndarray) -> np.ndarray:
        table = self.raw_values[field]
        position = np.searchsorted(table, raw)
        clipped = np.clip(position, 0, table.size - 1)
        found = table[clipped] == raw
        return np.where(found, clipped + self.code_offset, self.unknown_index).astype(
            np.int64
        )

    def encode_feedback(self, field: str, raw: np.ndarray) -> np.ndarray:
        if field not in self.feedback_sizes:
            raise KeyError(f"未知历史反馈字段 {field}")
        valid = (raw == 0) | (raw == 1)
        return np.where(valid, raw + self.feedback_offset, self.unknown_index).astype(
            np.int64
        )


def _unique_join(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(chunks))


def fit_vocabulary(
    config: Config,
    exposure: ExposureTable,
    train_rows: np.ndarray,
    train_history_rows: np.ndarray,
) -> Vocabulary:
    padding_index = int(config.data.padding_index)
    unknown_index = int(config.data.unknown_index)
    if padding_index != 0 or unknown_index != 1:
        raise ValueError(
            f"当前实现要求 PAD 为 0、UNK 为 1，配置为 {padding_index} 与 {unknown_index}"
        )
    feedback_offset = int(config.data.feedback_index_offset)
    if feedback_offset != 2:
        raise ValueError(f"历史反馈编码的增加值固定为 2，配置为 {feedback_offset}")
    offset = unknown_index + 1

    used_history = np.zeros(exposure.size, dtype=bool)
    for start in range(0, train_history_rows.shape[0], HISTORY_FIT_CHUNK):
        chunk = train_history_rows[start : start + HISTORY_FIT_CHUNK]
        used_history[chunk[chunk >= 0]] = True
    used_history_rows = np.flatnonzero(used_history)
    del used_history
    raw_values: dict[str, np.ndarray] = {}
    for field in config.data.categorical_fields:
        target_values = exposure.categorical(field)[train_rows].astype(np.int64)
        if field in config.data.history_video_fields:
            history_values = exposure.categorical(field)[used_history_rows].astype(np.int64)
            raw_values[field] = _unique_join([target_values, history_values])
        else:
            raw_values[field] = np.unique(target_values)
    missing = [field for field in CATEGORICAL_FIELDS if field not in raw_values]
    if missing:
        raise KeyError(f"类别字典缺少字段 {missing}")

    sizes = {field: int(values.size) + offset for field, values in raw_values.items()}
    feedback_sizes = {
        field: FEEDBACK_TABLE_SIZE for field in config.data.history_feedback_fields
    }
    return Vocabulary(
        sizes=sizes,
        raw_values=raw_values,
        feedback_sizes=feedback_sizes,
        padding_index=padding_index,
        unknown_index=unknown_index,
        code_offset=offset,
        feedback_offset=feedback_offset,
    )
