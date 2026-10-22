from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import read_json, write_json
from .preprocessing import NumericalParameters
from .raw import (
    CATEGORICAL_FIELDS,
    EXPOSURE_INT8,
    EXPOSURE_INT32,
    EXPOSURE_INT64,
    ExposureTable,
    RequestIndex,
)
from .vocabulary import Vocabulary

EXPOSURE_FILE = "exposure.parquet"
REQUEST_FILE = "requests.parquet"
VOCABULARY_FILE = "vocabulary.npz"
PREPROCESSING_FILE = "preprocessing.json"
METADATA_FILE = "metadata.json"
TARGET_DIRECTORY = "targets"
HISTORY_DIRECTORY = "history"
SPLIT_NAMES = ("train", "validation", "test")

EXPOSURE_COLUMNS = (
    "event_id",
    "request_id",
    "user_id",
    "request_timestamp",
    "device_id",
    "age",
    "gender",
    "province",
    "video_id",
    "author_id",
    "duration",
    "category_level_one",
    "category_level_two",
    "upload_type",
    "upload_timestamp",
    "playing_time",
    "effective_view",
    "long_view",
    "file_index",
    "record_index",
)
REQUEST_COLUMNS = (
    "request_id",
    "user_id",
    "request_timestamp",
    "available_ms",
    "event_counts",
)


@dataclass(frozen=True)
class ProcessedData:
    exposure: ExposureTable
    request_index: RequestIndex
    vocabulary: Vocabulary
    numerical: NumericalParameters
    metadata: dict[str, Any]
    target_rows: dict[str, np.ndarray]
    history_rows: dict[str, np.ndarray]
    history_counts: dict[str, np.ndarray]
    history_requests: dict[str, np.ndarray]
    history_target_positions: dict[str, np.ndarray]
    event_available_ms: np.ndarray
    config: Config

    @property
    def max_history_length(self) -> int:
        return int(self.config.data.max_history_length)


def _exposure_from_frame(frame: pd.DataFrame) -> ExposureTable:
    missing = [name for name in EXPOSURE_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"曝光文件缺少字段 {missing}")
    values: dict[str, np.ndarray] = {}
    for name in EXPOSURE_INT32:
        values[name] = frame[name].to_numpy(dtype=np.int32)
    for name in EXPOSURE_INT64:
        values[name] = frame[name].to_numpy(dtype=np.int64)
    for name in EXPOSURE_INT8:
        values[name] = frame[name].to_numpy(dtype=np.int8)
    return ExposureTable(
        request_id=values["request_id"],
        user_id=values["user_id"],
        request_timestamp=values["request_timestamp"],
        device_id=values["device_id"],
        age=values["age"],
        gender=values["gender"],
        province=values["province"],
        video_id=values["video_id"],
        author_id=values["author_id"],
        duration=values["duration"],
        category_level_one=values["category_level_one"],
        category_level_two=values["category_level_two"],
        upload_type=values["upload_type"],
        upload_timestamp=values["upload_timestamp"],
        playing_time=values["playing_time"],
        effective_view=values["effective_view"],
        long_view=values["long_view"],
        event_id=frame["event_id"].to_numpy(dtype=np.int64),
        file_index=frame["file_index"].to_numpy(dtype=np.int16),
        record_index=frame["record_index"].to_numpy(dtype=np.int32),
    )


def _request_index_from_frame(frame: pd.DataFrame) -> RequestIndex:
    missing = [name for name in REQUEST_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"请求文件缺少字段 {missing}")
    return RequestIndex(
        request_ids=frame["request_id"].to_numpy(dtype=np.int64),
        user_id=frame["user_id"].to_numpy(dtype=np.int32),
        request_timestamp=frame["request_timestamp"].to_numpy(dtype=np.int64),
        available_ms=frame["available_ms"].to_numpy(dtype=np.int64),
        event_counts=frame["event_counts"].to_numpy(dtype=np.int32),
    )


def _exposure_frame(exposure: ExposureTable) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": exposure.event_id,
            "request_id": exposure.request_id,
            "user_id": exposure.user_id,
            "request_timestamp": exposure.request_timestamp,
            "device_id": exposure.device_id,
            "age": exposure.age,
            "gender": exposure.gender,
            "province": exposure.province,
            "video_id": exposure.video_id,
            "author_id": exposure.author_id,
            "duration": exposure.duration,
            "category_level_one": exposure.category_level_one,
            "category_level_two": exposure.category_level_two,
            "upload_type": exposure.upload_type,
            "upload_timestamp": exposure.upload_timestamp,
            "playing_time": exposure.playing_time,
            "effective_view": exposure.effective_view,
            "long_view": exposure.long_view,
            "file_index": exposure.file_index,
            "record_index": exposure.record_index,
        }
    )


def _request_frame(index: RequestIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "request_id": index.request_ids,
            "user_id": index.user_id,
            "request_timestamp": index.request_timestamp,
            "available_ms": index.available_ms,
            "event_counts": index.event_counts,
        }
    )


def save_processed(
    config: Config,
    exposure: ExposureTable,
    request_index: RequestIndex,
    vocabulary: Vocabulary,
    numerical: NumericalParameters,
    metadata: dict[str, Any],
    target_rows: dict[str, np.ndarray],
    history_rows: dict[str, np.ndarray],
    history_counts: dict[str, np.ndarray],
    history_requests: dict[str, np.ndarray],
    history_target_positions: dict[str, np.ndarray],
) -> Path:
    root = config.processed_directory()
    root.mkdir(parents=True, exist_ok=True)
    _exposure_frame(exposure).to_parquet(root / EXPOSURE_FILE, index=False)
    _request_frame(request_index).to_parquet(root / REQUEST_FILE, index=False)
    np.savez(
        root / VOCABULARY_FILE,
        **{
            field: vocabulary.raw_values[field].astype(np.int64)
            for field in CATEGORICAL_FIELDS
        },
    )
    target_directory = root / TARGET_DIRECTORY
    target_directory.mkdir(parents=True, exist_ok=True)
    history_directory = root / HISTORY_DIRECTORY
    history_directory.mkdir(parents=True, exist_ok=True)
    for split, rows in target_rows.items():
        np.save(target_directory / f"{split}.npy", rows.astype(np.int64))
        np.save(
            history_directory / f"{split}_rows.npy", history_rows[split].astype(np.int32)
        )
        np.save(
            history_directory / f"{split}_counts.npy",
            history_counts[split].astype(np.int32),
        )
        np.save(history_directory / f"{split}_requests.npy", history_requests[split])
        np.save(
            history_directory / f"{split}_target_positions.npy",
            history_target_positions[split],
        )
    write_json(
        root / PREPROCESSING_FILE,
        {
            "vocabulary": vocabulary.rules(),
            "numerical": numerical.to_json(),
        },
    )
    write_json(root / METADATA_FILE, metadata)
    return root


def load_processed(config: Config) -> ProcessedData:
    root = config.processed_directory()
    required = (
        EXPOSURE_FILE,
        REQUEST_FILE,
        VOCABULARY_FILE,
        PREPROCESSING_FILE,
        METADATA_FILE,
    )
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(
                f"加工数据缺少文件 {root / name}，请先执行 prepare_data"
            )
    metadata = read_json(root / METADATA_FILE)
    if config.schema_version == 3 and metadata.get("processed_format_version") != 3:
        raise ValueError("版本 3 需要加工格式 3")
    if metadata.get("preprocessing_config") != config.preprocessing_snapshot():
        raise ValueError("加工数据的日期、抽样或预处理规则与当前配置不一致")
    if metadata.get("history_storage") != "per_request_v1":
        raise ValueError("加工数据的历史索引保存方式与当前实现不一致")
    if metadata.get("data_name") != config.data.name:
        raise ValueError(
            f"加工数据的数据集为 {metadata.get('data_name')}，配置为 {config.data.name}"
        )
    if int(metadata.get("schema_version", -1)) != config.schema_version:
        raise ValueError(
            f"加工数据的配置版本为 {metadata.get('schema_version')}，"
            f"当前配置为 {config.schema_version}"
        )
    if metadata.get("label") != config.data.label:
        raise ValueError(
            f"加工数据的标签为 {metadata.get('label')}，配置为 {config.data.label}"
        )
    if int(metadata.get("max_history_length", -1)) != config.data.max_history_length:
        raise ValueError("加工数据的历史长度与当前配置不一致")
    recorded_files = metadata.get("source_files")
    if recorded_files != list(config.data.file_names()):
        raise ValueError("加工数据的来源文件清单与当前配置不一致")

    exposure = _exposure_from_frame(pd.read_parquet(root / EXPOSURE_FILE))
    request_index = _request_index_from_frame(pd.read_parquet(root / REQUEST_FILE))
    payload = read_json(root / PREPROCESSING_FILE)
    with np.load(root / VOCABULARY_FILE) as archive:
        raw_values = {field: archive[field].copy() for field in CATEGORICAL_FIELDS}
    rules = payload["vocabulary"]
    vocabulary = Vocabulary(
        sizes={field: int(value) for field, value in rules["sizes"].items()},
        raw_values=raw_values,
        feedback_sizes={
            field: int(value) for field, value in rules["feedback_sizes"].items()
        },
        padding_index=int(rules["padding_index"]),
        unknown_index=int(rules["unknown_index"]),
        code_offset=int(rules["code_offset"]),
        feedback_offset=int(rules["feedback_offset"]),
    )
    numerical = NumericalParameters.from_json(payload["numerical"])
    target_rows = {
        split: np.load(root / TARGET_DIRECTORY / f"{split}.npy", mmap_mode="r")
        for split in SPLIT_NAMES
    }
    history_rows = {
        split: np.load(root / HISTORY_DIRECTORY / f"{split}_rows.npy", mmap_mode="r")
        for split in SPLIT_NAMES
    }
    history_counts = {
        split: np.load(root / HISTORY_DIRECTORY / f"{split}_counts.npy", mmap_mode="r")
        for split in SPLIT_NAMES
    }
    history_requests = {
        split: np.load(root / HISTORY_DIRECTORY / f"{split}_requests.npy", mmap_mode="r")
        for split in SPLIT_NAMES
    }
    history_target_positions = {
        split: np.load(root / HISTORY_DIRECTORY / f"{split}_target_positions.npy", mmap_mode="r")
        for split in SPLIT_NAMES
    }
    if exposure.size != metadata["event_count"] or request_index.size != metadata["request_count"]:
        raise ValueError("加工数据的曝光或请求数量与元数据不一致")
    for split in SPLIT_NAMES:
        rows = target_rows[split]
        requests = history_requests[split]
        mapping = history_target_positions[split]
        if rows.dtype != np.int64 or rows.shape != (config.data.target_samples[split],):
            raise ValueError(f"数据集合 {split} 的目标索引类型或数量不符合配置")
        if np.any(rows < 0) or np.any(rows >= exposure.size):
            raise ValueError(f"数据集合 {split} 的目标索引超出曝光范围")
        if history_rows[split].dtype != np.int32 or history_rows[split].shape != (requests.size, config.data.max_history_length):
            raise ValueError(
                f"数据集合 {split} 的历史行索引形状为 {history_rows[split].shape}，"
                "与目标数量或历史长度不一致"
            )
        if history_counts[split].shape != requests.shape:
            raise ValueError(f"数据集合 {split} 的历史长度数量与请求数量不一致")
        if mapping.shape != rows.shape or np.any(mapping < 0) or np.any(mapping >= requests.size):
            raise ValueError(f"数据集合 {split} 的目标与历史请求映射无效")
        if not np.array_equal(requests[mapping], exposure.request_id[rows]):
            raise ValueError(f"数据集合 {split} 的历史请求与目标请求不一致")
    event_slot = request_index.positions(exposure.request_id)
    if np.any(event_slot < 0):
        raise ValueError("曝光记录的 request_id 未出现在请求索引中")
    return ProcessedData(
        exposure=exposure,
        request_index=request_index,
        vocabulary=vocabulary,
        numerical=numerical,
        metadata=metadata,
        target_rows=target_rows,
        history_rows=history_rows,
        history_counts=history_counts,
        history_requests=history_requests,
        history_target_positions=history_target_positions,
        event_available_ms=request_index.available_ms[event_slot],
        config=config,
    )
