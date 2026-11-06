import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..utils import current_timestamp, write_json
from .official_history import read_official_histories
from .preprocessing import fit_numerical_parameters, raw_numerical_values
from .raw import (
    ExposureTable,
    build_request_index,
    read_exposure,
)
from .split import SPLIT_ORDER, candidate_rows, sample_targets, split_windows
from .store import save_processed
from .vocabulary import fit_vocabulary

HISTORY_CHUNK = 8192


def _host_memory() -> dict[str, int]:
    import resource

    # macOS 返回字节，Linux 返回 KiB，统一记录为字节。
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "peak_rss_bytes": int(usage.ru_maxrss) * (1 if sys.platform == "darwin" else 1024),
    }


def numerical_inputs(
    exposure: ExposureTable, rows: np.ndarray
) -> dict[str, np.ndarray]:
    # 只读取当前目标行，供训练数值参数拟合及后续统计使用。
    return raw_numerical_values(
        exposure.duration[rows],
        exposure.request_timestamp[rows],
        exposure.upload_timestamp[rows],
    )


def history_statistics(
    exposure: ExposureTable,
    history_rows: np.ndarray,
    history_counts: np.ndarray,
    target_positions: np.ndarray,
    max_length: int,
    unknown_masks: dict[str, np.ndarray] | None = None,
    feedback_fields: tuple[str, ...] = ("effective_view",),
) -> dict[str, Any]:
    lengths = history_counts[target_positions].astype(np.int64)
    # 同一请求可能包含多个目标，历史统计按其被目标引用的次数加权。
    weights = np.bincount(target_positions, minlength=history_counts.size)
    feedback_positive = {field: 0 for field in feedback_fields}
    total_positions = 0
    unknown_counts = {field: 0 for field in (unknown_masks or {})}
    for start in range(0, history_rows.shape[0], HISTORY_CHUNK):
        end = min(start + HISTORY_CHUNK, history_rows.shape[0])
        chunk = history_rows[start:end]
        valid = chunk >= 0
        safe = np.maximum(chunk, 0)
        multiplicity = weights[start:end]
        for field in feedback_fields:
            feedback_positive[field] += int(np.dot(
                (exposure.feedback(field)[safe] * valid).sum(axis=1), multiplicity
            ))
        total_positions += int(np.dot(valid.sum(axis=1), multiplicity))
        for field, mask in (unknown_masks or {}).items():
            unknown_counts[field] += int(np.dot((mask[safe] & valid).sum(axis=1), multiplicity))
    return {
        "history_length_min": int(lengths.min()),
        "history_length_mean": float(lengths.mean()),
        "history_length_max": int(lengths.max()),
        "empty_history_count": int((lengths == 0).sum()),
        "full_history_count": int((lengths >= max_length).sum()),
        "c32": float((lengths >= 32).mean()),
        **({"c50": float((lengths >= 50).mean())} if max_length == 50 else {}),
        **{f"history_{field}_positive": value for field, value in feedback_positive.items()},
        "history_total_positions": total_positions,
        **({"history_unknown_ratio": {
            field: count / total_positions if total_positions else None
            for field, count in unknown_counts.items()
        }} if unknown_masks is not None else {}),
    }


def unknown_ratios(
    vocabulary, exposure: ExposureTable, rows: np.ndarray
) -> dict[str, float]:
    # 统计目标中的 UNK 比例，用于描述训练词表对各集合的覆盖情况。
    ratios: dict[str, float] = {}
    for field in vocabulary.raw_values:
        encoded = vocabulary.encode(field, exposure.categorical(field)[rows])
        ratios[field] = float((encoded == vocabulary.unknown_index).mean())
    return ratios


def prepare(config: Config) -> dict[str, Any]:
    # 加工目录需为空，完成后保存三个模型共同使用的数据和来源统计。
    directory = config.processed_directory()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"加工目录已有内容，不能覆盖已有数据：{directory}")
    started = time.perf_counter()
    windows = split_windows(config)
    exposure, sources = read_exposure(config)
    request_index = build_request_index(exposure)
    candidates = candidate_rows(exposure, config)
    target_rows = sample_targets(exposure, config, candidates)
    history_requests: dict[str, np.ndarray] = {}
    history_target_positions: dict[str, np.ndarray] = {}
    # 按请求共用历史，通过 inverse 将每个目标映射回对应请求。
    for split in SPLIT_ORDER:
        rows = target_rows[split]
        requests, inverse = np.unique(exposure.request_id[rows], return_inverse=True)
        history_requests[split] = requests.astype(np.int64)
        history_target_positions[split] = inverse.astype(np.int32)
    history_rows, history_counts, official_audit = read_official_histories(
        config, exposure, request_index, history_requests
    )

    train_rows = target_rows["train"]
    train_history = history_rows["train"]
    # 词表使用训练目标及其历史，数值参数只使用训练目标。
    vocabulary = fit_vocabulary(config, exposure, train_rows, train_history)
    numerical = fit_numerical_parameters(numerical_inputs(exposure, train_rows))
    unknown_masks = {
        field: vocabulary.encode(field, exposure.video_attribute(field)) == vocabulary.unknown_index
        for field in config.data.history_video_fields
    }

    split_metadata: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        rows = target_rows[split]
        labels = exposure.long_view[rows]
        start, end = windows[split]
        split_metadata[split] = {
            "window_start_ms": int(start),
            "window_end_ms": int(end),
            "candidate_count": int(candidates[split].size),
            "target_count": int(rows.size),
            "request_count": int(np.unique(exposure.request_id[rows]).size),
            "positive_count": int(labels.sum()),
            "positive_rate": float(labels.mean()),
            "first_timestamp_ms": int(exposure.request_timestamp[rows].min()),
            "last_timestamp_ms": int(exposure.request_timestamp[rows].max()),
            "distinct_users": int(np.unique(exposure.user_id[rows]).size),
            "distinct_videos": int(np.unique(exposure.video_id[rows]).size),
            "unknown_ratio": unknown_ratios(vocabulary, exposure, rows),
            **history_statistics(
                exposure,
                history_rows[split],
                history_counts[split],
                history_target_positions[split],
                config.data.max_history_length,
                unknown_masks,
                config.data.history_feedback_fields,
            ),
        }

    metadata = {
        "schema_version": config.schema_version,
        "created_at": current_timestamp(),
        "project": config.project,
        "seed": config.seed,
        "data_name": config.data.name,
        "preprocessing_config": config.preprocessing_snapshot(),
        "history_storage": "per_request_v1",
        "label": config.data.label,
        "source_files": list(config.data.file_names()),
        "source_file_rows": {source.name: source.row_count for source in sources},
        "source_file_count": len(sources),
        "event_count": int(exposure.size),
        "event_first_timestamp_ms": int(exposure.request_timestamp.min()),
        "event_last_timestamp_ms": int(exposure.request_timestamp.max()),
        "request_count": int(request_index.size),
        "distinct_users": int(np.unique(exposure.user_id).size),
        "distinct_videos": int(np.unique(exposure.video_id).size),
        "playing_time_negative": int((exposure.playing_time < 0).sum()),
        "playing_time_max_ms": int(exposure.playing_time.max()),
        "duration_nonpositive": int((exposure.duration <= 0).sum()),
        "upload_timestamp_nonpositive": int((exposure.upload_timestamp <= 0).sum()),
        "upload_after_request": int(
            (exposure.upload_timestamp > exposure.request_timestamp).sum()
        ),
        "max_history_length": config.data.max_history_length,
        "vocabulary_sizes": dict(vocabulary.sizes),
        "feedback_sizes": dict(vocabulary.feedback_sizes),
        "numerical": numerical.to_json(),
        "splits": split_metadata,
        "history_conditions": {
            "source": config.data.history_source,
            "same_user": True,
            "different_request": True,
            "request_time_strictly_before": True,
            "availability": config.data.history_availability,
            # 当前条件验证请求时间，原始数据未确认反馈实际可用时间。
            "feedback_completion_time_verified": False,
            "feedback_fields": list(config.data.history_feedback_fields),
            "effective_view_required": 1,
            "order": "(request_timestamp, request_id, event_id)",
            "padding_side": config.data.padding_side,
            "policy": "author_supplied_members_with_strict_request_time_filter",
        },
        "environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "prepare_seconds": float(time.perf_counter() - started),
        "host_memory": _host_memory(),
    }
    metadata["official_history_audit"] = official_audit
    metadata["processed_format_version"] = config.schema_version
    root = save_processed(
        config,
        exposure,
        request_index,
        vocabulary,
        numerical,
        metadata,
        target_rows,
        history_rows,
        history_counts,
        history_requests,
        history_target_positions,
    )
    # 全部加工文件生成后，补充实际文件大小和最终耗时。
    sizes = {
        path.name: int(path.stat().st_size)
        for path in sorted(Path(root).rglob("*"))
        if path.is_file()
    }
    metadata["processed_files"] = sizes
    metadata["processed_bytes"] = int(sum(sizes.values()))
    metadata["prepare_seconds"] = float(time.perf_counter() - started)
    metadata["host_memory"] = _host_memory()
    write_json(Path(root) / "metadata.json", metadata)
    return metadata
