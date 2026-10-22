import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..utils import current_timestamp, write_json
from .history import HistoryReader, build_history_index
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
STATISTICS_CHUNK = 20000


def _host_memory() -> dict[str, int]:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "peak_rss_bytes": int(usage.ru_maxrss) * (1 if sys.platform == "darwin" else 1024),
    }


def numerical_inputs(
    exposure: ExposureTable, rows: np.ndarray
) -> dict[str, np.ndarray]:
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
) -> dict[str, Any]:
    lengths = history_counts[target_positions].astype(np.int64)
    weights = np.bincount(target_positions, minlength=history_counts.size)
    effective_positive = 0
    long_positive = 0
    total_positions = 0
    unknown_counts = {field: 0 for field in (unknown_masks or {})}
    for start in range(0, history_rows.shape[0], HISTORY_CHUNK):
        end = min(start + HISTORY_CHUNK, history_rows.shape[0])
        chunk = history_rows[start:end]
        valid = chunk >= 0
        safe = np.maximum(chunk, 0)
        multiplicity = weights[start:end]
        effective_positive += int(np.dot(
            (exposure.effective_view[safe] * valid).sum(axis=1), multiplicity
        ))
        long_positive += int(np.dot(
            (exposure.long_view[safe] * valid).sum(axis=1), multiplicity
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
        "c64": float((lengths >= 64).mean()),
        "c128": float((lengths >= max_length).mean()),
        "history_effective_view_positive": effective_positive,
        "history_long_view_positive": long_positive,
        "history_total_positions": total_positions,
        **({"history_unknown_ratio": {
            field: count / total_positions if total_positions else None
            for field, count in unknown_counts.items()
        }} if unknown_masks is not None else {}),
    }


def unknown_ratios(
    vocabulary, exposure: ExposureTable, rows: np.ndarray
) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for field in vocabulary.raw_values:
        encoded = vocabulary.encode(field, exposure.categorical(field)[rows])
        ratios[field] = float((encoded == vocabulary.unknown_index).mean())
    return ratios


def prepare(config: Config, *, baseline_config: Config | None = None) -> dict[str, Any]:
    directory = config.processed_directory()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"加工目录已有内容，不能覆盖已有数据：{directory}")
    started = time.perf_counter()
    windows = split_windows(config)
    exposure, sources = read_exposure(config)
    request_index = build_request_index(exposure, config.data.request_gap_ms)
    candidates = candidate_rows(exposure, config)
    target_rows = sample_targets(exposure, config, candidates, baseline_config=baseline_config)

    history_index = build_history_index(exposure, request_index)
    reader = HistoryReader(history_index, config.data.max_history_length)
    history_rows: dict[str, np.ndarray] = {}
    history_counts: dict[str, np.ndarray] = {}
    history_requests: dict[str, np.ndarray] = {}
    history_target_positions: dict[str, np.ndarray] = {}
    for split in SPLIT_ORDER:
        rows = target_rows[split]
        requests, first, inverse = np.unique(
            exposure.request_id[rows], return_index=True, return_inverse=True
        )
        representative = rows[first]
        collected = np.full(
            (requests.size, config.data.max_history_length), -1, dtype=np.int32
        )
        counts = np.zeros(requests.size, dtype=np.int32)
        for start in range(0, requests.size, HISTORY_CHUNK):
            end = min(start + HISTORY_CHUNK, requests.size)
            chunk = representative[start:end]
            chunk_rows, chunk_counts = reader.gather(
                exposure.user_id[chunk].astype(np.int64),
                exposure.request_timestamp[chunk],
                exposure.request_id[chunk].astype(np.int64),
            )
            collected[start:end] = chunk_rows.astype(np.int32)
            counts[start:end] = chunk_counts
        history_rows[split] = collected
        history_counts[split] = counts
        history_requests[split] = requests.astype(np.int64)
        history_target_positions[split] = inverse.astype(np.int32)
    del reader, history_index

    train_rows = target_rows["train"]
    train_history = history_rows["train"]
    vocabulary = fit_vocabulary(config, exposure, train_rows, train_history)
    numerical = fit_numerical_parameters(numerical_inputs(exposure, train_rows))
    unknown_masks = None
    if config.schema_version == 3:
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
            "available_time_strictly_before": True,
            "availability": "request_timestamp + sum(playing_time) + request_gap_ms",
            "playing_time_unit": config.data.playing_time_unit,
            "request_gap_ms": config.data.request_gap_ms,
            "order": "(request_timestamp, request_id, event_id)",
            "padding_side": config.data.padding_side,
            "policy": "rolling_observed",
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
    if config.schema_version == 3:
        metadata["processed_format_version"] = 3
    if baseline_config is not None:
        metadata["reconstructed_baseline_config"] = baseline_config.snapshot()
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
