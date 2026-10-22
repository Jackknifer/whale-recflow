import numpy as np

from ..config import Config
from ..utils import read_json
from .raw import ExposureTable

SPLIT_ORDER = ("train", "validation", "test")


def split_windows(config: Config) -> dict[str, tuple[int, int]]:
    windows = config.data.time_bounds_ms()
    ordered = [windows[name] for name in SPLIT_ORDER]
    for previous, current in zip(ordered, ordered[1:]):
        if previous[1] != current[0]:
            raise ValueError(f"时间划分区间不连续：{previous} 与 {current}")
    return windows


def candidate_rows(exposure: ExposureTable, config: Config) -> dict[str, np.ndarray]:
    windows = split_windows(config)
    time_ms = exposure.request_timestamp
    result: dict[str, np.ndarray] = {}
    for split in SPLIT_ORDER:
        start, end = windows[split]
        rows = np.nonzero((time_ms >= start) & (time_ms < end))[0]
        if rows.size == 0:
            raise ValueError(f"时间区间 {split} 没有候选记录")
        result[split] = rows.astype(np.int64)
    assigned = np.concatenate([result[split] for split in SPLIT_ORDER])
    if assigned.size != np.unique(assigned).size:
        raise ValueError("时间划分出现重复记录")
    return result


def sample_targets(
    exposure: ExposureTable,
    config: Config,
    candidates: dict[str, np.ndarray],
    *,
    baseline_config: Config | None = None,
) -> dict[str, np.ndarray]:
    if config.schema_version == 3:
        return extend_baseline_targets(exposure, config, candidates, baseline_config)
    if baseline_config is not None:
        raise ValueError("基线成员重建仅用于版本 3")
    generator = np.random.Generator(np.random.PCG64(config.seed))
    sampled: dict[str, np.ndarray] = {}
    for split in SPLIT_ORDER:
        rows = candidates[split]
        requested = int(config.data.target_samples[split])
        if requested > rows.size:
            raise ValueError(
                f"时间区间 {split} 的候选记录为 {rows.size} 条，不足目标数量 {requested} 条"
            )
        picked = generator.choice(rows.size, size=requested, replace=False)
        selected = np.sort(rows[picked])
        event_ids = exposure.event_id[selected]
        if np.any(event_ids[1:] <= event_ids[:-1]):
            raise ValueError(f"数据集合 {split} 的目标索引没有按 event_id 升序排列")
        sampled[split] = selected.astype(np.int64)
    all_rows = np.concatenate([sampled[split] for split in SPLIT_ORDER])
    if all_rows.size != np.unique(all_rows).size:
        raise ValueError("抽样结果出现重复目标")
    return sampled


def extend_baseline_targets(
    exposure: ExposureTable,
    config: Config,
    candidates: dict[str, np.ndarray],
    baseline_config: Config | None = None,
) -> dict[str, np.ndarray]:
    if baseline_config is not None:
        if baseline_config.schema_version != 2 or baseline_config.seed != config.seed:
            raise ValueError("成员重建要求相同种子的版本 2 配置")
        if baseline_config.data.file_names() != config.data.file_names():
            raise ValueError("成员重建的来源日文件清单不同")
        if baseline_config.data.raw_root != config.data.raw_root:
            raise ValueError("成员重建的原始数据目录不同")
        if baseline_config.data.time_bounds_ms() != config.data.time_bounds_ms():
            raise ValueError("成员重建的时间区间不同")
        expected_sizes = {"train": 1000000, **{
            split: config.data.target_samples[split] for split in ("validation", "test")
        }}
        if baseline_config.data.target_samples != expected_sizes:
            raise ValueError("成员重建的基线样本数量不同")
        baseline = sample_targets(exposure, baseline_config, candidates)
    else:
        baseline_path = config.experiment.baseline_indices
        if baseline_path is None:
            raise ValueError("训练扩充需要基线目标索引")
        path = config.resolve_output_path(baseline_path)
        context = read_json(path.parent / "config.json")["data_context"]
        if context["source_files"] != list(config.data.file_names()) or context["event_count"] != exposure.size:
            raise ValueError("基线来源清单或曝光数量与当前来源不一致")
        file_counts = np.bincount(exposure.file_index, minlength=config.data.expected_file_count)
        expected_counts = [context["source_file_rows"][name] for name in config.data.file_names()]
        if not np.array_equal(file_counts, expected_counts):
            raise ValueError("基线来源文件的逐文件行数与当前来源不一致")
        baseline = {}
        with np.load(path, allow_pickle=False) as archive:
            for split in SPLIT_ORDER:
                rows = archive[f"{split}_rows"].copy()
                expected = 1000000 if split == "train" else config.data.target_samples[split]
                if rows.dtype != np.int64 or rows.shape != (expected,):
                    raise ValueError(f"基线 {split} 索引的类型或数量不符")
                if np.any(rows < 0) or np.any(rows >= exposure.size) or np.any(np.diff(rows) <= 0):
                    raise ValueError(f"基线 {split} 索引范围、顺序或唯一性不符")
                if not np.isin(rows, candidates[split], assume_unique=True).all():
                    raise ValueError(f"基线 {split} 包含时间区间之外的成员")
                if not np.array_equal(archive[f"{split}_event_ids"], exposure.event_id[rows]):
                    raise ValueError(f"基线 {split} 的完整事件编号与当前来源不一致")
                baseline[split] = rows
    remaining = np.setdiff1d(candidates["train"], baseline["train"], assume_unique=True)
    additional = config.data.target_samples["train"] - baseline["train"].size
    if additional <= 0 or additional > remaining.size:
        raise ValueError("训练扩充数量不在剩余候选范围内")
    generator = np.random.Generator(np.random.PCG64(config.data.extension_seed))
    picked = generator.choice(remaining.size, size=additional, replace=False)
    baseline["train"] = np.sort(np.concatenate((baseline["train"], remaining[picked])))
    return baseline


def request_membership(
    exposure: ExposureTable, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    requests = exposure.request_id[rows].astype(np.int64)
    unique_requests, inverse = np.unique(requests, return_inverse=True)
    return unique_requests, inverse.astype(np.int64)
