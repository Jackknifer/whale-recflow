import numpy as np

from ..config import Config
from .raw import ExposureTable

SPLIT_ORDER = ("train", "validation", "test")


def split_windows(config: Config) -> dict[str, tuple[int, int]]:
    # 时间窗口按训练、验证、测试顺序连续排列，边界采用左闭右开规则。
    windows = config.data.time_bounds_ms()
    ordered = [windows[name] for name in SPLIT_ORDER]
    for previous, current in zip(ordered, ordered[1:]):
        if previous[1] != current[0]:
            raise ValueError("时间划分区间需要连续")
    return windows


def candidate_rows(exposure: ExposureTable, config: Config) -> dict[str, np.ndarray]:
    result = {}
    for split, (start, end) in split_windows(config).items():
        rows = np.flatnonzero((exposure.request_timestamp >= start) & (exposure.request_timestamp < end))
        if rows.size == 0:
            raise ValueError(f"时间区间 {split} 没有候选记录")
        result[split] = rows.astype(np.int64)
    return result


def sample_targets(
    exposure: ExposureTable, config: Config, candidates: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    # 首次抽样依次选取训练、验证和测试目标，共用配置种子的随机数序列。
    generator = np.random.Generator(np.random.PCG64(config.seed))
    initial_sizes = {"train": 1000000, **{
        split: config.data.target_samples[split] for split in ("validation", "test")
    }}
    sampled = {}
    for split in SPLIT_ORDER:
        rows = candidates[split]
        count = initial_sizes[split]
        if count > rows.size:
            raise ValueError(f"{split} 候选数量不足 {count}")
        selected = generator.choice(rows.size, size=count, replace=False)
        sampled[split] = np.sort(rows[selected]).astype(np.int64)
    # 使用独立种子从剩余训练候选中补充目标，保留首次抽样的全部成员。
    remaining = np.setdiff1d(candidates["train"], sampled["train"], assume_unique=True)
    additional = config.data.target_samples["train"] - sampled["train"].size
    if additional <= 0 or additional > remaining.size:
        raise ValueError("训练补充抽样数量超出候选范围")
    generator = np.random.Generator(np.random.PCG64(config.data.extension_seed))
    selected = generator.choice(remaining.size, size=additional, replace=False)
    sampled["train"] = np.sort(np.concatenate((sampled["train"], remaining[selected])))
    # 保存前统一检查数量、唯一性和事件顺序，供后续实验共享目标成员。
    for split, rows in sampled.items():
        if rows.size != config.data.target_samples[split] or np.any(np.diff(rows) <= 0):
            raise ValueError(f"{split} 抽样数量、唯一性或顺序无效")
        if np.any(np.diff(exposure.event_id[rows]) <= 0):
            raise ValueError(f"{split} 事件编号顺序无效")
    return sampled
