import json
import hashlib
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import PROJECT_ROOT, Config


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    # 完整写入同目录的中间文件后替换目标文件，防止中断留下部分内容。
    ensure_directory(path.parent)
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def save_checkpoint(payload: Any, path: Path) -> None:
    # 权重和恢复状态均采用同目录文件替换，完成保存后再更新目标文件。
    ensure_directory(path.parent)
    temporary = path.with_name(f".{path.name}.partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在 {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"文件 {path} 的内容不是映射")
    return payload


def current_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def implementation_record() -> dict[str, str]:
    # 运行记录保存源码摘要，用于核验检查结果和恢复状态对应的计算过程。
    paths = sorted((PROJECT_ROOT / "src").rglob("*.py"))
    paths.extend(sorted((PROJECT_ROOT / "scripts").glob("*.py")))
    return {
        str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def format_seconds(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def seed_everything(seed: int) -> None:
    # 同步初始化 Python、NumPy、CPU 和 CUDA 的随机数状态。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(config: Config) -> torch.device:
    # 正式实验要求 NVIDIA CUDA，设备不符合要求时立即终止。
    if config.runtime.device != "cuda":
        raise ValueError(f"正式实验要求 CUDA，配置为 {config.runtime.device}")
    if not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA 设备，当前环境没有可用设备；禁止更换设备继续运行")
    device = torch.device("cuda")
    index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    if "NVIDIA" not in name:
        raise RuntimeError(f"当前 CUDA 设备名称为 {name}，无法确认为 NVIDIA 显卡")
    return device


def environment_record(device: torch.device) -> dict[str, Any]:
    # 记录运行环境与设备信息，供检查、恢复训练和实验结果追溯使用。
    if device.type == "cpu":
        return {
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "device": "cpu", "device_memory_bytes": None,
            "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "numpy": np.__version__,
            "project_root": str(PROJECT_ROOT), "recorded_at": current_timestamp(),
        }
    if device.type != "cuda":
        raise ValueError(f"不支持的记录设备 {device}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(index),
        "device_capability": f"{properties.major}.{properties.minor}",
        "device_memory_bytes": properties.total_memory,
        "device_memory_gib": round(properties.total_memory / 1024**3, 2),
        "multi_processor_count": properties.multi_processor_count,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "project_root": str(PROJECT_ROOT),
        "recorded_at": current_timestamp(),
    }


def rng_states() -> dict[str, Any]:
    # 保存训练实际使用的全部随机数状态，恢复后继续生成相同序列。
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_states(states: dict[str, Any]) -> None:
    # 与 rng_states 成对使用，需在恢复训练和数据迭代之前调用。
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    torch.cuda.set_rng_state_all(states["cuda"])


# 使用单调时钟统计代码区间的经过时间，退出后保留最终耗时。
class Stopwatch:
    def __init__(self) -> None:
        self._start = 0.0
        self._elapsed = 0.0

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._elapsed = time.perf_counter() - self._start

    @property
    def elapsed(self) -> float:
        return self._elapsed
