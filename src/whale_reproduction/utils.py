import json
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
    ensure_directory(path.parent)
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def save_checkpoint(payload: Any, path: Path) -> None:
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


def format_seconds(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(config: Config) -> torch.device:
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


def resolve_dtype(config: Config) -> torch.dtype:
    if config.runtime.dtype != "float32":
        raise ValueError(f"正式实验要求 FP32，配置为 {config.runtime.dtype}")
    return torch.float32


def environment_record(device: torch.device) -> dict[str, Any]:
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
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_states(states: dict[str, Any]) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    torch.cuda.set_rng_state_all(states["cuda"])


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
