import numpy as np
import torch
from torch import nn

from ..data.batch import BatchReader


def collect_logits(
    model: nn.Module,
    reader: BatchReader,
    split: str,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits, labels, event_id, _, _, _, _ = collect_predictions(
        model, reader, split, batch_size, device
    )
    return logits, labels, event_id


def collect_predictions(
    model: nn.Module,
    reader: BatchReader,
    split: str,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logit_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    event_parts: list[np.ndarray] = []
    request_parts: list[np.ndarray] = []
    duration_parts: list[np.ndarray] = []
    video_unknown_parts: list[np.ndarray] = []
    author_unknown_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in reader.iter_batches(
            split, batch_size, shuffle=False, generator=None
        ):
            tensors = batch.as_tensors(device)
            logits = model(tensors)
            if logits.shape != (batch.size,):
                raise ValueError(
                    f"模型输出形状为 {tuple(logits.shape)}，期望 ({batch.size},)"
                )
            if not torch.isfinite(logits).all():
                raise ValueError("模型输出含有非有限数值")
            logit_parts.append(logits.detach().to("cpu", torch.float64).numpy())
            label_parts.append(batch.labels.astype(np.float64))
            event_parts.append(batch.event_id.astype(np.int64))
            request_parts.append(batch.request_id.astype(np.int64))
            duration_parts.append(batch.duration_ms.astype(np.int64))
            video_unknown_parts.append(batch.video_is_unknown)
            author_unknown_parts.append(batch.author_is_unknown)
    if not logit_parts:
        raise ValueError(f"数据集合 {split} 没有可用批次")
    return (
        np.concatenate(logit_parts),
        np.concatenate(label_parts),
        np.concatenate(event_parts),
        np.concatenate(request_parts),
        np.concatenate(duration_parts),
        np.concatenate(video_unknown_parts),
        np.concatenate(author_unknown_parts),
    )
