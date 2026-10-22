from typing import Any

import torch
from torch import nn

from ..models.fusion import FusionBlock
from ..models.hstu import HSTUBlock
from ..models.wukong import WukongBlock


def forward_compute(model: nn.Module, tensors: dict[str, torch.Tensor], max_history_length: int) -> dict[str, Any]:
    counts = {"linear": 0, "wukong_interactions": 0, "hstu_attention": 0, "fusion_attention": 0}

    def count_linear(module, inputs, output):
        counts["linear"] += 2 * inputs[0].numel() * module.out_features

    def count_wukong(module, inputs, output):
        batch, tokens, dim = inputs[0].shape
        counts["wukong_interactions"] += 2 * batch * tokens * tokens * dim
        counts["wukong_interactions"] += 2 * batch * module._lcb_tokens * tokens * dim

    def count_hstu(module, inputs, output):
        batch, tokens, dim = inputs[0].shape
        projected = module._num_heads * (2 * module._value_dim + 2 * module._attention_dim)
        counts["linear"] += 2 * batch * tokens * dim * projected
        counts["hstu_attention"] += 2 * batch * module._num_heads * tokens * tokens * (module._attention_dim + module._value_dim)

    def count_fusion(module, inputs, output):
        nonsequence, sequence, mask = inputs[:3]
        active = int(mask.any(dim=1).sum())
        counts["fusion_attention"] += 4 * active * nonsequence.size(1) * sequence.size(1) * module._dim

    handles = []
    for module in model.modules():
        for kind, hook in ((nn.Linear, count_linear), (WukongBlock, count_wukong), (HSTUBlock, count_hstu), (FusionBlock, count_fusion)):
            if isinstance(module, kind):
                handles.append(module.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(tensors)
    for handle in handles:
        handle.remove()
    batch_size = tensors["history_mask"].size(0)
    ranking = getattr(model, "_ranking_input", None)
    return {
        "matrix_forward_flops_per_sample": sum(counts.values()) / batch_size,
        "batch_size": batch_size,
        "matrix_forward_flops_by_operation": counts,
        "history_behaviors": max_history_length,
        "max_sequence_tokens": ranking.max_tokens if ranking is not None else (0 if model._config.name == "wukong" else max_history_length),
        "scope": "实际前向中的矩阵乘加，每次乘加计为 2 FLOPs；不包含查表、归一化、逐元素操作、反向和优化器。",
    }
