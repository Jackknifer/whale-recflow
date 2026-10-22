# 交替输入和时间约定参考 HSTU，RecFlow 适配及来源见 NOTICE。
from dataclasses import dataclass

import torch
from torch import nn

from ..config import ModelConfig
from .embedding import InputEmbedding


@dataclass(frozen=True)
class RankingBatch:
    tokens: torch.Tensor
    mask: torch.Tensor
    time_bucket: torch.Tensor
    causal_mask: torch.Tensor
    candidate_positions: torch.Tensor


class RankingInput(nn.Module):
    def __init__(self, config: ModelConfig, max_history_length: int, item_fields: int, action_fields: int) -> None:
        super().__init__()
        if config.hstu.ranking_time_alignment != "next_event":
            raise ValueError("交替排序输入要求 next_event 时间方式")
        self._max_history_length = max_history_length
        self.max_tokens = 2 * max_history_length + 1
        self._buckets = config.hstu.relative_time_buckets
        self._divisor = config.hstu.relative_time_log_divisor
        dim = config.embedding_dim
        hidden = config.sequence_hidden_dim
        self._item = nn.Sequential(nn.Linear(item_fields * dim + 1, hidden), nn.SiLU(), nn.Linear(hidden, dim))
        self._action = nn.Sequential(nn.Linear(action_fields * dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))
        self._candidate = nn.Sequential(nn.Linear(config.input_tokens * dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, batch: dict[str, torch.Tensor], embedding: InputEmbedding) -> RankingBatch:
        history_mask = batch["history_mask"]
        history_time = batch["history_time_ms"]
        target_time = batch["target_time_ms"]
        batch_size, length = history_mask.shape
        if length != self._max_history_length:
            raise ValueError("排序输入的历史长度与配置不一致")
        counts = history_mask.sum(dim=1)
        candidate_positions = counts * 2
        indices = torch.arange(self.max_tokens, device=history_mask.device)
        if not torch.equal(history_mask, indices[:length].unsqueeze(0) < counts.unsqueeze(1)):
            raise ValueError("排序输入要求历史位于左侧、PAD 位于右侧")
        item_values, action_values, candidate_values = embedding.ranking_components(batch)
        items = self._item(item_values)
        actions = self._action(action_values)
        candidates = self._candidate(candidate_values)
        tokens = items.new_zeros((batch_size, self.max_tokens, items.size(-1)))
        tokens[:, :2 * length:2] = items
        tokens[:, 1:2 * length:2] = actions
        batch_indices = torch.arange(batch_size, device=history_mask.device)
        tokens[batch_indices, candidate_positions] = candidates
        valid = indices.unsqueeze(0) <= candidate_positions.unsqueeze(1)
        tokens = tokens * valid.unsqueeze(-1).to(tokens.dtype)

        time = torch.cat((history_time.repeat_interleave(2, dim=1), target_time.unsqueeze(1)), dim=1)
        time = torch.where(indices.unsqueeze(0) >= candidate_positions.unsqueeze(1), target_time.unsqueeze(1), time)
        extended = torch.cat((time, target_time.unsqueeze(1)), dim=1)
        difference = extended[:, 1:, None] - time[:, None, :]
        bucket = (torch.log(difference.abs().clamp(min=1).to(torch.float64)) / self._divisor).to(torch.long)
        bucket = bucket.clamp(min=0, max=self._buckets)
        row = indices[:, None]
        column = indices[None, :]
        ordered = column <= row
        same_event = (column // 2) == (row // 2)
        earlier = time[:, None, :] < time[:, :, None]
        candidate_row = row.unsqueeze(0) == candidate_positions[:, None, None]
        causal = valid[:, :, None] & valid[:, None, :] & ordered.unsqueeze(0)
        causal = causal & (same_event.unsqueeze(0) | earlier | candidate_row)
        return RankingBatch(tokens, valid, bucket, causal, candidate_positions)
