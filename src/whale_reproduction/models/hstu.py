# HSTU 计算约定参考 Meta 的 Apache-2.0 实现，来源和版权见 NOTICE。
import torch
import torch.nn.functional as F
from torch import nn

from ..config import ModelConfig

POSITION_BIAS_CENTER = 0


class HSTUBlock(nn.Module):
    def __init__(self, config: ModelConfig, max_history_length: int) -> None:
        super().__init__()
        hstu = config.hstu
        dim = config.embedding_dim
        self._dim = dim
        self._num_heads = hstu.num_heads
        self._attention_dim = hstu.attention_dim_per_head
        self._value_dim = hstu.value_dim_per_head
        self._max_length = max_history_length
        self._num_buckets = hstu.relative_time_buckets
        self._log_divisor = hstu.relative_time_log_divisor
        self._eps = config.layer_norm_eps
        if hstu.input_projection_bias:
            raise ValueError("HSTU 的 UVQK 投影按官方参考代码不使用 bias")
        if hstu.time_alignment != "current_event":
            raise ValueError(f"不支持的时间计算方式 {hstu.time_alignment}")
        self._uvqk = nn.Parameter(
            torch.empty(
                dim, self._num_heads * (self._value_dim * 2 + self._attention_dim * 2)
            ).normal_(mean=0.0, std=0.02)
        )
        self._output = nn.Linear(
            self._num_heads * self._value_dim, dim, bias=hstu.output_projection_bias
        )
        nn.init.xavier_uniform_(self._output.weight)
        if self._output.bias is not None:
            nn.init.zeros_(self._output.bias)
        self._position_bias = nn.Parameter(
            torch.empty(2 * max_history_length - 1).normal_(mean=0.0, std=0.02)
        )
        self._time_bias = nn.Parameter(
            torch.empty(self._num_buckets + 1).normal_(mean=0.0, std=0.02)
        )
        indices = torch.arange(max_history_length, dtype=torch.long)
        self.register_buffer(
            "position_index",
            (indices.unsqueeze(0) - indices.unsqueeze(1)) + (max_history_length - 1),
            persistent=False,
        )

    def relative_bias(self, time_bucket: torch.Tensor) -> torch.Tensor:
        position = self._position_bias[self.position_index].unsqueeze(0).unsqueeze(1)
        time = self._time_bias[time_bucket].unsqueeze(1)
        return position + time

    def forward(
        self,
        tokens: torch.Tensor,
        time_bucket: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, length, dim = tokens.shape
        if length != self._max_length:
            raise ValueError(
                f"HSTUBlock 期望历史长度为 {self._max_length}，实际为 {length}"
            )
        normalized = F.layer_norm(tokens, (dim,), eps=self._eps)
        projected = F.silu(normalized @ self._uvqk)
        split_sizes = [
            self._num_heads * self._value_dim,
            self._num_heads * self._value_dim,
            self._num_heads * self._attention_dim,
            self._num_heads * self._attention_dim,
        ]
        u, v, q, k = torch.split(projected, split_sizes, dim=-1)
        u = u.view(batch_size, length, self._num_heads, self._value_dim)
        v = v.view(batch_size, length, self._num_heads, self._value_dim)
        q = q.view(batch_size, length, self._num_heads, self._attention_dim)
        k = k.view(batch_size, length, self._num_heads, self._attention_dim)

        scores = torch.einsum("bihd,bjhd->bhij", q, k)
        scores = scores + self.relative_bias(time_bucket)
        scores = F.silu(scores) / self._max_length
        scores = scores * causal_mask.unsqueeze(1).to(scores.dtype)
        aggregated = torch.einsum("bhij,bjhd->bihd", scores, v)
        aggregated = aggregated.reshape(
            batch_size, length, self._num_heads * self._value_dim
        )
        aggregated = F.layer_norm(
            aggregated, (self._num_heads * self._value_dim,), eps=self._eps
        )
        gated = aggregated * u.reshape(
            batch_size, length, self._num_heads * self._value_dim
        )
        return tokens + self._output(gated)

    def forward_masked(
        self,
        tokens: torch.Tensor,
        history_mask: torch.Tensor,
        time_bucket: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self(tokens, time_bucket, causal_mask)
        return output * history_mask.unsqueeze(-1).to(output.dtype)


class HSTUStack(nn.Module):
    def __init__(self, config: ModelConfig, max_history_length: int) -> None:
        super().__init__()
        self._blocks = nn.ModuleList(
            HSTUBlock(config, max_history_length) for _ in range(config.num_layers)
        )
        self._max_length = max_history_length

    def forward(
        self,
        tokens: torch.Tensor,
        history_mask: torch.Tensor,
        time_bucket: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = history_mask.unsqueeze(-1).to(tokens.dtype)
        for block in self._blocks:
            tokens = block(tokens, time_bucket, causal_mask) * mask
        return tokens

    def pooled_representation(
        self, tokens: torch.Tensor, history_mask: torch.Tensor
    ) -> torch.Tensor:
        weights = history_mask.unsqueeze(-1).to(tokens.dtype)
        total = weights.sum(dim=1)
        return (tokens * weights).sum(dim=1) / total.clamp(min=1.0)
