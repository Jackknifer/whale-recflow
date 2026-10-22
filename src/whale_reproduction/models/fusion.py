import torch
import torch.nn.functional as F
from torch import nn

from ..config import ModelConfig

class FusionBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        fusion = config.fusion
        dim = config.embedding_dim
        self._dim = dim
        self._eps = config.layer_norm_eps
        if fusion.num_heads != 1:
            raise ValueError("Fusion 使用单个 Head，配置的 num_heads 必须为 1")
        if fusion.projection_bias:
            raise ValueError("Fusion 的线性投影按设计不包含 bias")
        if fusion.attention != "softmax":
            raise ValueError(f"不支持的 Fusion 注意力方式 {fusion.attention}")
        self._query_norm = nn.LayerNorm(dim, eps=self._eps)
        self._sequence_norm = nn.LayerNorm(dim, eps=self._eps)
        self._query = nn.Linear(dim, dim, bias=False)
        self._key = nn.Linear(dim, dim, bias=False)
        self._value = nn.Linear(dim, dim, bias=False)
        self._fusion_projection = nn.Linear(2 * dim, dim, bias=False)
        self._ffn_norm = nn.LayerNorm(dim, eps=self._eps)
        hidden = fusion.swiglu_hidden_dim
        self._gate = nn.Linear(dim, hidden, bias=False)
        self._up = nn.Linear(dim, hidden, bias=False)
        self._down = nn.Linear(hidden, dim, bias=False)

    def project_value(self, sequence: torch.Tensor) -> torch.Tensor:
        return self._value(self._sequence_norm(sequence))

    def attend(
        self,
        nonsequence: torch.Tensor,
        sequence: torch.Tensor,
        history_mask: torch.Tensor,
        value_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        present = history_mask.any(dim=1)
        output = torch.zeros_like(nonsequence)
        if not present.any():
            return output
        selected_sequence = sequence[present]
        query = self._query(self._query_norm(nonsequence[present]))
        key = self._key(self._sequence_norm(selected_sequence))
        value = self.project_value(selected_sequence) if value_override is None else value_override[present]
        scores = query @ key.transpose(1, 2) / self._dim**0.5
        mask = history_mask[present].unsqueeze(1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        output[present] = weights @ value
        return output

    def fuse(self, nonsequence: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        concatenated = torch.cat([nonsequence, attended], dim=-1)
        fused = nonsequence + self._fusion_projection(concatenated)
        normalized = self._ffn_norm(fused)
        gated = F.silu(self._gate(normalized)) * self._up(normalized)
        return fused + self._down(gated)

    def forward(
        self,
        nonsequence: torch.Tensor,
        sequence: torch.Tensor,
        history_mask: torch.Tensor,
        value_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended = self.attend(nonsequence, sequence, history_mask, value_override)
        return self.fuse(nonsequence, attended)
