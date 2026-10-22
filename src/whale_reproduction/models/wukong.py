import torch
from torch import nn

from ..config import ModelConfig


class WukongBlock(nn.Module):
    def __init__(self, config: ModelConfig, input_tokens: int) -> None:
        super().__init__()
        dim = config.embedding_dim
        wukong = config.wukong
        self._input_tokens = input_tokens
        self._fmb_tokens = wukong.fmb_tokens
        self._lcb_tokens = wukong.lcb_tokens
        self._output_tokens = wukong.fmb_tokens + wukong.lcb_tokens
        if self._output_tokens != config.nonsequence_tokens:
            raise ValueError(
                f"FMB 与 LCB 的 Token 数量之和为 {self._output_tokens}，"
                f"与非序列 Token 数量 {config.nonsequence_tokens} 不一致"
            )
        self._gram_norm = nn.LayerNorm(input_tokens * input_tokens, eps=config.layer_norm_eps)
        layers: list[nn.Module] = []
        previous = input_tokens * input_tokens
        for hidden in wukong.mlp_hidden_dims:
            layers.append(nn.Linear(previous, hidden))
            layers.append(nn.SiLU())
            previous = hidden
        layers.append(nn.Linear(previous, self._fmb_tokens * dim))
        self._fmb_mlp = nn.Sequential(*layers)
        self._lcb_weight = nn.Parameter(torch.empty(self._lcb_tokens, input_tokens))
        nn.init.xavier_uniform_(self._lcb_weight)
        if input_tokens == config.nonsequence_tokens:
            self._residual: nn.Module | None = None
        else:
            projection = nn.Linear(input_tokens, config.nonsequence_tokens, bias=False)
            with torch.no_grad():
                projection.weight.zero_()
                projection.weight[:, : config.nonsequence_tokens] = torch.eye(
                    config.nonsequence_tokens
                )
            self._residual = projection
        self._layer_norm_output = nn.LayerNorm(dim, eps=config.layer_norm_eps)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, dim = tokens.shape
        if token_count != self._input_tokens:
            raise ValueError(
                f"WukongBlock 期望 {self._input_tokens} 个 Token，实际为 {token_count}"
            )
        gram = tokens @ tokens.transpose(1, 2)
        flattened = self._gram_norm(gram.reshape(batch_size, token_count * token_count))
        fmb = self._fmb_mlp(flattened).reshape(batch_size, self._fmb_tokens, dim)
        lcb = torch.einsum("ij,bjd->bid", self._lcb_weight, tokens)
        combined = torch.cat([fmb, lcb], dim=1)
        if self._residual is None:
            residual = tokens
        else:
            residual = self._residual(tokens.transpose(1, 2)).transpose(1, 2)
        return self._layer_norm_output(combined + residual)


class WukongStack(nn.Module):
    def __init__(self, config: ModelConfig, first_input_tokens: int) -> None:
        super().__init__()
        blocks = []
        for layer in range(config.num_layers):
            input_tokens = first_input_tokens if layer == 0 else config.nonsequence_tokens
            blocks.append(WukongBlock(config, input_tokens))
        self._blocks = nn.ModuleList(blocks)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self._blocks:
            tokens = block(tokens)
        return tokens
