from dataclasses import replace

import torch
from torch import nn

from ..config import ModelConfig
from .embedding import InputEmbedding, VocabularySizes
from .fusion import FusionBlock
from .head import PredictionHead
from .hstu import HSTUBlock, HSTUStack
from .ranking_input import RankingInput
from .wukong import WukongBlock, WukongStack

MODEL_NAMES = ("wukong", "hstu", "shallow_hybrid", "whale", "whale_mean_pool")


class BaseModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        vocabulary: VocabularySizes,
        max_history_length: int,
        categorical_fields: tuple[str, ...],
        history_video_fields: tuple[str, ...],
        history_feedback_fields: tuple[str, ...],
    ) -> None:
        super().__init__()
        self._config = config
        self._max_history_length = max_history_length
        self._embedding = InputEmbedding(
            config,
            vocabulary,
            categorical_fields,
            history_video_fields,
            history_feedback_fields,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def describe(self) -> dict[str, object]:
        return {"name": self._config.name}

    def _prepare(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._embedding(batch)

    def _flatten_nonsequence(self, nonsequence: torch.Tensor) -> torch.Tensor:
        return nonsequence.reshape(nonsequence.size(0), -1)

    def _head_input_dim(self) -> int:
        return self._config.input_tokens * self._config.embedding_dim

    @property
    def history_input_dim(self) -> int:
        return self._embedding.history_input_dim


class WukongModel(BaseModel):
    def __init__(
        self,
        config: ModelConfig,
        vocabulary: VocabularySizes,
        max_history_length: int,
        categorical_fields: tuple[str, ...],
        history_video_fields: tuple[str, ...],
        history_feedback_fields: tuple[str, ...],
    ) -> None:
        super().__init__(
            config,
            vocabulary,
            max_history_length,
            categorical_fields,
            history_video_fields,
            history_feedback_fields,
        )
        self._stack = WukongStack(config, config.input_tokens)
        self._head = PredictionHead(
            config, config.nonsequence_tokens * config.embedding_dim
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        nonsequence = self._embedding.nonsequence_embedding(batch)
        representation = self._stack(nonsequence)
        return self._head(self._flatten_nonsequence(representation))


class HSTUModel(BaseModel):
    def __init__(
        self,
        config: ModelConfig,
        vocabulary: VocabularySizes,
        max_history_length: int,
        categorical_fields: tuple[str, ...],
        history_video_fields: tuple[str, ...],
        history_feedback_fields: tuple[str, ...],
    ) -> None:
        super().__init__(
            config,
            vocabulary,
            max_history_length,
            categorical_fields,
            history_video_fields,
            history_feedback_fields,
        )
        self._ranking_input = None
        if config.hstu.ranking_input == "interleaved":
            self._ranking_input = RankingInput(config, max_history_length, len(history_video_fields), len(history_feedback_fields))
            self._stack = HSTUStack(config, self._ranking_input.max_tokens)
            self._head = PredictionHead(config, config.embedding_dim)
        elif config.hstu.ranking_input == "pooled":
            self._stack = HSTUStack(config, max_history_length)
            self._head = PredictionHead(config, self._head_input_dim() + config.embedding_dim)
        else:
            raise ValueError(f"未知排序输入方式 {config.hstu.ranking_input}")

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self._ranking_input is not None:
            ranking = self._ranking_input(batch, self._embedding)
            encoded = self._stack(ranking.tokens, ranking.mask, ranking.time_bucket, ranking.causal_mask)
            rows = torch.arange(encoded.size(0), device=encoded.device)
            return self._head(encoded[rows, ranking.candidate_positions])
        nonsequence, sequence = self._prepare(batch)
        mask = batch["history_mask"]
        encoded = self._stack(
            sequence, mask, batch["time_bucket"], batch["causal_mask"]
        )
        pooled = self._stack.pooled_representation(encoded, mask)
        features = torch.cat(
            [self._flatten_nonsequence(nonsequence), pooled], dim=-1
        )
        return self._head(features)


class ShallowHybridModel(BaseModel):
    def __init__(
        self,
        config: ModelConfig,
        vocabulary: VocabularySizes,
        max_history_length: int,
        categorical_fields: tuple[str, ...],
        history_video_fields: tuple[str, ...],
        history_feedback_fields: tuple[str, ...],
    ) -> None:
        super().__init__(
            config,
            vocabulary,
            max_history_length,
            categorical_fields,
            history_video_fields,
            history_feedback_fields,
        )
        self._stack = HSTUStack(config, max_history_length)
        self._wukong = WukongStack(config, config.input_tokens + 1)
        self._head = PredictionHead(
            config, config.nonsequence_tokens * config.embedding_dim
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        nonsequence, sequence = self._prepare(batch)
        mask = batch["history_mask"]
        encoded = self._stack(
            sequence, mask, batch["time_bucket"], batch["causal_mask"]
        )
        pooled = self._stack.pooled_representation(encoded, mask)
        tokens = torch.cat([nonsequence, pooled.unsqueeze(1)], dim=1)
        representation = self._wukong(tokens)
        return self._head(self._flatten_nonsequence(representation))


class WhaleModel(BaseModel):
    def __init__(
        self,
        config: ModelConfig,
        vocabulary: VocabularySizes,
        max_history_length: int,
        categorical_fields: tuple[str, ...],
        history_video_fields: tuple[str, ...],
        history_feedback_fields: tuple[str, ...],
    ) -> None:
        super().__init__(
            config,
            vocabulary,
            max_history_length,
            categorical_fields,
            history_video_fields,
            history_feedback_fields,
        )
        input_tokens = config.input_tokens
        self._wukong_blocks = nn.ModuleList()
        self._hstu_blocks = nn.ModuleList()
        self._fusion_blocks = nn.ModuleList()
        for layer in range(config.num_layers):
            self._wukong_blocks.append(
                WukongBlock(config, input_tokens if layer == 0 else config.nonsequence_tokens)
            )
            self._hstu_blocks.append(HSTUBlock(config, max_history_length))
            self._fusion_blocks.append(FusionBlock(config))
        self._head = PredictionHead(
            config, config.nonsequence_tokens * config.embedding_dim
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        nonsequence, sequence = self._prepare(batch)
        mask = batch["history_mask"]
        time_bucket = batch["time_bucket"]
        causal_mask = batch["causal_mask"]
        for wukong, hstu, fusion in zip(
            self._wukong_blocks, self._hstu_blocks, self._fusion_blocks
        ):
            nonsequence = wukong(nonsequence)
            sequence = hstu.forward_masked(sequence, mask, time_bucket, causal_mask)
            nonsequence = fusion(nonsequence, sequence, mask)
        return self._head(self._flatten_nonsequence(nonsequence))


class WhaleMeanPoolModel(BaseModel):
    def __init__(
        self,
        config: ModelConfig,
        vocabulary: VocabularySizes,
        max_history_length: int,
        categorical_fields: tuple[str, ...],
        history_video_fields: tuple[str, ...],
        history_feedback_fields: tuple[str, ...],
    ) -> None:
        super().__init__(
            config,
            vocabulary,
            max_history_length,
            categorical_fields,
            history_video_fields,
            history_feedback_fields,
        )
        input_tokens = config.input_tokens
        self._wukong_blocks = nn.ModuleList()
        self._hstu_blocks = nn.ModuleList()
        self._fusion_blocks = nn.ModuleList()
        for layer in range(config.num_layers):
            self._wukong_blocks.append(
                WukongBlock(config, input_tokens if layer == 0 else config.nonsequence_tokens)
            )
            self._hstu_blocks.append(HSTUBlock(config, max_history_length))
            self._fusion_blocks.append(FusionBlock(config))
        self._head = PredictionHead(
            config, config.nonsequence_tokens * config.embedding_dim
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        nonsequence, sequence = self._prepare(batch)
        mask = batch["history_mask"]
        time_bucket = batch["time_bucket"]
        causal_mask = batch["causal_mask"]
        for wukong, hstu, fusion in zip(
            self._wukong_blocks, self._hstu_blocks, self._fusion_blocks
        ):
            nonsequence = wukong(nonsequence)
            sequence = hstu.forward_masked(sequence, mask, time_bucket, causal_mask)
            projected = fusion.project_value(sequence)
            pooled = masked_mean(projected, mask)
            replicated = pooled.unsqueeze(1).expand(-1, nonsequence.size(1), -1)
            nonsequence = fusion.fuse(nonsequence, replicated)
        return self._head(self._flatten_nonsequence(nonsequence))


def masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(sequence.dtype)
    total = weights.sum(dim=1)
    return (sequence * weights).sum(dim=1) / total.clamp(min=1.0)


MODEL_REGISTRY = {
    "wukong": WukongModel,
    "hstu": HSTUModel,
    "shallow_hybrid": ShallowHybridModel,
    "whale": WhaleModel,
    "whale_mean_pool": WhaleMeanPoolModel,
}


def build_model(
    name: str,
    config: ModelConfig,
    vocabulary: VocabularySizes,
    max_history_length: int,
    categorical_fields: tuple[str, ...],
    history_video_fields: tuple[str, ...],
    history_feedback_fields: tuple[str, ...],
) -> BaseModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型 {name}，可选值为 {MODEL_NAMES}")
    return MODEL_REGISTRY[name](
        replace(config, name=name),
        vocabulary,
        max_history_length,
        categorical_fields,
        history_video_fields,
        history_feedback_fields,
    )
