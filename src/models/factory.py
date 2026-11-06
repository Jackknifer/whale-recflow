from dataclasses import replace

from ..config import ModelConfig
from .base import BaseModel
from .embedding import VocabularySizes
from .hstu import HSTUModel
from .whale import WhaleModel
from .wukong import WukongModel

MODEL_NAMES = ("wukong", "hstu", "whale")


# 模型名称同时用于配置、结果文件和模型构造。
MODEL_REGISTRY = {
    "wukong": WukongModel,
    "hstu": HSTUModel,
    "whale": WhaleModel,
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
    # 构造独立配置，确保共用输入编码采用对应模型的输入方式。
    return MODEL_REGISTRY[name](
        replace(config, name=name),
        vocabulary,
        max_history_length,
        categorical_fields,
        history_video_fields,
        history_feedback_fields,
    )
