from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class Metrics:
    # 指标与评估成员数量一起保存，便于检查不同结果的比较条件。
    auc: float
    logloss: float
    ne: float
    count: int
    positive_count: int
    positive_rate: float

    def to_json(self) -> dict[str, float | int]:
        return {
            "auc": self.auc,
            "logloss": self.logloss,
            "ne": self.ne,
            "count": self.count,
            "positive_count": self.positive_count,
            "positive_rate": self.positive_rate,
        }


def compute_metrics(
    logits: np.ndarray, labels: np.ndarray, probability_dtype: str, single_class_policy: str
) -> Metrics:
    # 整体评估要求完整的二分类样本，先验证输入和概率计算精度。
    validate_predictions(logits, labels)
    if logits.size == 0:
        raise ValueError("评估集合为空")
    if probability_dtype != "float64":
        raise ValueError(f"概率计算要求 float64，配置为 {probability_dtype}")
    if single_class_policy != "error":
        raise ValueError(f"不支持的单一类别处理方式 {single_class_policy}")
    binary = np.unique(labels)
    if binary.size < 2:
        raise ValueError(f"评估集合只包含单一类别 {binary.tolist()}，无法计算指标")

    logits64 = logits.astype(np.float64)
    labels64 = labels.astype(np.float64)
    auc = float(roc_auc_score(labels64, logits64))
    loss = binary_logloss(logits64, labels64)
    positive_rate = float(labels64.mean())
    # NE 的分母为当前评估集合正例比例对应的二元熵。
    entropy = -(positive_rate * np.log(positive_rate) + (1.0 - positive_rate) * np.log1p(-positive_rate))
    if entropy <= 0.0:
        raise ValueError(f"正例比例为 {positive_rate}，参考熵为零，无法计算 NE")
    return Metrics(
        auc=auc,
        logloss=loss,
        ne=loss / entropy,
        count=int(labels64.size),
        positive_count=int(labels64.sum()),
        positive_rate=positive_rate,
    )


def validate_predictions(logits: np.ndarray, labels: np.ndarray) -> None:
    if logits.ndim != 1 or labels.ndim != 1:
        raise ValueError(f"指标计算需要一维输入，实际形状为 {logits.shape} 与 {labels.shape}")
    if logits.size != labels.size:
        raise ValueError(f"预测数量 {logits.size} 与标签数量 {labels.size} 不一致")
    if not np.isfinite(logits).all():
        raise ValueError("预测 logits 含有非有限数值")
    binary = np.unique(labels)
    if not np.isin(binary, (0.0, 1.0)).all():
        raise ValueError(f"标签出现非二值取值 {binary}")


def binary_logloss(logits: np.ndarray, labels: np.ndarray) -> float:
    validate_predictions(logits, labels)
    if logits.size == 0:
        raise ValueError("LogLoss 计算需要有效样本")
    # 直接计算 log(1 + exp(-y*z))，其中 y 为 -1 或 1。
    signed = np.where(labels == 1, -logits.astype(np.float64), logits.astype(np.float64))
    losses = np.logaddexp(0.0, signed)
    return float(np.sum(losses / losses.size))
