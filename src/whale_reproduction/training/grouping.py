from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

DURATION_MISSING_GROUP = "missing"
DURATION_ALL_POSITIVE_GROUP = "all_positive"


def sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _safe_logloss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(log_loss(labels, probabilities, labels=[0.0, 1.0], normalize=True))


def _ne(logloss: float, positive_rate: float) -> float | None:
    if positive_rate <= 0.0 or positive_rate >= 1.0:
        return None
    entropy = -(
        positive_rate * np.log(positive_rate)
        + (1.0 - positive_rate) * np.log(1.0 - positive_rate)
    )
    if entropy <= 0.0:
        return None
    return float(logloss / entropy)


def group_metrics(
    labels: np.ndarray, logits: np.ndarray, name: str, lower, upper
) -> dict[str, Any]:
    count = int(labels.size)
    positive = int(labels.sum())
    positive_rate = float(labels.mean()) if count else None
    entry: dict[str, Any] = {
        "group": name,
        "lower_seconds": None if lower is None else float(lower),
        "upper_seconds": None if upper is None else float(upper),
        "count": count,
        "positive_count": positive,
        "positive_rate": positive_rate,
        "auc": None,
        "logloss": None,
        "ne": None,
        "notes": [],
    }
    if count == 0:
        entry["notes"].append("空分组，指标为空")
        return entry
    probabilities = sigmoid(logits.astype(np.float64))
    entry["logloss"] = _safe_logloss(labels.astype(np.float64), probabilities)
    if positive == 0 or positive == count:
        entry["notes"].append("单一类别，AUC 与 NE 为空")
        return entry
    entry["auc"] = float(roc_auc_score(labels.astype(np.float64), probabilities))
    entry["ne"] = _ne(entry["logloss"], positive_rate)
    return entry


def duration_groups(
    config,
    labels: np.ndarray,
    logits: np.ndarray,
    duration_ms: np.ndarray,
) -> list[dict[str, Any]]:
    seconds = duration_ms.astype(np.float64) / 1000.0
    missing = duration_ms <= 0
    groups: list[dict[str, Any]] = []
    groups.append(
        group_metrics(
            labels[missing], logits[missing], DURATION_MISSING_GROUP, None, None
        )
    )
    observed = ~missing
    groups.append(
        group_metrics(
            labels[observed],
            logits[observed],
            DURATION_ALL_POSITIVE_GROUP,
            None,
            None,
        )
    )
    for lower, upper in config.evaluation.duration_groups:
        if upper is None:
            selected = observed & (seconds > lower)
            name = f"({lower:g},+inf)"
        else:
            selected = observed & (seconds > lower) & (seconds <= upper)
            name = f"({lower:g},{upper:g}]"
        groups.append(group_metrics(labels[selected], logits[selected], name, lower, upper))
    return groups


def relative_improvement(baseline: float | None, current: float | None) -> float | None:
    if baseline is None or current is None:
        return None
    if baseline == 0.0:
        return None
    return float((baseline - current) / baseline)


def id_groups(labels: np.ndarray, logits: np.ndarray, video_unknown: np.ndarray, author_unknown: np.ndarray) -> list[dict[str, Any]]:
    groups = []
    for field, unknown in (("video", video_unknown), ("author", author_unknown)):
        if unknown.dtype != np.bool_ or unknown.shape != labels.shape:
            raise ValueError(f"{field} 的自然 UNK 标记类型或形状不符")
        for name, selected in (("known", ~unknown), ("unknown", unknown)):
            record = group_metrics(labels[selected], logits[selected], f"{field}_{name}", None, None)
            record.pop("lower_seconds")
            record.pop("upper_seconds")
            groups.append(record)
    return groups


def build_deltas(
    records: dict[str, dict[str, float]],
    reference: str = "whale",
    baselines: tuple[str, ...] = ("shallow_hybrid", "whale_mean_pool"),
) -> dict[str, Any]:
    if reference not in records:
        raise KeyError(f"差值计算缺少参照模型 {reference}")
    reference_metrics = records[reference]
    deltas: dict[str, Any] = {}
    for name in baselines:
        if name not in records:
            raise KeyError(f"差值计算缺少对比模型 {name}")
        baseline = records[name]
        deltas[f"{reference}_minus_{name}"] = {
            "auc": _difference(reference_metrics, baseline, "auc"),
            "logloss": _difference(reference_metrics, baseline, "logloss"),
            "ne": _difference(reference_metrics, baseline, "ne"),
            "ne_relative_improvement": relative_improvement(
                baseline.get("ne"), reference_metrics.get("ne")
            ),
            "logloss_relative_improvement": relative_improvement(
                baseline.get("logloss"), reference_metrics.get("logloss")
            ),
        }
    return deltas


def _difference(
    reference: dict[str, float], baseline: dict[str, float], name: str
) -> float | None:
    left = reference.get(name)
    right = baseline.get(name)
    if left is None or right is None:
        return None
    return float(left - right)


def duration_frame(groups: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(groups)
