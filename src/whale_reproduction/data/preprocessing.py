from dataclasses import dataclass

import numpy as np

DAY_MS = 86_400_000
SHANGHAI_OFFSET_MS = 8 * 3600 * 1000

STANDARDIZED_FEATURES = ("log_video_duration", "log_video_age_days")
NUMERICAL_FEATURE_ORDER = (
    "log_video_duration",
    "video_duration_missing",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "log_video_age_days",
    "video_age_missing",
)
MISSING_PARTNERS = {
    "log_video_duration": "video_duration_missing",
    "log_video_age_days": "video_age_missing",
}


@dataclass(frozen=True)
class NumericalParameters:
    fill_values: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]

    def to_json(self) -> dict[str, object]:
        return {
            "fill_values": dict(self.fill_values),
            "means": dict(self.means),
            "stds": dict(self.stds),
            "standardized_features": list(STANDARDIZED_FEATURES),
            "feature_order": list(NUMERICAL_FEATURE_ORDER),
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "NumericalParameters":
        return cls(
            fill_values={
                key: float(value) for key, value in payload["fill_values"].items()
            },
            means={key: float(value) for key, value in payload["means"].items()},
            stds={key: float(value) for key, value in payload["stds"].items()},
        )


def numerical_group_slices(config) -> tuple[slice, ...]:
    slices: list[slice] = []
    position = 0
    for group in config.data.numerical_groups:
        width = len(group)
        slices.append(slice(position, position + width))
        position += width
    if position != len(NUMERICAL_FEATURE_ORDER):
        raise ValueError(
            f"配置的数值分组共 {position} 个特征，"
            f"实现为 {len(NUMERICAL_FEATURE_ORDER)} 个"
        )
    return tuple(slices)


def local_day_index(time_ms: np.ndarray) -> np.ndarray:
    return (time_ms + SHANGHAI_OFFSET_MS) // DAY_MS


def request_time_features(time_ms: np.ndarray) -> tuple[np.ndarray, ...]:
    local = time_ms + SHANGHAI_OFFSET_MS
    milliseconds_of_day = local % DAY_MS
    hour = milliseconds_of_day.astype(np.float64) / 3_600_000.0
    day_index = local // DAY_MS
    weekday = ((day_index + 3) % 7).astype(np.float64)
    hour_angle = 2.0 * np.pi * hour / 24.0
    weekday_angle = 2.0 * np.pi * weekday / 7.0
    return (
        np.sin(hour_angle),
        np.cos(hour_angle),
        np.sin(weekday_angle),
        np.cos(weekday_angle),
    )


def duration_missing_mask(duration_ms: np.ndarray) -> np.ndarray:
    return duration_ms <= 0


def video_age_features(
    request_timestamp: np.ndarray,
    upload_timestamp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    missing = (upload_timestamp <= 0) | (upload_timestamp > request_timestamp)
    safe_upload = np.where(missing, request_timestamp, upload_timestamp)
    delta = (request_timestamp - safe_upload).astype(np.float64)
    age_days = delta / DAY_MS
    return age_days, missing


def raw_numerical_values(
    duration_ms: np.ndarray,
    request_timestamp: np.ndarray,
    upload_timestamp: np.ndarray,
) -> dict[str, np.ndarray]:
    duration_missing = duration_missing_mask(duration_ms)
    safe_duration = np.where(duration_missing, 0, duration_ms).astype(np.float64)
    log_duration = np.log1p(np.maximum(safe_duration, 0.0) / 1000.0)
    hour_sin, hour_cos, weekday_sin, weekday_cos = request_time_features(
        request_timestamp
    )
    age_days, age_missing = video_age_features(request_timestamp, upload_timestamp)
    log_age = np.log1p(np.maximum(age_days, 0.0))
    return {
        "log_video_duration": log_duration,
        "video_duration_missing": duration_missing.astype(np.float64),
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "weekday_sin": weekday_sin,
        "weekday_cos": weekday_cos,
        "log_video_age_days": log_age,
        "video_age_missing": age_missing.astype(np.float64),
    }


def fit_numerical_parameters(values: dict[str, np.ndarray]) -> NumericalParameters:
    fill_values: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for feature in NUMERICAL_FEATURE_ORDER:
        if feature.endswith("_missing"):
            continue
        partner = MISSING_PARTNERS.get(feature)
        if partner is not None:
            observed = values[partner] < 0.5
            if not observed.any():
                raise ValueError(f"训练集没有任何可用的 {feature} 观测值")
            fill_values[feature] = float(values[feature][observed].mean())
        else:
            fill_values[feature] = float(values[feature].mean())
    filled = _fill_missing(values, fill_values)
    for feature in NUMERICAL_FEATURE_ORDER:
        column = filled[feature]
        means[feature] = float(column.mean())
        stds[feature] = float(column.std(ddof=0))
    return NumericalParameters(fill_values=fill_values, means=means, stds=stds)


def _fill_missing(
    values: dict[str, np.ndarray], fill_values: dict[str, float]
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for feature in NUMERICAL_FEATURE_ORDER:
        column = values[feature].astype(np.float64, copy=True)
        partner = MISSING_PARTNERS.get(feature)
        if partner is not None and feature in fill_values:
            column[values[partner] > 0.5] = fill_values[feature]
        result[feature] = column
    return result


def apply_numerical_parameters(
    values: dict[str, np.ndarray], parameters: NumericalParameters
) -> np.ndarray:
    filled = _fill_missing(values, parameters.fill_values)
    columns = []
    for feature in NUMERICAL_FEATURE_ORDER:
        column = filled[feature]
        if feature in STANDARDIZED_FEATURES:
            column = column - parameters.means[feature]
            std = parameters.stds[feature]
            if std > 0.0:
                column = column / std
        columns.append(column)
    return np.stack(columns, axis=-1).astype(np.float32)


def history_age_seconds(
    target_time_ms: np.ndarray, history_time_ms: np.ndarray
) -> np.ndarray:
    delta = target_time_ms[:, None] - history_time_ms
    return np.log1p(np.maximum(delta, 0).astype(np.float64) / 1000.0).astype(np.float32)
