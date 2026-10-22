from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_SCHEMA_VERSIONS = (2, 3)


def _field(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"配置缺少字段 {key}")
    return mapping[key]


def _expect(mapping: dict[str, Any], key: str, expected: Any) -> Any:
    value = _field(mapping, key)
    if value != expected:
        raise ValueError(f"配置字段 {key} 的值为 {value!r}，当前实现要求 {expected!r}")
    return value


def _as_tuple(mapping: dict[str, Any], key: str) -> tuple[str, ...]:
    value = _field(mapping, key)
    if not isinstance(value, list):
        raise TypeError(f"配置字段 {key} 需要列表")
    return tuple(value)


def parse_time_ms(text: str, timezone_name: str) -> int:
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        raise ValueError(f"时间字符串 {text} 缺少时区")
    zone = ZoneInfo(timezone_name)
    expected = moment.astimezone(zone).utcoffset()
    if moment.utcoffset() != expected:
        raise ValueError(f"时间字符串 {text} 的时区信息与配置时区 {timezone_name} 不一致")
    return int(round(moment.timestamp() * 1000))


@dataclass(frozen=True)
class DataConfig:
    name: str
    raw_root: Path
    processed_root: Path
    label: str
    timezone: str
    file_start_date: str
    file_end_date: str
    file_extension: str
    expected_file_count: int
    train_start: str
    validation_start: str
    test_start: str
    end_exclusive: str
    target_samples: dict[str, int]
    sampling: str
    max_history_length: int
    history_source: str
    history_availability: str
    playing_time_unit: str
    request_gap_ms: int
    same_request_timestamp: str
    padding_side: str
    vocabulary_fit: str
    padding_index: int
    unknown_index: int
    feedback_index_offset: int
    categorical_fields: tuple[str, ...]
    history_video_fields: tuple[str, ...]
    history_feedback_fields: tuple[str, ...]
    numerical_groups: tuple[tuple[str, ...], ...]
    extension_seed: tuple[int, ...] = ()

    def time_bounds_ms(self) -> dict[str, tuple[int, int]]:
        bounds = {
            "train": (self.train_start, self.validation_start),
            "validation": (self.validation_start, self.test_start),
            "test": (self.test_start, self.end_exclusive),
        }
        return {
            split: (parse_time_ms(start, self.timezone), parse_time_ms(end, self.timezone))
            for split, (start, end) in bounds.items()
        }

    def file_dates(self) -> tuple[str, ...]:
        start = datetime.strptime(self.file_start_date, "%Y-%m-%d").date()
        end = datetime.strptime(self.file_end_date, "%Y-%m-%d").date()
        if end < start:
            raise ValueError(
                f"文件结束日期 {self.file_end_date} 早于开始日期 {self.file_start_date}"
            )
        dates = []
        current = start
        while current <= end:
            dates.append(current.isoformat())
            current += timedelta(days=1)
        if len(dates) != self.expected_file_count:
            raise ValueError(
                f"日期范围得到 {len(dates)} 个文件，配置要求 {self.expected_file_count} 个"
            )
        return tuple(dates)

    def file_names(self) -> tuple[str, ...]:
        return tuple(date + self.file_extension for date in self.file_dates())


@dataclass(frozen=True)
class WukongConfig:
    fm: str
    fmb_tokens: int
    lcb_tokens: int
    mlp_hidden_dims: tuple[int, ...]
    activation: str


@dataclass(frozen=True)
class HSTUConfig:
    num_heads: int
    attention_dim_per_head: int
    value_dim_per_head: int
    attention_normalization: str
    relative_time_buckets: int
    relative_time_log_divisor: float
    time_alignment: str
    input_projection_bias: bool
    output_projection_bias: bool
    norm_affine: bool
    dropout: float
    ranking_input: str = "pooled"
    ranking_time_alignment: str = "next_event"


@dataclass(frozen=True)
class FusionConfig:
    attention: str
    num_heads: int
    projection_bias: bool
    swiglu_hidden_dim: int


@dataclass(frozen=True)
class HeadConfig:
    hidden_dims: tuple[int, ...]
    activation: str
    dropout: float


@dataclass(frozen=True)
class ModelConfig:
    name: str
    embedding_dim: int
    layer_norm_eps: float
    num_layers: int
    input_tokens: int
    nonsequence_tokens: int
    numerical_hidden_dim: int
    sequence_hidden_dim: int
    wukong: WukongConfig
    hstu: HSTUConfig
    fusion: FusionConfig
    head: HeadConfig


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    max_epochs: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    loss: str
    gradient_clip_norm: float
    selection_metric: str
    selection_mode: str
    tie_break: str
    early_stopping_patience: int
    early_stopping_min_delta: float
    checkpoint_on_epoch_end: bool
    log_every_steps: int
    sample_order: str = "global_shuffle"
    id_mask_probability: float = 0.0
    id_mask_fields: tuple[str, ...] = ()
    id_mask_seed_namespace: int = 97


@dataclass(frozen=True)
class EvaluationConfig:
    metrics: tuple[str, ...]
    probability_dtype: str
    single_class_policy: str
    final_test_policy: str
    duration_groups: tuple[tuple[float, float | None], ...]
    duration_group_unit: str


@dataclass(frozen=True)
class RuntimeConfig:
    python: str
    device: str
    dtype: str
    num_workers: int
    allow_device_fallback: bool


@dataclass(frozen=True)
class ExperimentConfig:
    models: tuple[str, ...]
    runs_per_model: int
    parallel_runs: int
    output_root: Path
    baseline_indices: Path | None = None


@dataclass(frozen=True)
class Config:
    path: Path
    schema_version: int
    project: str
    seed: int
    experiment: ExperimentConfig
    runtime: RuntimeConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    @property
    def model_names(self) -> tuple[str, ...]:
        return self.experiment.models

    def resolve_output_path(self, path: Path) -> Path:
        relative = Path(path)
        parts = relative.parts
        # 原基线记录中的输出路径统一对应当前归档目录。
        if len(parts) > 1 and parts[0] == "outputs" and parts[1] in (
            "runs", "checks", "results", "train.log"
        ):
            relative = Path("outputs/baseline").joinpath(*parts[1:])
        return PROJECT_ROOT / relative

    def run_directory(self, model_name: str) -> Path:
        return self.resolve_output_path(
            self.experiment.output_root / "runs" / f"{model_name}-s{self.seed}"
        )

    def results_directory(self) -> Path:
        return self.resolve_output_path(self.experiment.output_root / "results")

    def checks_directory(self) -> Path:
        return self.resolve_output_path(self.experiment.output_root / "checks")

    def processed_directory(self) -> Path:
        return PROJECT_ROOT / self.data.processed_root

    def raw_directory(self) -> Path:
        return PROJECT_ROOT / self.data.raw_root

    def preprocessing_snapshot(self) -> dict[str, Any]:
        data = self.snapshot()["data"]
        del data["raw_root"]
        del data["processed_root"]
        result = {"schema_version": self.schema_version, "seed": self.seed, "data": data}
        if self.schema_version == 3:
            result["baseline_indices"] = str(self.experiment.baseline_indices)
        return result

    def snapshot(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "project": self.project,
            "seed": self.seed,
            "experiment": {
                "models": list(self.experiment.models),
                "runs_per_model": self.experiment.runs_per_model,
                "parallel_runs": self.experiment.parallel_runs,
                "output_root": str(self.experiment.output_root),
            },
            "runtime": {
                "python": self.runtime.python,
                "device": self.runtime.device,
                "dtype": self.runtime.dtype,
                "num_workers": self.runtime.num_workers,
                "allow_device_fallback": self.runtime.allow_device_fallback,
            },
            "data": {
                "name": self.data.name,
                "raw_root": str(self.data.raw_root),
                "processed_root": str(self.data.processed_root),
                "label": self.data.label,
                "timezone": self.data.timezone,
                "file_start_date": self.data.file_start_date,
                "file_end_date": self.data.file_end_date,
                "file_extension": self.data.file_extension,
                "expected_file_count": self.data.expected_file_count,
                "file_names": list(self.data.file_names()),
                "train_start": self.data.train_start,
                "validation_start": self.data.validation_start,
                "test_start": self.data.test_start,
                "end_exclusive": self.data.end_exclusive,
                "target_samples": dict(self.data.target_samples),
                "sampling": self.data.sampling,
                "max_history_length": self.data.max_history_length,
                "history_source": self.data.history_source,
                "history_availability": self.data.history_availability,
                "playing_time_unit": self.data.playing_time_unit,
                "request_gap_ms": self.data.request_gap_ms,
                "same_request_timestamp": self.data.same_request_timestamp,
                "padding_side": self.data.padding_side,
                "vocabulary_fit": self.data.vocabulary_fit,
                "padding_index": self.data.padding_index,
                "unknown_index": self.data.unknown_index,
                "feedback_index_offset": self.data.feedback_index_offset,
                "categorical_fields": list(self.data.categorical_fields),
                "history_video_fields": list(self.data.history_video_fields),
                "history_feedback_fields": list(self.data.history_feedback_fields),
                "numerical_groups": [list(group) for group in self.data.numerical_groups],
            },
            "model": {
                "name": self.model.name,
                "embedding_dim": self.model.embedding_dim,
                "layer_norm_eps": self.model.layer_norm_eps,
                "num_layers": self.model.num_layers,
                "input_tokens": self.model.input_tokens,
                "nonsequence_tokens": self.model.nonsequence_tokens,
                "numerical_hidden_dim": self.model.numerical_hidden_dim,
                "sequence_hidden_dim": self.model.sequence_hidden_dim,
                "wukong": {
                    "fm": self.model.wukong.fm,
                    "fmb_tokens": self.model.wukong.fmb_tokens,
                    "lcb_tokens": self.model.wukong.lcb_tokens,
                    "mlp_hidden_dims": list(self.model.wukong.mlp_hidden_dims),
                    "activation": self.model.wukong.activation,
                },
                "hstu": {
                    "num_heads": self.model.hstu.num_heads,
                    "attention_dim_per_head": self.model.hstu.attention_dim_per_head,
                    "value_dim_per_head": self.model.hstu.value_dim_per_head,
                    "attention_normalization": self.model.hstu.attention_normalization,
                    "relative_time_buckets": self.model.hstu.relative_time_buckets,
                    "relative_time_log_divisor": self.model.hstu.relative_time_log_divisor,
                    "time_alignment": self.model.hstu.time_alignment,
                    "input_projection_bias": self.model.hstu.input_projection_bias,
                    "output_projection_bias": self.model.hstu.output_projection_bias,
                    "norm_affine": self.model.hstu.norm_affine,
                    "dropout": self.model.hstu.dropout,
                },
                "fusion": {
                    "attention": self.model.fusion.attention,
                    "num_heads": self.model.fusion.num_heads,
                    "projection_bias": self.model.fusion.projection_bias,
                    "swiglu_hidden_dim": self.model.fusion.swiglu_hidden_dim,
                },
                "head": {
                    "hidden_dims": list(self.model.head.hidden_dims),
                    "activation": self.model.head.activation,
                    "dropout": self.model.head.dropout,
                },
            },
            "training": {
                "batch_size": self.training.batch_size,
                "max_epochs": self.training.max_epochs,
                "optimizer": self.training.optimizer,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "loss": self.training.loss,
                "gradient_clip_norm": self.training.gradient_clip_norm,
                "selection_metric": self.training.selection_metric,
                "selection_mode": self.training.selection_mode,
                "tie_break": self.training.tie_break,
                "early_stopping_patience": self.training.early_stopping_patience,
                "early_stopping_min_delta": self.training.early_stopping_min_delta,
                "checkpoint_on_epoch_end": self.training.checkpoint_on_epoch_end,
                "log_every_steps": self.training.log_every_steps,
            },
            "evaluation": {
                "metrics": list(self.evaluation.metrics),
                "probability_dtype": self.evaluation.probability_dtype,
                "single_class_policy": self.evaluation.single_class_policy,
                "final_test_policy": self.evaluation.final_test_policy,
                "duration_groups": [
                    [lower, upper] for lower, upper in self.evaluation.duration_groups
                ],
                "duration_group_unit": self.evaluation.duration_group_unit,
            },
        }
        if self.schema_version == 3:
            result["experiment"]["baseline_indices"] = str(self.experiment.baseline_indices)
            result["data"]["extension_seed"] = list(self.data.extension_seed)
            result["model"]["hstu"].update({
                "ranking_input": self.model.hstu.ranking_input,
                "ranking_time_alignment": self.model.hstu.ranking_time_alignment,
            })
            result["training"].update({
                "sample_order": self.training.sample_order,
                "id_mask_probability": self.training.id_mask_probability,
                "id_mask_fields": list(self.training.id_mask_fields),
                "id_mask_seed_namespace": self.training.id_mask_seed_namespace,
            })
        return result


def _load_data_config(section: dict[str, Any], version: int) -> DataConfig:
    groups = _field(section, "numerical_groups")
    if not isinstance(groups, list) or not all(isinstance(g, list) for g in groups):
        raise TypeError("配置字段 numerical_groups 需要二维列表")
    target_samples = _field(section, "target_samples")
    if not isinstance(target_samples, dict):
        raise TypeError("配置字段 target_samples 需要映射")
    for split in ("train", "validation", "test"):
        if split not in target_samples:
            raise KeyError(f"配置字段 target_samples 缺少 {split}")
    files = _field(section, "files")
    if not isinstance(files, dict):
        raise TypeError("配置字段 files 需要映射")
    return DataConfig(
        name=_expect(section, "name", "RecFlow"),
        raw_root=Path(_field(section, "raw_root")),
        processed_root=Path(_field(section, "processed_root")),
        label=_expect(section, "label", "long_view"),
        timezone=_expect(section, "timezone", "Asia/Shanghai"),
        file_start_date=_field(files, "start_date"),
        file_end_date=_field(files, "end_date"),
        file_extension=_field(files, "extension"),
        expected_file_count=int(_field(files, "expected_count")),
        train_start=_field(section, "train_start"),
        validation_start=_field(section, "validation_start"),
        test_start=_field(section, "test_start"),
        end_exclusive=_field(section, "end_exclusive"),
        target_samples={split: int(value) for split, value in target_samples.items()},
        sampling=_expect(section, "sampling", "uniform_without_replacement" if version == 2 else "extend_baseline_uniform_without_replacement"),
        max_history_length=int(_field(section, "max_history_length")),
        history_source=_expect(section, "history_source", "realshow"),
        history_availability=_expect(
            section, "history_availability", "request_timestamp_plus_playing_time"
        ),
        playing_time_unit=_expect(section, "playing_time_unit", "ms"),
        request_gap_ms=int(_field(section, "request_gap_ms")),
        same_request_timestamp=_expect(
            section, "same_request_timestamp", "exclude"
        ),
        padding_side=_expect(section, "padding_side", "right"),
        vocabulary_fit=_expect(
            section, "vocabulary_fit", "training_inputs_and_their_histories"
        ),
        padding_index=int(_field(section, "padding_index")),
        unknown_index=int(_field(section, "unknown_index")),
        feedback_index_offset=int(_field(section, "feedback_index_offset")),
        categorical_fields=_as_tuple(section, "categorical_fields"),
        history_video_fields=_as_tuple(section, "history_video_fields"),
        history_feedback_fields=_as_tuple(section, "history_feedback_fields"),
        numerical_groups=tuple(tuple(group) for group in groups),
        extension_seed=tuple(int(v) for v in _field(section, "extension_seed")) if version == 3 else (),
    )


def _load_model_config(section: dict[str, Any], version: int) -> ModelConfig:
    wukong = _field(section, "wukong")
    hstu = _field(section, "hstu")
    fusion = _field(section, "fusion")
    head = _field(section, "head")
    return ModelConfig(
        name=_field(section, "name"),
        embedding_dim=int(_field(section, "embedding_dim")),
        layer_norm_eps=float(_field(section, "layer_norm_eps")),
        num_layers=int(_field(section, "num_layers")),
        input_tokens=int(_field(section, "input_tokens")),
        nonsequence_tokens=int(_field(section, "nonsequence_tokens")),
        numerical_hidden_dim=int(_field(section, "numerical_hidden_dim")),
        sequence_hidden_dim=int(_field(section, "sequence_hidden_dim")),
        wukong=WukongConfig(
            fm=_expect(wukong, "fm", "full_dot_product"),
            fmb_tokens=int(_field(wukong, "fmb_tokens")),
            lcb_tokens=int(_field(wukong, "lcb_tokens")),
            mlp_hidden_dims=tuple(int(v) for v in _field(wukong, "mlp_hidden_dims")),
            activation=_expect(wukong, "activation", "silu"),
        ),
        hstu=HSTUConfig(
            num_heads=int(_field(hstu, "num_heads")),
            attention_dim_per_head=int(_field(hstu, "attention_dim_per_head")),
            value_dim_per_head=int(_field(hstu, "value_dim_per_head")),
            attention_normalization=_expect(
                hstu, "attention_normalization", "silu_div_max_history_length"
            ),
            relative_time_buckets=int(_field(hstu, "relative_time_buckets")),
            relative_time_log_divisor=float(_field(hstu, "relative_time_log_divisor")),
            time_alignment=_expect(hstu, "time_alignment", "current_event"),
            input_projection_bias=bool(_field(hstu, "input_projection_bias")),
            output_projection_bias=bool(_field(hstu, "output_projection_bias")),
            norm_affine=bool(_field(hstu, "norm_affine")),
            dropout=float(_field(hstu, "dropout")),
            ranking_input=_expect(hstu, "ranking_input", "interleaved") if version == 3 else "pooled",
            ranking_time_alignment=_expect(hstu, "ranking_time_alignment", "next_event") if version == 3 else "next_event",
        ),
        fusion=FusionConfig(
            attention=_expect(fusion, "attention", "softmax"),
            num_heads=int(_field(fusion, "num_heads")),
            projection_bias=bool(_field(fusion, "projection_bias")),
            swiglu_hidden_dim=int(_field(fusion, "swiglu_hidden_dim")),
        ),
        head=HeadConfig(
            hidden_dims=tuple(int(v) for v in _field(head, "hidden_dims")),
            activation=_expect(head, "activation", "silu"),
            dropout=float(_field(head, "dropout")),
        ),
    )


def _load_training_config(section: dict[str, Any], version: int) -> TrainingConfig:
    return TrainingConfig(
        batch_size=int(_field(section, "batch_size")),
        max_epochs=int(_field(section, "max_epochs")),
        optimizer=_expect(section, "optimizer", "adamw"),
        learning_rate=float(_field(section, "learning_rate")),
        weight_decay=float(_field(section, "weight_decay")),
        loss=_expect(section, "loss", "bce_with_logits"),
        gradient_clip_norm=float(_field(section, "gradient_clip_norm")),
        selection_metric=_expect(section, "selection_metric", "validation_logloss"),
        selection_mode=_expect(section, "selection_mode", "min"),
        tie_break=_expect(section, "tie_break", "earliest_epoch"),
        early_stopping_patience=int(_field(section, "early_stopping_patience")),
        early_stopping_min_delta=float(_field(section, "early_stopping_min_delta")),
        checkpoint_on_epoch_end=bool(_field(section, "checkpoint_on_epoch_end")),
        log_every_steps=int(_field(section, "log_every_steps")),
        sample_order=_expect(section, "sample_order", "daily_shuffle") if version == 3 else "global_shuffle",
        id_mask_probability=float(_expect(section, "id_mask_probability", 0.05)) if version == 3 else 0.0,
        id_mask_fields=_as_tuple(section, "id_mask_fields") if version == 3 else (),
        id_mask_seed_namespace=int(_expect(section, "id_mask_seed_namespace", 97)) if version == 3 else 97,
    )


def _load_evaluation_config(section: dict[str, Any], version: int) -> EvaluationConfig:
    groups = _field(section, "duration_groups")
    if not isinstance(groups, list):
        raise TypeError("配置字段 duration_groups 需要列表")
    parsed: list[tuple[float, float | None]] = []
    for entry in groups:
        if not isinstance(entry, list) or len(entry) != 2:
            raise TypeError("配置字段 duration_groups 的每项需要两个元素")
        lower = float(entry[0])
        upper = None if entry[1] is None else float(entry[1])
        parsed.append((lower, upper))
    return EvaluationConfig(
        metrics=_as_tuple(section, "metrics"),
        probability_dtype=_expect(section, "probability_dtype", "float64"),
        single_class_policy=_expect(section, "single_class_policy", "error"),
        final_test_policy=_expect(
            section, "final_test_policy", "selected_checkpoint_only" if version == 2 else "final_epoch_only"
        ),
        duration_groups=tuple(parsed),
        duration_group_unit=_expect(section, "duration_group_unit", "second"),
    )


def _load_runtime_config(section: dict[str, Any]) -> RuntimeConfig:
    runtime = RuntimeConfig(
        python=_field(section, "python"),
        device=_field(section, "device"),
        dtype=_field(section, "dtype"),
        num_workers=int(_field(section, "num_workers")),
        allow_device_fallback=bool(_field(section, "allow_device_fallback")),
    )
    if runtime.python != "3.13":
        raise ValueError(f"配置要求的 Python 版本为 3.13，实际配置为 {runtime.python}")
    if runtime.device != "cuda":
        raise ValueError(f"正式实验要求 device 为 cuda，实际配置为 {runtime.device}")
    if runtime.dtype != "float32":
        raise ValueError(f"正式实验要求 dtype 为 float32，实际配置为 {runtime.dtype}")
    if runtime.allow_device_fallback:
        raise ValueError("配置禁止设备回退，allow_device_fallback 必须为 false")
    if runtime.num_workers != 0:
        raise ValueError("批次读取在主进程执行，num_workers 必须为 0")
    return runtime


def _load_experiment_config(section: dict[str, Any], version: int) -> ExperimentConfig:
    models = _as_tuple(section, "models")
    expected = ("wukong", "hstu", "shallow_hybrid", "whale", "whale_mean_pool")
    if models != expected:
        raise ValueError(f"配置字段 models 需要 {expected}，实际为 {models}")
    if int(_field(section, "runs_per_model")) != 1:
        raise ValueError("每种模型只执行一次正式训练")
    if int(_field(section, "parallel_runs")) != 1:
        raise ValueError("正式实验按顺序执行，parallel_runs 必须为 1")
    return ExperimentConfig(
        models=models,
        runs_per_model=1,
        parallel_runs=1,
        output_root=Path(_field(section, "output_root")),
        baseline_indices=Path(_field(section, "baseline_indices")) if version == 3 else None,
    )


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在 {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"配置文件 {config_path} 的内容不是映射")
    for key in (
        "schema_version",
        "project",
        "seed",
        "experiment",
        "runtime",
        "data",
        "model",
        "training",
        "evaluation",
    ):
        if key not in raw:
            raise KeyError(f"配置文件缺少顶层字段 {key}")
    version = int(raw["schema_version"])
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"配置版本为 {version}，支持的版本为 {SUPPORTED_SCHEMA_VERSIONS}"
        )

    config = Config(
        path=config_path,
        schema_version=version,
        project=_field(raw, "project"),
        seed=int(_field(raw, "seed")),
        experiment=_load_experiment_config(_field(raw, "experiment"), version),
        runtime=_load_runtime_config(_field(raw, "runtime")),
        data=_load_data_config(_field(raw, "data"), version),
        model=_load_model_config(_field(raw, "model"), version),
        training=_load_training_config(_field(raw, "training"), version),
        evaluation=_load_evaluation_config(_field(raw, "evaluation"), version),
    )
    if config.seed != 42:
        raise ValueError(f"正式实验固定使用种子 42，实际配置为 {config.seed}")
    if config.data.target_samples != {
        "train": 1000000 if version == 2 else 3000000,
        "validation": 100000,
        "test": 200000,
    }:
        raise ValueError(f"目标样本数量与正式设置不一致：{config.data.target_samples}")
    if config.model.embedding_dim != 64 or config.model.num_layers != 2:
        raise ValueError("正式实验固定 Embedding 为 64、层数为 2")
    if config.data.max_history_length != 128:
        raise ValueError("正式实验固定历史长度为 128")
    if config.training.batch_size != 256 or config.training.max_epochs != (3 if version == 2 else 1):
        raise ValueError("Batch Size 或训练轮数与当前配置版本不一致")
    if version == 3:
        if config.data.extension_seed != (42, 3000000):
            raise ValueError("扩充抽样种子需要为 [42,3000000]")
        if config.training.id_mask_fields != ("user_id", "device_id", "video_id", "author_id"):
            raise ValueError("训练 ID 遮盖字段或顺序与方案不一致")
        if config.experiment.output_root != Path("outputs/paper") or config.data.processed_root != Path("data/processed/recflow-paper"):
            raise ValueError("版本 3 需要独立的正式结果与加工目录")
    if config.model.input_tokens != 13 or config.model.nonsequence_tokens != 8:
        raise ValueError("正式实验固定初始 Token 为 13、Wukong 输出 Token 为 8")
    if config.data.playing_time_unit != "ms":
        raise ValueError("历史可用时间按毫秒解释 playing_time")
    if config.data.request_gap_ms != 60000:
        raise ValueError("请求可用时间固定增加 60000 毫秒")
    expected_fields = (
        "user_id", "device_id", "age", "gender", "province", "video_id",
        "author_id", "category_level_one", "category_level_two", "upload_type",
    )
    if config.data.categorical_fields != expected_fields:
        raise ValueError("类别字段及其顺序与 RecFlow 输入设计不一致")
    if config.data.history_video_fields != expected_fields[5:]:
        raise ValueError("历史视频属性及其顺序与 RecFlow 输入设计不一致")
    if config.data.history_feedback_fields != ("effective_view", "long_view"):
        raise ValueError("历史反馈字段及其顺序与 RecFlow 输入设计不一致")
    if config.data.numerical_groups != (
        ("log_video_duration", "video_duration_missing"),
        ("hour_sin", "hour_cos", "weekday_sin", "weekday_cos"),
        ("log_video_age_days", "video_age_missing"),
    ):
        raise ValueError("数值分组及其顺序与模型输入不一致")
    if config.model.hstu.norm_affine or config.model.hstu.dropout != 0.0:
        raise ValueError("HSTU 固定使用无可学习参数的 LayerNorm 和零 dropout")
    if config.model.hstu.relative_time_buckets != 128 or config.model.hstu.relative_time_log_divisor != 0.301:
        raise ValueError("HSTU 时间分桶参数与批次计算不一致")
    if len(config.data.file_names()) != config.data.expected_file_count:
        raise ValueError("文件清单数量与配置不一致")
    return config
