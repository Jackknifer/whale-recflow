from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

# 所有项目路径以源码所在仓库为基准，保持从不同目录启动时的路径一致。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _field(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"配置缺少字段 {key}")
    return mapping[key]


def _expect(mapping: dict[str, Any], key: str, expected: Any) -> Any:
    # 固定实验约定在加载配置时检查，错误直接报告到具体字段。
    value = _field(mapping, key)
    if value != expected:
        raise ValueError(f"配置字段 {key} 的值为 {value!r}，当前实现要求 {expected!r}")
    return value


def _as_tuple(mapping: dict[str, Any], key: str) -> tuple[str, ...]:
    value = _field(mapping, key)
    if not isinstance(value, list):
        raise TypeError(f"配置字段 {key} 需要列表")
    return tuple(value)


def _snapshot(value: Any) -> Any:
    # 将 Path 和 tuple 转换为 JSON 可保存的基础类型。
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_snapshot(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"配置包含不支持的类型 {type(value)}")


def parse_time_ms(text: str, timezone_name: str) -> int:
    # 配置时间必须包含显式时区，统一转换为 UTC 毫秒时间戳。
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        raise ValueError(f"时间字符串 {text} 缺少时区")
    if moment.utcoffset() != moment.astimezone(ZoneInfo(timezone_name)).utcoffset():
        raise ValueError(f"时间字符串 {text} 的时区信息与配置时区 {timezone_name} 不一致")
    return int(round(moment.timestamp() * 1000))


@dataclass(frozen=True)
class DataConfig:
    # 各特征列表的顺序决定预处理结果和模型输入的排列。
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
    history_archive: Path
    history_availability: str
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
    extension_seed: tuple[int, ...]

    def time_bounds_ms(self) -> dict[str, tuple[int, int]]:
        # 每组时间范围均采用 [start, end)，相邻集合共享边界。
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
        # 原始日文件包含起止日期，数量必须与配置一致。
        start = datetime.strptime(self.file_start_date, "%Y-%m-%d").date()
        end = datetime.strptime(self.file_end_date, "%Y-%m-%d").date()
        if end < start:
            raise ValueError("原始文件日期范围无效")
        dates = tuple((start + timedelta(days=index)).isoformat()
                      for index in range((end - start).days + 1))
        if len(dates) != self.expected_file_count:
            raise ValueError("原始文件日期数量与配置不一致")
        return dates

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
    # 历史编码和交替排序输入分别声明时间计算方式，共用时间分桶设置。
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
    ranking_input: str
    ranking_time_alignment: str


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
    # 三个模型共用输入维度；各计算模块使用自己的参数设置。
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
    # 同时记录优化过程、验证选权、样本顺序和训练 ID 遮盖规则。
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
    sample_order: str
    id_mask_probability: float
    id_mask_fields: tuple[str, ...]
    id_mask_seed_namespace: int


@dataclass(frozen=True)
class EvaluationConfig:
    # 测试策略与训练轮数共同决定最终使用的权重。
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


@dataclass(frozen=True)
class Config:
    # 聚合各部分配置，并集中生成运行、数据和结果路径。
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

    def run_directory(self, model_name: str) -> Path:
        return PROJECT_ROOT / self.experiment.output_root / "runs" / f"{model_name}-s{self.seed}"

    def results_directory(self) -> Path:
        return PROJECT_ROOT / self.experiment.output_root / "results"

    def checks_directory(self) -> Path:
        return PROJECT_ROOT / self.experiment.output_root / "checks"

    def processed_directory(self) -> Path:
        return PROJECT_ROOT / self.data.processed_root

    def raw_directory(self) -> Path:
        return PROJECT_ROOT / self.data.raw_root

    def preprocessing_snapshot(self) -> dict[str, Any]:
        # 排除可迁移的数据目录，仅记录格式版本、数据规则和种子。
        data = self.snapshot()["data"]
        del data["raw_root"]
        del data["processed_root"]
        return {"schema_version": self.schema_version, "seed": self.seed, "data": data}

    def snapshot(self) -> dict[str, Any]:
        # 保存完整实验设置，同时排除本机配置文件路径。
        result = asdict(self)
        del result["path"]
        result["data"]["file_names"] = self.data.file_names()
        return _snapshot(result)


def _load_data_config(section: dict[str, Any]) -> DataConfig:
    # 检查完整字段集合，防止多余字段或拼写错误被忽略。
    files = _field(section, "files")
    groups = _field(section, "numerical_groups")
    if not isinstance(groups, list) or not all(isinstance(group, list) for group in groups):
        raise TypeError("配置字段 numerical_groups 需要二维列表")
    expected = {field.name for field in fields(DataConfig)}
    expected.difference_update(("file_start_date", "file_end_date", "file_extension", "expected_file_count"))
    expected.add("files")
    if set(section) != expected:
        raise ValueError(f"数据配置字段不匹配：{set(section) ^ expected}")
    return DataConfig(
        name=_expect(section, "name", "RecFlow"),
        raw_root=Path(_field(section, "raw_root")),
        processed_root=Path(_field(section, "processed_root")),
        label=_expect(section, "label", "long_view"),
        timezone=_expect(section, "timezone", "Asia/Shanghai"),
        file_start_date=_field(files, "start_date"),
        file_end_date=_field(files, "end_date"),
        file_extension=_expect(files, "extension", ".feather"),
        expected_file_count=int(_field(files, "expected_count")),
        train_start=_field(section, "train_start"),
        validation_start=_field(section, "validation_start"),
        test_start=_field(section, "test_start"),
        end_exclusive=_field(section, "end_exclusive"),
        target_samples={name: int(value) for name, value in _field(section, "target_samples").items()},
        sampling=_expect(section, "sampling", "two_stage_uniform_without_replacement"),
        max_history_length=int(_field(section, "max_history_length")),
        history_source=_expect(section, "history_source", "seq_effective_50_dict"),
        history_archive=Path(_field(section, "history_archive")),
        history_availability=_expect(section, "history_availability", "official_history_strict_request_time"),
        same_request_timestamp=_expect(section, "same_request_timestamp", "exclude"),
        padding_side=_expect(section, "padding_side", "right"),
        vocabulary_fit=_expect(section, "vocabulary_fit", "training_inputs_and_their_histories"),
        padding_index=int(_expect(section, "padding_index", 0)),
        unknown_index=int(_expect(section, "unknown_index", 1)),
        feedback_index_offset=int(_expect(section, "feedback_index_offset", 2)),
        categorical_fields=_as_tuple(section, "categorical_fields"),
        history_video_fields=_as_tuple(section, "history_video_fields"),
        history_feedback_fields=_as_tuple(section, "history_feedback_fields"),
        numerical_groups=tuple(tuple(group) for group in groups),
        extension_seed=tuple(int(value) for value in _field(section, "extension_seed")),
    )


def _load_model_config(section: dict[str, Any]) -> ModelConfig:
    wukong, hstu, fusion, head = (_field(section, name) for name in ("wukong", "hstu", "fusion", "head"))
    return ModelConfig(
        name=_expect(section, "name", "whale"),
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
            mlp_hidden_dims=tuple(int(value) for value in _field(wukong, "mlp_hidden_dims")),
            activation=_expect(wukong, "activation", "silu"),
        ),
        hstu=HSTUConfig(
            num_heads=int(_field(hstu, "num_heads")),
            attention_dim_per_head=int(_field(hstu, "attention_dim_per_head")),
            value_dim_per_head=int(_field(hstu, "value_dim_per_head")),
            attention_normalization=_expect(hstu, "attention_normalization", "silu_div_max_history_length"),
            relative_time_buckets=int(_field(hstu, "relative_time_buckets")),
            relative_time_log_divisor=float(_field(hstu, "relative_time_log_divisor")),
            time_alignment=_expect(hstu, "time_alignment", "current_event"),
            input_projection_bias=bool(_expect(hstu, "input_projection_bias", False)),
            output_projection_bias=bool(_expect(hstu, "output_projection_bias", True)),
            norm_affine=bool(_expect(hstu, "norm_affine", False)),
            dropout=float(_expect(hstu, "dropout", 0.0)),
            ranking_input=_expect(hstu, "ranking_input", "interleaved"),
            ranking_time_alignment=_expect(hstu, "ranking_time_alignment", "next_event"),
        ),
        fusion=FusionConfig(
            attention=_expect(fusion, "attention", "softmax"),
            num_heads=int(_expect(fusion, "num_heads", 1)),
            projection_bias=bool(_expect(fusion, "projection_bias", False)),
            swiglu_hidden_dim=int(_field(fusion, "swiglu_hidden_dim")),
        ),
        head=HeadConfig(
            hidden_dims=tuple(int(value) for value in _field(head, "hidden_dims")),
            activation=_expect(head, "activation", "silu"),
            dropout=float(_field(head, "dropout")),
        ),
    )


def load_config(path: str | Path = "configs/official.yaml") -> Config:
    # 加载配置后统一验证实验规模、模型输入顺序和评估规则。
    config_path = PROJECT_ROOT / path
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError("配置内容需要映射")
    experiment = _field(raw, "experiment")
    training = _field(raw, "training")
    evaluation = _field(raw, "evaluation")
    runtime = _field(raw, "runtime")
    if set(experiment) != {"models", "runs_per_model", "parallel_runs", "output_root"}:
        raise ValueError("实验配置字段不匹配")
    config = Config(
        path=config_path,
        schema_version=int(_expect(raw, "schema_version", 4)),
        project=_field(raw, "project"),
        seed=int(_expect(raw, "seed", 42)),
        experiment=ExperimentConfig(
            models=_as_tuple(experiment, "models"),
            runs_per_model=int(_expect(experiment, "runs_per_model", 1)),
            parallel_runs=int(_expect(experiment, "parallel_runs", 1)),
            output_root=Path(_field(experiment, "output_root")),
        ),
        runtime=RuntimeConfig(
            python=_expect(runtime, "python", "3.13"),
            device=_expect(runtime, "device", "cuda"),
            dtype=_expect(runtime, "dtype", "float32"),
            num_workers=int(_expect(runtime, "num_workers", 0)),
            allow_device_fallback=bool(_expect(runtime, "allow_device_fallback", False)),
        ),
        data=_load_data_config(_field(raw, "data")),
        model=_load_model_config(_field(raw, "model")),
        training=TrainingConfig(
            batch_size=int(_field(training, "batch_size")),
            max_epochs=int(_field(training, "max_epochs")),
            optimizer=_expect(training, "optimizer", "adamw"),
            learning_rate=float(_field(training, "learning_rate")),
            weight_decay=float(_field(training, "weight_decay")),
            loss=_expect(training, "loss", "bce_with_logits"),
            gradient_clip_norm=float(_field(training, "gradient_clip_norm")),
            selection_metric=_expect(training, "selection_metric", "validation_logloss"),
            selection_mode=_expect(training, "selection_mode", "min"),
            tie_break=_expect(training, "tie_break", "earliest_epoch"),
            early_stopping_patience=int(_field(training, "early_stopping_patience")),
            early_stopping_min_delta=float(_field(training, "early_stopping_min_delta")),
            checkpoint_on_epoch_end=bool(_expect(training, "checkpoint_on_epoch_end", True)),
            log_every_steps=int(_field(training, "log_every_steps")),
            sample_order=_expect(training, "sample_order", "daily_shuffle"),
            id_mask_probability=float(_expect(training, "id_mask_probability", 0.05)),
            id_mask_fields=_as_tuple(training, "id_mask_fields"),
            id_mask_seed_namespace=int(_expect(training, "id_mask_seed_namespace", 97)),
        ),
        evaluation=EvaluationConfig(
            metrics=_as_tuple(evaluation, "metrics"),
            probability_dtype=_expect(evaluation, "probability_dtype", "float64"),
            single_class_policy=_expect(evaluation, "single_class_policy", "error"),
            final_test_policy=_field(evaluation, "final_test_policy"),
            duration_groups=tuple((float(lower), None if upper is None else float(upper))
                                  for lower, upper in _field(evaluation, "duration_groups")),
            duration_group_unit=_expect(evaluation, "duration_group_unit", "second"),
        ),
    )
    if config.model_names != ("wukong", "hstu", "whale"):
        raise ValueError("比较模型需要为 wukong、hstu、whale")
    if config.data.target_samples != {"train": 3000000, "validation": 100000, "test": 200000}:
        raise ValueError("目标样本数量与正式设置不一致")
    if (config.model.embedding_dim, config.model.num_layers) != (64, 2):
        raise ValueError("正式实验使用 64 维 Embedding 和两层模型")
    if config.data.max_history_length != 50 or config.data.extension_seed != (42, 3000000):
        raise ValueError("历史上限或抽样种子与正式设置不一致")
    if config.training.batch_size != 256 or config.training.max_epochs not in (1, 3):
        raise ValueError("正式实验的 Batch Size 为 256，训练轮数为 1 或 3")
    # 单轮训练直接评估该轮权重，多轮训练评估验证集选出的权重。
    policy = "final_epoch_only" if config.training.max_epochs == 1 else "selected_checkpoint_only"
    if config.evaluation.final_test_policy != policy:
        raise ValueError("最终测试策略与训练轮数不一致")
    fields = ("user_id", "device_id", "age", "gender", "province", "video_id", "author_id",
              "category_level_one", "category_level_two", "upload_type")
    if config.data.categorical_fields != fields or config.data.history_video_fields != fields[5:]:
        raise ValueError("类别字段或顺序与模型输入不一致")
    if config.data.history_feedback_fields != ("effective_view",):
        raise ValueError("历史反馈需要为 effective_view")
    if config.training.id_mask_fields != ("user_id", "device_id", "video_id", "author_id"):
        raise ValueError("训练 ID 遮盖字段与模型输入不一致")
    if config.data.numerical_groups != (
        ("log_video_duration", "video_duration_missing"),
        ("hour_sin", "hour_cos", "weekday_sin", "weekday_cos"),
        ("log_video_age_days", "video_age_missing"),
    ):
        raise ValueError("数值分组与模型输入不一致")
    if (config.model.input_tokens, config.model.nonsequence_tokens) != (13, 8):
        raise ValueError("输入 Token 数量需要为 13，Wukong 输出需要为 8")
    if config.model.hstu.relative_time_buckets != 128 or config.model.hstu.relative_time_log_divisor != 0.301:
        raise ValueError("HSTU 时间分桶参数与批次计算不一致")
    config.data.file_names()
    return config
