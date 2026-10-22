from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from ..config import Config

DAY_MS = 86_400_000
SHANGHAI_OFFSET_MS = 8 * 3600 * 1000
INT32_MAX = 2**31 - 1
EVENT_ID_FILE_SHIFT = 32

CATEGORICAL_FIELDS = (
    "user_id",
    "device_id",
    "age",
    "gender",
    "province",
    "video_id",
    "author_id",
    "category_level_one",
    "category_level_two",
    "upload_type",
)
USER_FIELDS = ("user_id", "device_id", "age", "gender", "province")
VIDEO_FIELDS = (
    "video_id",
    "author_id",
    "category_level_one",
    "category_level_two",
    "upload_type",
)
HISTORY_VIDEO_FIELDS = VIDEO_FIELDS
HISTORY_FEEDBACK_FIELDS = ("effective_view", "long_view")
BINARY_FIELDS = ("effective_view", "long_view")

EXPOSURE_INT32 = (
    "user_id",
    "device_id",
    "age",
    "gender",
    "province",
    "video_id",
    "author_id",
    "duration",
    "category_level_one",
    "category_level_two",
    "upload_type",
    "playing_time",
)
EXPOSURE_INT64 = ("request_id", "request_timestamp", "upload_timestamp")
EXPOSURE_INT8 = ("effective_view", "long_view")

@dataclass(frozen=True)
class ExposureTable:
    request_id: np.ndarray
    user_id: np.ndarray
    request_timestamp: np.ndarray
    device_id: np.ndarray
    age: np.ndarray
    gender: np.ndarray
    province: np.ndarray
    video_id: np.ndarray
    author_id: np.ndarray
    duration: np.ndarray
    category_level_one: np.ndarray
    category_level_two: np.ndarray
    upload_type: np.ndarray
    upload_timestamp: np.ndarray
    playing_time: np.ndarray
    effective_view: np.ndarray
    long_view: np.ndarray
    event_id: np.ndarray
    file_index: np.ndarray
    record_index: np.ndarray

    @property
    def size(self) -> int:
        return int(self.event_id.size)

    def categorical(self, name: str) -> np.ndarray:
        if name not in CATEGORICAL_FIELDS:
            raise KeyError(f"未知类别字段 {name}")
        return getattr(self, name)

    def video_attribute(self, name: str) -> np.ndarray:
        if name not in VIDEO_FIELDS:
            raise KeyError(f"未知视频属性 {name}")
        return getattr(self, name)

    def feedback(self, name: str) -> np.ndarray:
        if name not in HISTORY_FEEDBACK_FIELDS:
            raise KeyError(f"未知反馈字段 {name}")
        return getattr(self, name)


@dataclass(frozen=True)
class SourceFile:
    index: int
    name: str
    date: str
    row_count: int


@dataclass(frozen=True)
class RequestIndex:
    request_ids: np.ndarray
    user_id: np.ndarray
    request_timestamp: np.ndarray
    available_ms: np.ndarray
    event_counts: np.ndarray

    @property
    def size(self) -> int:
        return int(self.request_ids.size)

    def positions(self, request_ids: np.ndarray) -> np.ndarray:
        position = np.searchsorted(self.request_ids, request_ids)
        clipped = np.clip(position, 0, self.request_ids.size - 1)
        found = self.request_ids[clipped] == request_ids
        return np.where(found, clipped, -1).astype(np.int64)


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"原始数据文件不存在 {path}")
    return path


def _day_index_to_calendar(day_index: np.ndarray) -> np.ndarray:
    days = day_index.astype("datetime64[D]")
    years = days.astype("datetime64[Y]").astype(np.int64) + 1970
    months = days.astype("datetime64[M]").astype(np.int64) % 12 + 1
    day_of_month = (days - days.astype("datetime64[M]")).astype(np.int64) + 1
    return years * 10000 + months * 100 + day_of_month


def _read_table(path: Path) -> tuple[list[str], pa.Table]:
    table = feather.read_table(path, memory_map=True)
    original = [field.name for field in table.schema]
    stripped = [name.strip() for name in original]
    if len(set(stripped)) != len(stripped):
        raise ValueError(f"文件 {path} 清除两端空白后的列名出现重名：{stripped}")
    return original, table.rename_columns(stripped)


def _integer_column(table: pa.Table, path: Path, name: str) -> np.ndarray:
    if name not in table.column_names:
        raise KeyError(f"文件 {path} 缺少字段 {name}")
    column = table.column(name)
    if column.null_count:
        raise ValueError(f"文件 {path} 的字段 {name} 含有空值")
    if not pa.types.is_integer(column.type):
        raise TypeError(f"文件 {path} 的字段 {name} 类型为 {column.type}，需要整数")
    return column.to_numpy()


def source_row_counts(config: Config) -> list[SourceFile]:
    sources: list[SourceFile] = []
    for file_index, file_name in enumerate(config.data.file_names()):
        path = _require_file(config.raw_directory() / file_name)
        _, table = _read_table(path)
        sources.append(
            SourceFile(
                index=file_index,
                name=file_name,
                date=file_name[: -len(config.data.file_extension)],
                row_count=table.num_rows,
            )
        )
        del table
    if len(sources) != config.data.expected_file_count:
        raise ValueError(
            f"实际文件数量 {len(sources)} 与配置 {config.data.expected_file_count} 不一致"
        )
    return sources


def _validate_file(path: Path, expected_date: str, table: pa.Table) -> None:
    timestamp = _integer_column(table, path, "request_timestamp")
    if (timestamp <= 0).any():
        raise ValueError(f"文件 {path} 的 request_timestamp 出现非正值")
    local_day = (timestamp + SHANGHAI_OFFSET_MS) // DAY_MS
    unique_days = np.unique(local_day)
    calendar = _day_index_to_calendar(unique_days)
    expected = int(expected_date.replace("-", ""))
    if unique_days.size != 1 or int(calendar[0]) != expected:
        raise ValueError(
            f"文件 {path} 的请求日期为 {[str(value) for value in calendar]}，"
            f"与文件日期 {expected_date} 不一致"
        )
    realshow = _integer_column(table, path, "realshow")
    if not np.all(realshow == 1):
        raise ValueError(f"文件 {path} 的 realshow 字段存在非 1 取值")
    for name in BINARY_FIELDS:
        values = _integer_column(table, path, name)
        if not np.isin(np.unique(values), (0, 1)).all():
            raise ValueError(f"文件 {path} 的字段 {name} 出现非二值取值")
    playing = _integer_column(table, path, "playing_time")
    if (playing < 0).any():
        raise ValueError(f"文件 {path} 的 playing_time 出现负值")
    for name in EXPOSURE_INT32:
        values = _integer_column(table, path, name)
        if values.min() < 0:
            raise ValueError(f"文件 {path} 的字段 {name} 出现负值")
        if values.max() > INT32_MAX:
            raise ValueError(f"文件 {path} 的字段 {name} 超出 int32 表示范围")
    for name in EXPOSURE_INT64:
        values = _integer_column(table, path, name)
        if values.min() < 0:
            raise ValueError(f"文件 {path} 的字段 {name} 出现负值")


def read_exposure(config: Config) -> tuple[ExposureTable, list[SourceFile]]:
    sources = source_row_counts(config)
    total = sum(source.row_count for source in sources)
    if total <= 0:
        raise ValueError("原始数据没有记录")
    if total > INT32_MAX:
        raise ValueError("曝光数量超过历史行索引的 int32 范围")
    buffer: dict[str, np.ndarray] = {}
    for name in EXPOSURE_INT32:
        buffer[name] = np.zeros(total, dtype=np.int32)
    for name in EXPOSURE_INT64:
        buffer[name] = np.zeros(total, dtype=np.int64)
    for name in EXPOSURE_INT8:
        buffer[name] = np.zeros(total, dtype=np.int8)
    event_id = np.zeros(total, dtype=np.int64)
    file_index_array = np.zeros(total, dtype=np.int16)
    record_index = np.zeros(total, dtype=np.int32)

    cursor = 0
    for source in sources:
        path = _require_file(config.raw_directory() / source.name)
        _, table = _read_table(path)
        _validate_file(path, source.date, table)
        if table.num_rows != source.row_count:
            raise ValueError(
                f"文件 {path} 的记录数由 {source.row_count} 变为 {table.num_rows}"
            )
        span = slice(cursor, cursor + table.num_rows)
        for name in EXPOSURE_INT32:
            buffer[name][span] = table.column(name).to_numpy()
        for name in EXPOSURE_INT64:
            buffer[name][span] = table.column(name).to_numpy()
        for name in EXPOSURE_INT8:
            buffer[name][span] = table.column(name).to_numpy()
        rows = np.arange(1, table.num_rows + 1, dtype=np.int64)
        record_index[span] = rows
        file_index_array[span] = source.index
        event_id[span] = (source.index << EVENT_ID_FILE_SHIFT) + rows
        cursor += table.num_rows
        del table
    if cursor != total:
        raise ValueError(f"读取记录总数为 {cursor}，预分配数量为 {total}")
    if np.unique(event_id).size != event_id.size:
        raise ValueError("全局 event_id 出现重复")
    return (
        ExposureTable(
            request_id=buffer["request_id"],
            user_id=buffer["user_id"],
            request_timestamp=buffer["request_timestamp"],
            device_id=buffer["device_id"],
            age=buffer["age"],
            gender=buffer["gender"],
            province=buffer["province"],
            video_id=buffer["video_id"],
            author_id=buffer["author_id"],
            duration=buffer["duration"],
            category_level_one=buffer["category_level_one"],
            category_level_two=buffer["category_level_two"],
            upload_type=buffer["upload_type"],
            upload_timestamp=buffer["upload_timestamp"],
            playing_time=buffer["playing_time"],
            effective_view=buffer["effective_view"],
            long_view=buffer["long_view"],
            event_id=event_id,
            file_index=file_index_array,
            record_index=record_index,
        ),
        sources,
    )


def build_request_index(exposure: ExposureTable, request_gap_ms: int) -> RequestIndex:
    order = np.argsort(exposure.request_id, kind="stable")
    sorted_request = exposure.request_id[order]
    starts_mask = np.zeros(sorted_request.size, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_request[1:] != sorted_request[:-1]
    starts = np.nonzero(starts_mask)[0].astype(np.int64)
    counts = np.diff(np.append(starts, sorted_request.size)).astype(np.int64)
    unique_request = sorted_request[starts]
    sorted_time = exposure.request_timestamp[order]
    sorted_user = exposure.user_id[order]
    group = np.repeat(np.arange(unique_request.size, dtype=np.int64), counts)
    if np.any(sorted_user != sorted_user[starts][group]):
        raise ValueError("同一 request_id 对应多个用户")
    if np.any(sorted_time != sorted_time[starts][group]):
        raise ValueError("同一 request_id 对应多个请求时间")
    playing_sum = np.add.reduceat(exposure.playing_time[order].astype(np.int64), starts)
    unique_time = sorted_time[starts]
    available = unique_time + playing_sum + request_gap_ms
    if np.any(available < unique_time):
        raise ValueError("请求可用时间计算出现整数溢出")
    return RequestIndex(
        request_ids=unique_request.astype(np.int64),
        user_id=sorted_user[starts].astype(np.int32),
        request_timestamp=unique_time.astype(np.int64),
        available_ms=available.astype(np.int64),
        event_counts=counts.astype(np.int32),
    )
