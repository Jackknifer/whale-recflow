import io
import pickle
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import PROJECT_ROOT, Config
from .raw import ExposureTable, RequestIndex

OFFICIAL_LENGTH = 50
OFFICIAL_COLUMNS = 10
CHUNK_SIZE = 2048
# 作者历史数组中的列索引，顺序取自 RecFlow 的原始历史定义。
VIDEO_COLUMNS = {
    "video_id": 0,
    "author_id": 1,
    "category_level_two": 2,
    "category_level_one": 3,
    "upload_type": 4,
}


# 使用请求与视频的组合键，将作者历史连接到原始曝光行。
class ExposureLookup:
    def __init__(self, exposure: ExposureTable) -> None:
        if np.any(exposure.request_id < 0) or np.any(exposure.video_id < 0):
            raise ValueError("曝光的请求与视频编号需要为非负整数")
        self.video_bits = max(1, int(exposure.video_id.max()).bit_length())
        if int(exposure.request_id.max()).bit_length() + self.video_bits > 63:
            raise ValueError("请求与视频组合键超出 int64 范围")
        self.max_request = int(exposure.request_id.max())
        self.max_video = int(exposure.video_id.max())
        # 先校验组合键的整数范围，再通过位移构造无冲突的 int64 键。
        keys = (exposure.request_id << self.video_bits) | exposure.video_id.astype(np.int64)
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        duplicate = np.flatnonzero(keys[1:] == keys[:-1])
        left, right = order[duplicate], order[duplicate + 1]
        for field in (*VIDEO_COLUMNS, "user_id", "request_timestamp", "effective_view"):
            values = getattr(exposure, field)
            if np.any(values[left] != values[right]):
                raise ValueError(f"同一请求和视频对应不同的 {field}，无法唯一恢复作者历史")
        # 重复组合所需字段一致时保留首条曝光，形成唯一的历史查找表。
        starts = np.concatenate(([0], np.flatnonzero(keys[1:] != keys[:-1]) + 1))
        self.keys = keys[starts]
        self.rows = order[starts].astype(np.int32)
        self.duplicate_count = int(duplicate.size)

    def find(self, requests: np.ndarray, videos: np.ndarray) -> np.ndarray:
        if np.any((requests < 0) | (requests > self.max_request)):
            raise ValueError("作者历史请求编号超出曝光范围")
        if np.any((videos < 0) | (videos > self.max_video)):
            raise ValueError("作者历史视频编号超出曝光范围")
        keys = (requests << self.video_bits) | videos
        positions = np.searchsorted(self.keys, keys)
        if np.any(positions >= self.keys.size):
            raise ValueError("作者历史包含无法连接的请求和视频")
        if not np.array_equal(self.keys[positions], keys):
            raise ValueError("作者历史包含无法连接的请求和视频")
        return self.rows[positions]


def verify_official_history_rows(
    exposure: ExposureTable,
    request_index: RequestIndex,
    requests: np.ndarray,
    rows: np.ndarray,
    counts: np.ndarray,
    audit: dict[str, int],
) -> None:
    # 加载加工数据时重新验证来源统计、PAD、用户归属和严格时间条件。
    if requests.dtype != np.int64 or np.any(np.diff(requests) <= 0):
        raise ValueError("作者历史请求索引需要唯一且升序的 int64 编号")
    if counts.dtype != np.int32 or np.any((counts < 0) | (counts > OFFICIAL_LENGTH)):
        raise ValueError("作者历史长度超出 0～50 或类型不符合要求")
    if audit["request_count"] != requests.size or audit["retained_history_occurrences"] != int(counts.sum()):
        raise ValueError("作者历史的请求数量或条目数量与来源记录不一致")
    for begin in range(0, requests.size, CHUNK_SIZE):
        end = min(begin + CHUNK_SIZE, requests.size)
        chunk = rows[begin:end]
        valid = np.arange(OFFICIAL_LENGTH)[None, :] < counts[begin:end, None]
        if np.any(chunk[~valid] != -1) or np.any((chunk[valid] < 0) | (chunk[valid] >= exposure.size)):
            raise ValueError("作者历史索引的有效位置、PAD 或事件范围无效")
        slots = request_index.positions(requests[begin:end])
        if np.any(slots < 0):
            raise ValueError("作者历史请求没有对应的原始请求")
        safe = np.maximum(chunk, 0)
        time_ms = exposure.request_timestamp[safe]
        history_request = exposure.request_id[safe]
        if np.any(valid & (exposure.user_id[safe] != request_index.user_id[slots, None])):
            raise ValueError("作者历史包含其他用户的事件")
        if np.any(valid & (time_ms >= request_index.request_timestamp[slots, None])):
            raise ValueError("作者历史包含时间不早于目标请求的事件")
        if np.any(valid & (history_request == requests[begin:end, None])):
            raise ValueError("作者历史包含目标请求的事件")
        if np.any(valid & (exposure.effective_view[safe] != 1)):
            raise ValueError("作者历史包含无效观看事件")
        event = exposure.event_id[safe]
        earlier = (time_ms[:, 1:] < time_ms[:, :-1]) | (
            (time_ms[:, 1:] == time_ms[:, :-1]) & (
                (history_request[:, 1:] < history_request[:, :-1]) |
                ((history_request[:, 1:] == history_request[:, :-1]) & (event[:, 1:] < event[:, :-1]))
            )
        )
        if np.any(valid[:, 1:] & earlier):
            raise ValueError("作者历史没有按请求时间、请求编号和事件编号升序排列")


def _gather_chunk(
    exposure: ExposureTable,
    lookup: ExposureLookup,
    arrays: np.ndarray,
    target_users: np.ndarray,
    target_times: np.ndarray,
    target_requests: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    expected_shape = (target_requests.size, OFFICIAL_LENGTH, OFFICIAL_COLUMNS)
    if arrays.shape != expected_shape or not np.issubdtype(arrays.dtype, np.integer):
        raise ValueError(f"作者历史数组需要整数类型及形状 {expected_shape}，实际为 {arrays.shape}、{arrays.dtype}")
    if np.any(arrays[:, :, 7] < 0):
        raise ValueError("作者历史请求时间出现负值")
    # 列索引 7 为秒级请求时间，9 为 request_id；先连接曝光恢复毫秒时间。
    valid = arrays[:, :, 7] > 0
    rows = np.full(valid.shape, -1, dtype=np.int32)
    rows[valid] = lookup.find(arrays[:, :, 9][valid], arrays[:, :, 0][valid])
    safe = np.maximum(rows, 0)
    for field, column in VIDEO_COLUMNS.items():
        if not np.array_equal(getattr(exposure, field)[safe][valid], arrays[:, :, column][valid]):
            raise ValueError(f"作者历史的 {field} 与曝光记录不一致")
    history_time = exposure.request_timestamp[safe]
    if not np.array_equal(history_time[valid] // 1000, arrays[:, :, 7][valid]):
        raise ValueError("作者历史秒级时间与曝光毫秒时间不一致")
    if np.any(valid & (exposure.user_id[safe] != target_users[:, None])):
        raise ValueError("作者历史包含其他用户的记录")
    if np.any(valid & (exposure.effective_view[safe] != 1)):
        raise ValueError("作者有效观看历史包含 effective_view 非 1 的记录")
    # 历史成员沿用作者选择，再排除同一请求及时间不早于目标的记录。
    same_request = valid & (arrays[:, :, 9] == target_requests[:, None])
    nonpast = valid & (history_time >= target_times[:, None])
    keep = valid & ~same_request & ~nonpast
    counts = keep.sum(axis=1).astype(np.int32)
    # 恢复毫秒顺序，同时间按请求与事件编号排序，PAD 位于右侧。
    order = np.lexsort((
        exposure.event_id[safe], arrays[:, :, 9],
        np.where(keep, history_time, np.iinfo(np.int64).max),
    ), axis=1)
    selected = np.take_along_axis(np.where(keep, rows, -1), order, axis=1)
    if not np.array_equal(selected >= 0, np.arange(OFFICIAL_LENGTH)[None, :] < counts[:, None]):
        raise ValueError("作者历史的有效位置与长度不一致")
    return selected, counts, {
        "request_count": int(target_requests.size),
        "source_history_occurrences": int(valid.sum()),
        "retained_history_occurrences": int(keep.sum()),
        "same_request_occurrences": int(same_request.sum()),
        "nonpast_time_occurrences": int(nonpast.sum()),
        "removed_history_occurrences": int((valid & ~keep).sum()),
        "changed_request_count": int(np.any(valid & ~keep, axis=1).sum()),
    }


def read_official_histories(
    config: Config,
    exposure: ExposureTable,
    request_index: RequestIndex,
    history_requests: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    path = PROJECT_ROOT / config.data.history_archive
    if not path.is_file():
        raise FileNotFoundError(f"作者历史压缩包不存在：{path}")
    if config.data.max_history_length != OFFICIAL_LENGTH:
        raise ValueError("作者历史需要 50 条上限及原始请求时间")
    started = time.perf_counter()
    lookup = ExposureLookup(exposure)
    # 每个目标请求保存一份最多 50 条历史，空余位置使用 -1。
    histories = {split: np.full((ids.size, OFFICIAL_LENGTH), -1, dtype=np.int32) for split, ids in history_requests.items()}
    counts = {split: np.zeros(ids.size, dtype=np.int32) for split, ids in history_requests.items()}
    positions = {split: request_index.positions(ids) for split, ids in history_requests.items()}
    if any(np.any(slots < 0) for slots in positions.values()):
        raise ValueError("正式目标请求未出现在曝光请求索引中")
    zone_offset = 8 * 3600 * 1000
    days = {
        split: ((request_index.request_timestamp[slots] + zone_offset) // 86400000).astype("datetime64[D]").astype(str)
        for split, slots in positions.items()
    }
    expected_dates = set(config.data.file_dates())
    seen_dates: set[str] = set()
    daily_records: list[dict[str, Any]] = []
    # 仅加载使用者从 RecFlow 作者来源取得的 pickle 文件。
    # 顺序读取压缩包中的日字典，每天处理完成后释放字典内存。
    with tarfile.open(path, mode="r|gz", bufsize=64 * 1024) as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".pkl"):
                continue
            day = Path(member.name).stem
            if day not in expected_dates or day in seen_dates:
                raise ValueError(f"作者历史日期不在来源清单中或出现重复：{day}")
            if not 0 < member.size < 2_000_000_000:
                raise ValueError(f"作者历史文件大小异常：{member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"无法读取作者历史文件：{member.name}")
            day_started = time.perf_counter()
            with io.BufferedReader(stream, buffer_size=8 * 1024 * 1024) as buffered:
                payload = pickle.load(buffered)
            if not isinstance(payload, dict):
                raise TypeError(f"作者历史文件需要请求字典：{member.name}")
            record: dict[str, Any] = {"day": day, "official_request_count": len(payload), "splits": {}}
            for split, ids in history_requests.items():
                selection = np.flatnonzero(days[split] == day)
                if not selection.size:
                    continue
                stats: dict[str, int] = {}
                for begin in range(0, selection.size, CHUNK_SIZE):
                    chosen = selection[begin:begin + CHUNK_SIZE]
                    request_ids = ids[chosen]
                    arrays = np.stack([payload[int(request)] for request in request_ids])
                    slots = positions[split][chosen]
                    selected, lengths, entry = _gather_chunk(
                        exposure, lookup, arrays, request_index.user_id[slots],
                        request_index.request_timestamp[slots], request_ids,
                    )
                    histories[split][chosen] = selected
                    counts[split][chosen] = lengths
                    for key, value in entry.items():
                        stats[key] = stats.get(key, 0) + value
                record["splits"][split] = stats
            record["seconds"] = float(time.perf_counter() - day_started)
            daily_records.append(record)
            seen_dates.add(day)
            del payload
            print(f"作者历史 {day} 已核验，累计 {len(seen_dates)}/{len(expected_dates)} 天", flush=True)
    # 全部日期和正式请求均需得到覆盖，才能将来源记录标记为完整。
    if seen_dates != expected_dates:
        raise ValueError(f"作者历史缺少日期：{sorted(expected_dates - seen_dates)}")
    totals: dict[str, dict[str, int]] = {}
    for record in daily_records:
        for split, entry in record["splits"].items():
            totals.setdefault(split, {})
            for key, value in entry.items():
                totals[split][key] = totals[split].get(key, 0) + value
    for split, ids in history_requests.items():
        if totals.get(split, {}).get("request_count") != ids.size:
            raise ValueError(f"作者历史未覆盖完整的 {split} 正式请求")
    audit = {
        "source": config.data.history_source,
        "archive_name": path.name,
        "archive_bytes": path.stat().st_size,
        "request_timestamp_unit": "seconds_joined_to_raw_milliseconds",
        "raw_pair_count": int(lookup.keys.size),
        "raw_duplicate_pair_rows": lookup.duplicate_count,
        "official_request_count": sum(record["official_request_count"] for record in daily_records),
        "splits": totals,
        "days": daily_records,
        "seconds": float(time.perf_counter() - started),
        "complete": True,
    }
    return histories, counts, audit
