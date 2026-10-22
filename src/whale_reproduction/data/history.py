from dataclasses import dataclass

import numpy as np

from .raw import ExposureTable, RequestIndex

HISTORY_PAD = -1
CHUNK_SIZE = 1024
INITIAL_WINDOW = 256


@dataclass(frozen=True)
class HistoryIndex:
    user_offsets: np.ndarray
    user_ids: np.ndarray
    keys: np.ndarray
    sorted_time: np.ndarray
    sorted_available: np.ndarray
    sorted_request: np.ndarray
    sorted_event: np.ndarray
    time_base_ms: int
    user_shift: int

    @property
    def num_users(self) -> int:
        return int(self.user_ids.size)

    @property
    def num_events(self) -> int:
        return int(self.sorted_event.size)

    def target_key(self, users: np.ndarray, time_ms: np.ndarray) -> np.ndarray:
        return users.astype(np.int64) * (1 << self.user_shift) + (
            time_ms - self.time_base_ms
        )


def build_history_index(
    exposure: ExposureTable, request_index: RequestIndex
) -> HistoryIndex:
    slot = request_index.positions(exposure.request_id)
    if np.any(slot < 0):
        raise ValueError("曝光记录的 request_id 未出现在请求索引中")
    user_values, compact = np.unique(
        exposure.user_id.astype(np.int64), return_inverse=True
    )
    order = np.lexsort(
        (
            exposure.event_id,
            exposure.request_id.astype(np.int64),
            exposure.request_timestamp,
            compact,
        )
    )
    counts = np.bincount(compact, minlength=user_values.size)
    user_offsets = np.zeros(user_values.size + 1, dtype=np.int64)
    np.cumsum(counts, out=user_offsets[1:])
    sorted_user = compact[order].astype(np.int64)
    sorted_time = exposure.request_timestamp[order]
    time_base = int(sorted_time.min())
    span = int(sorted_time.max()) - time_base + 1
    shift = max(1, int(np.ceil(np.log2(max(span, 2)))))
    if shift + int(np.ceil(np.log2(max(user_values.size, 2)))) > 62:
        raise ValueError("用户编号与请求时间的组合键超出 int64 表示范围")
    keys = sorted_user * (1 << shift) + (sorted_time - time_base)
    if np.any(keys[1:] < keys[:-1]):
        raise ValueError("历史索引的组合键没有按升序排列")
    return HistoryIndex(
        user_offsets=user_offsets,
        user_ids=user_values.astype(np.int64),
        keys=keys,
        sorted_time=sorted_time.astype(np.int64),
        sorted_available=request_index.available_ms[slot[order]].astype(np.int64),
        sorted_request=exposure.request_id[order].astype(np.int64),
        sorted_event=order.astype(np.int32),
        time_base_ms=time_base,
        user_shift=shift,
    )


class HistoryReader:
    def __init__(self, index: HistoryIndex, max_length: int) -> None:
        self._index = index
        self._max_length = max_length

    @property
    def max_length(self) -> int:
        return self._max_length

    def gather(
        self,
        users: np.ndarray,
        target_times: np.ndarray,
        target_requests: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        sizes = (users.size, target_times.size, target_requests.size)
        if len(set(sizes)) != 1:
            raise ValueError(f"用户、时间与请求的数量不一致：{sizes}")
        history = np.full((users.size, self._max_length), HISTORY_PAD, dtype=np.int64)
        counts = np.zeros(users.size, dtype=np.int32)
        for start in range(0, users.size, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, users.size)
            chunk_history, chunk_counts = self._gather_chunk(
                users[start:end],
                target_times[start:end],
                target_requests[start:end],
            )
            history[start:end] = chunk_history
            counts[start:end] = chunk_counts
        return history, counts

    def _gather_chunk(
        self,
        users: np.ndarray,
        target_times: np.ndarray,
        target_requests: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        index = self._index
        compact = np.searchsorted(index.user_ids, users.astype(np.int64))
        clipped = np.clip(compact, 0, index.user_ids.size - 1)
        if np.any(index.user_ids[clipped] != users.astype(np.int64)):
            raise ValueError("目标用户不在历史索引的用户集合中")
        bases = index.user_offsets[clipped]
        ends = index.user_offsets[clipped + 1]
        query = index.target_key(compact, target_times)
        cutoffs = np.clip(
            np.searchsorted(index.keys, query, side="left"), bases, ends
        )

        size = users.size
        output = np.full((size, self._max_length), HISTORY_PAD, dtype=np.int64)
        counts = np.zeros(size, dtype=np.int32)
        remaining = np.ones(size, dtype=bool)
        window = INITIAL_WINDOW
        while remaining.any():
            active = np.nonzero(remaining)[0]
            lower = np.maximum(bases[active], cutoffs[active] - window)
            span = cutoffs[active] - lower
            widest = int(span.max()) if span.size else 0
            if widest <= 0:
                remaining[active] = False
                continue
            positions = lower[:, None] + np.arange(widest, dtype=np.int64)[None, :]
            inside = positions < cutoffs[active][:, None]
            bounded = np.minimum(positions, index.sorted_event.size - 1)
            legal = inside & (
                index.sorted_available[bounded] < target_times[active][:, None]
            )
            legal &= index.sorted_request[bounded] != target_requests[active][:, None]
            reverse = np.cumsum(legal[:, ::-1], axis=1)[:, ::-1]
            total = reverse[:, 0]
            kept = np.minimum(total, self._max_length)
            destination = kept[:, None] - reverse
            selected = legal & (destination >= 0)
            rows, columns = np.nonzero(selected)
            if rows.size:
                output[active[rows], destination[rows, columns]] = index.sorted_event[
                    bounded[rows, columns]
                ]
            counts[active] = kept.astype(np.int32)
            finished = (total >= self._max_length) | (lower <= bases[active])
            remaining[active[finished]] = False
            window *= 2
        return output, counts
