import argparse
import sys
from pathlib import Path

from whale_reproduction.config import load_config
from whale_reproduction.data.prepare import prepare
from whale_reproduction.utils import Stopwatch, format_seconds, write_json


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成五模型共用的 RecFlow 加工数据")
    parser.add_argument("--config", required=True, type=Path, help="配置文件路径")
    parser.add_argument(
        "--baseline-config", type=Path,
        help="显式从版本 2 配置重建基线成员，用于首次下载仓库后的版本 3 加工",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    arguments = parse_arguments(argv)
    config = load_config(arguments.config)
    baseline_config = load_config(arguments.baseline_config) if arguments.baseline_config else None
    with Stopwatch() as stopwatch:
        metadata = prepare(config, baseline_config=baseline_config)
    root = config.processed_directory()
    write_json(
        root / "prepare_summary.json",
        {
            "elapsed_seconds": stopwatch.elapsed,
            "processed_root": str(root),
            "event_count": metadata["event_count"],
            "request_count": metadata["request_count"],
            "source_file_count": metadata["source_file_count"],
            "processed_bytes": metadata["processed_bytes"],
            "splits": {
                split: {
                    "target_count": entry["target_count"],
                    "request_count": entry["request_count"],
                    "positive_rate": entry["positive_rate"],
                    "history_length_mean": entry["history_length_mean"],
                    "empty_history_count": entry["empty_history_count"],
                    "full_history_count": entry["full_history_count"],
                }
                for split, entry in metadata["splits"].items()
            },
            "vocabulary_sizes": metadata["vocabulary_sizes"],
        },
    )
    print(f"加工数据保存至 {root}")
    print(
        f"曝光记录 {metadata['event_count']:,} 条，"
        f"请求 {metadata['request_count']:,} 个，"
        f"来源文件 {metadata['source_file_count']} 个"
    )
    print(
        f"用户 {metadata['distinct_users']:,} 个，"
        f"视频 {metadata['distinct_videos']:,} 个，"
        f"加工文件 {metadata['processed_bytes'] / 1024**3:.2f} GiB"
    )
    for split, entry in metadata["splits"].items():
        print(
            f"{split}: 候选 {entry['candidate_count']:,} 条，"
            f"目标 {entry['target_count']:,} 条，"
            f"请求 {entry['request_count']:,} 个，"
            f"正例比例 {entry['positive_rate']:.6f}，"
            f"历史长度均值 {entry['history_length_mean']:.2f}，"
            f"空历史 {entry['empty_history_count']:,} 条，"
            f"满历史 {entry['full_history_count']:,} 条"
        )
    print(f"数据准备耗时 {format_seconds(stopwatch.elapsed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
