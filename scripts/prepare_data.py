import argparse
import sys
from pathlib import Path

from src.config import load_config
from src.data.prepare import prepare
from src.utils import Stopwatch, format_seconds


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成三模型共用的 RecFlow 加工数据")
    parser.add_argument("--config", default=Path("configs/official.yaml"), type=Path, help="配置文件路径")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    arguments = parse_arguments(argv)
    config = load_config(arguments.config)
    with Stopwatch() as stopwatch:
        metadata = prepare(config)
    root = config.processed_directory()
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
