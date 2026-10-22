import argparse
import sys
from pathlib import Path

from whale_reproduction.config import load_config
from whale_reproduction.training.reporting import build_comparison


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从已保存指标导出整体比较表、时长分组表与模型差值"
    )
    parser.add_argument("--config", required=True, type=Path, help="配置文件路径")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    arguments = parse_arguments(argv)
    config = load_config(arguments.config)
    frame, summary = build_comparison(config)
    print("整体指标")
    print(frame.to_string(index=False))
    print()
    print("WHALE 相对参照模型的差值")
    for name, values in summary["deltas"].items():
        parts = [f"{name}:"]
        if values["auc"] is not None:
            parts.append(f"AUC 差值 {values['auc']:+.6f}")
        if values["logloss"] is not None:
            parts.append(f"LogLoss 差值 {values['logloss']:+.6f}")
        if values["ne"] is not None:
            parts.append(f"NE 差值 {values['ne']:+.6f}")
        if values["ne_relative_improvement"] is not None:
            parts.append(f"NE 相对改善 {values['ne_relative_improvement']:+.4%}")
        print(" ".join(parts))
    print(f"结果保存至 {config.results_directory()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
