import argparse
import sys
from pathlib import Path

from whale_reproduction.config import load_config
from whale_reproduction.data.batch import BatchReader
from whale_reproduction.data.store import load_processed
from whale_reproduction.training.checks import run_checks
from whale_reproduction.training.experiment import run_all_models
from whale_reproduction.utils import Stopwatch, format_seconds, resolve_device

PHASES = ("check", "train")


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RecFlow 上的必要检查与三模型正式训练")
    parser.add_argument("--config", default=Path("configs/official.yaml"), type=Path, help="配置文件路径")
    parser.add_argument(
        "--phase",
        required=True,
        choices=PHASES,
        help="check 执行必要检查，train 执行三模型正式训练",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="按已完成的 Epoch 恢复训练，只在明确需要恢复时使用",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    arguments = parse_arguments(argv)
    config = load_config(arguments.config)
    if arguments.phase == "train" and arguments.resume:
        print("按完成的 Epoch 恢复训练，已完成的模型保持原有结果")
    device = resolve_device(config)
    processed = load_processed(config)
    reader = BatchReader(processed)
    with Stopwatch() as stopwatch:
        if arguments.phase == "check":
            results = run_checks(config, reader, device)
            print(f"检查记录保存至 {config.checks_directory()}")
            for name, entry in results["models"].items():
                speed = entry["training_speed"]
                print(
                    f"{name}: 训练速度 {speed['samples_per_second']:.2f} 样本/秒，"
                    f"峰值显存 {speed['peak_memory_gib']:.2f} GiB"
                )
        else:
            records = run_all_models(config, reader, device, resume=arguments.resume)
            for record in records:
                print(
                    f"{record['model']}: 最佳 Epoch {record['best_epoch']}，"
                    f"验证 LogLoss {record['best_validation']['logloss']:.6f}，"
                    f"测试 AUC {record['final_test']['auc']:.6f}，"
                    f"测试 LogLoss {record['final_test']['logloss']:.6f}，"
                    f"测试 NE {record['final_test']['ne']:.6f}"
                )
    print(f"阶段 {arguments.phase} 完成，用时 {format_seconds(stopwatch.elapsed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
