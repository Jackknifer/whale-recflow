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
    parser = argparse.ArgumentParser(description="RecFlow 上的集中必要检查与五项正式训练")
    parser.add_argument("--config", required=True, type=Path, help="配置文件路径")
    parser.add_argument(
        "--phase",
        required=True,
        choices=PHASES,
        help="check 执行必要检查，train 执行五项正式训练",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="按已完成的 Epoch 恢复训练，只在明确需要恢复时使用",
    )
    parser.add_argument(
        "--hourly-rate",
        type=float,
        default=None,
        help="实际实例的每小时单价，用于费用估算",
    )
    arguments = parser.parse_args(argv)
    if arguments.phase == "check" and arguments.hourly_rate is None:
        parser.error("check 阶段需要 --hourly-rate，填写实际实例每小时单价")
    return arguments


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
            results = run_checks(
                config, reader, device, hourly_rate=arguments.hourly_rate
            )
            estimate = results["estimate"]
            print(f"检查记录保存至 {config.checks_directory()}")
            print(
                f"Batch Size {estimate['batch_size']}，"
                f"Epoch 上限 {estimate['epochs']}，"
                f"单价 {estimate['hourly_rate_cny']:.2f} 元/小时"
            )
            for name, entry in estimate["models"].items():
                print(
                    f"{name}: 满 {estimate['epochs']} 轮计算时间 "
                    f"{entry['training_hours']:.4f} 小时，"
                    f"峰值显存 {entry['peak_memory_gib']:.2f} GiB"
                )
            print(
                f"五种模型合计计算时间 {estimate['total_training_hours']:.4f} 小时，"
                f"含开销 {estimate['total_hours_with_overhead']:.4f} 小时，"
                f"估算费用 {estimate['estimated_cost_cny']:.2f} 元"
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
