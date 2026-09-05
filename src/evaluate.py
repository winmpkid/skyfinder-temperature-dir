# 本项目新增评估脚本：读取 baseline/LDS checkpoint，统一评估验证集或测试集。
# 官方 DIR 的 imdb-wiki-dir/train.py::validate 报告整体及 many/median/few-shot 指标。
# https://github.com/YyzHarry/imbalanced-regression/blob/main/imdb-wiki-dir/train.py
# 这里使用 MAE、RMSE、5°C 分箱与训练中位数常数基线；没有复制官方 shot_metrics。
# 因而这里的温度分箱表不等同于论文的 many/median/few-shot 评估。

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

try:
    from .config import CHECKPOINT_DIR, RESULTS_DIR
    from .dataset import create_dataloaders
    from .model import build_model
except ImportError:
    from config import CHECKPOINT_DIR, RESULTS_DIR
    from dataset import create_dataloaders
    from model import build_model


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_trained_model(checkpoint_path, device):
    # 本项目接口：按 model.py 创建 ResNet-18，加载本项目 checkpoint 字段。
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint was not found: {checkpoint_path}")

    model = build_model(pretrained=False)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    return model, checkpoint


def collect_predictions(model, data_loader, device):
    # 通用 PyTorch 推理流程：eval + no_grad；LDS 评估时不需要样本权重。
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, temperatures in data_loader:
            images = images.to(device)
            predictions = model(images)

            if predictions.shape != temperatures.shape:
                raise ValueError(
                    f"Prediction shape {predictions.shape} does not match "
                    f"target shape {temperatures.shape}."
                )

            # 每个 batch 都转回 CPU，避免在 GPU 上积累预测结果。
            all_predictions.append(predictions.cpu())
            all_targets.append(temperatures.cpu())

    if not all_predictions:
        raise ValueError("The evaluation DataLoader did not produce any samples.")

    predictions = torch.cat(all_predictions).numpy()
    targets = torch.cat(all_targets).numpy()
    return predictions, targets


def calculate_metrics(predictions, targets):
    # 本项目指标实现：对全部样本计算普通 MAE、RMSE，不使用 LDS 加权误差。
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    if predictions.ndim != 1 or predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must be matching 1-D arrays.")
    if predictions.size == 0:
        raise ValueError("Cannot calculate metrics for empty arrays.")
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("Predictions or targets contain NaN or infinity.")

    errors = predictions - targets
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }

def summarize_temperature_bins(train_data,results,split_name,bin_width=5):
    # 本项目分析：按训练温度确定分箱边界，比较训练频数与各区间预测误差。
    # 此处默认 5°C 是展示粒度，与 LDS 训练权重默认使用的 1°C 分箱不同。
    if bin_width <= 0:
        raise ValueError("bin_width must be greater than 0.")

    train_data = train_data.copy()
    results = results.copy()

    minimum = train_data["temperature"].min()
    maximum = train_data["temperature"].max()

    lower = np.floor( minimum / bin_width) * bin_width
    upper = (np.floor( maximum / bin_width) + 1) * bin_width

    regular_edges = np.arange(
        lower,
        upper + bin_width / 2,
        bin_width,
    )

    edges = np.concatenate(
        [[-np.inf], regular_edges, [np.inf]]
    )

    train_data["temperature_bin"] = pd.cut(
        train_data["temperature"],
        bins=edges,
        right=False,
    )

    results["temperature_bin"] = pd.cut(
        results["true_temperature"],
        bins=edges,
        right=False,
    )

    train_counts = train_data.groupby(
        "temperature_bin",
        observed=False,
    ).size()

    evaluation_stats = results.groupby(
        "temperature_bin",
        observed=False,
    ).agg(**{
        f"{split_name}_count": ("true_temperature", "size"),
        f"{split_name}_mae": ("absolute_error", "mean"),
    })

    summary = evaluation_stats.copy()
    summary.insert(0, "train_count", train_counts)

    return summary.reset_index()

def compare_with_constant_baseline(
    train_data,predictions,targets,split_name,model_label="ResNet-18",
):
    # 本项目新增对照：始终预测训练温度中位数，不用验证或测试标签拟合常数。
    constant_temperature = train_data["temperature"].median()

    constant_predictions = np.full(
        shape=targets.shape,
        fill_value=constant_temperature,
        dtype=np.float64,
    )

    constant_metrics = calculate_metrics(
        constant_predictions,
        targets,
    )

    resnet_metrics = calculate_metrics(
        predictions,
        targets,
    )

    print(
        f"Constant prediction: {constant_temperature:.3f} °C"
    )

    return pd.DataFrame(
        [
            {
                "model": "Training-median baseline",
                f"{split_name}_mae": constant_metrics["mae"],
                f"{split_name}_rmse": constant_metrics["rmse"],
            },
            {
                "model": model_label,
                f"{split_name}_mae": resnet_metrics["mae"],
                f"{split_name}_rmse": resnet_metrics["rmse"],
            },
        ]
    )

def plot_temperature_analysis(summary, output_path, split_name):
    labels = summary["temperature_bin"].astype(str)
    positions = np.arange(len(summary))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
    )

    axes[0].bar(
        positions,
        summary["train_count"],
        color="steelblue",
    )
    axes[0].set_ylabel("Training samples")
    axes[0].set_title(
        f"Training distribution and {split_name} error"
    )

    axes[1].plot(
        positions,
        summary[f"{split_name}_mae"],
        marker="o",
        color="darkorange",
    )
    axes[1].set_ylabel(f"{split_name.title()} MAE (°C)")
    axes[1].set_xlabel("True temperature interval (°C)")

    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved SkyFinder temperature-regression model."
    )
    parser.add_argument(
        "--checkpoint",
        default="best_resnet18.pt",
        help="Checkpoint filename inside checkpoints/, or an absolute path.",
    )
    parser.add_argument(
        "--output-prefix",
        default="baseline",
        help="Prefix for result files, for example baseline or lds.",
    )
    parser.add_argument(
        "--model-label",
        default="ResNet-18",
        help="Model name written to the comparison CSV.",
    )
    parser.add_argument("--bin-width", type=float, default=5.0)
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Held-out split to evaluate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.output_prefix):
        raise ValueError(
            "output-prefix must start with a letter or digit and contain only "
            "letters, digits, underscores or hyphens."
        )
    if not np.isfinite(args.bin_width) or args.bin_width <= 0:
        raise ValueError("bin-width must be finite and greater than zero.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_DIR / CHECKPOINT_DIR / checkpoint_path
    results_dir = PROJECT_DIR / RESULTS_DIR

    train_loader, val_loader, test_loader = create_dataloaders()
    evaluation_loader = val_loader if args.split == "val" else test_loader
    train_data = train_loader.dataset.df.copy()
    model, checkpoint = load_trained_model(checkpoint_path, device)

    print(f"Device: {device}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    split_label = "Validation" if args.split == "val" else "Test"
    print(f"{split_label} samples: {len(evaluation_loader.dataset)}")

    predictions, targets = collect_predictions(model, evaluation_loader, device)
    metrics = calculate_metrics(predictions, targets)

    print(f"{split_label} MAE: {metrics['mae']:.3f} °C")
    print(f"{split_label} RMSE: {metrics['rmse']:.3f} °C")
    if args.split == "val":
        print(f"MAE recorded in checkpoint: {checkpoint['val_mae']:.3f} °C")

    results = evaluation_loader.dataset.df[
        ["image_path", "filename", "camera_id"]
    ].copy()
    if len(results) != len(targets):
        raise ValueError("Prediction count does not match the evaluation dataset.")

    results["true_temperature"] = targets
    results["predicted_temperature"] = predictions
    results["absolute_error"] = np.abs(predictions - targets)

    results_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    predictions_path = results_dir / f"{prefix}_{args.split}_predictions.csv"
    metrics_path = results_dir / f"{prefix}_{args.split}_metrics.csv"
    results.to_csv(predictions_path, index=False)
    pd.DataFrame(
        [{
            "split": args.split,
            "checkpoint_epoch": checkpoint["epoch"],
            "num_samples": len(targets),
            **metrics,
        }]
    ).to_csv(metrics_path, index=False)

    print(f"Predictions saved to: {predictions_path}")
    print(f"Metrics saved to: {metrics_path}")

    summary = summarize_temperature_bins(
        train_data,
        results,
        split_name=args.split,
        bin_width=args.bin_width,
    )

    comparison = compare_with_constant_baseline(
        train_data,
        predictions,
        targets,
        split_name=args.split,
        model_label=args.model_label,
    )

    print("\nTemperature-bin analysis:")
    print(summary.round(3).to_string(index=False))

    print("\nBaseline comparison:")
    print(comparison.round(3).to_string(index=False))

    summary.to_csv(
        results_dir / f"{prefix}_{args.split}_temperature_bins.csv",
        index=False,
    )

    comparison.to_csv(
        results_dir / f"{prefix}_{args.split}_comparison.csv",
        index=False,
    )

    plot_temperature_analysis(
        summary,
        results_dir / f"{prefix}_{args.split}_temperature_analysis.png",
        split_name=args.split,
    )


if __name__ == "__main__":
    main()
