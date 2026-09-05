# DIR 对应：imdb-wiki-dir/train.py 的加权训练、未加权验证与最佳模型选择流程。
# https://github.com/YyzHarry/imbalanced-regression/blob/main/imdb-wiki-dir/train.py
# 本项目训练脚本按 SkyFinder 任务重写，LDS 损失来源与差异见 utils.py。
# 未移植官方 FDS 统计更新、RRT 两阶段训练或学习率衰减流程。
import argparse
import csv
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

try:
    from .config import (
        BATCH_SIZE,
        CHECKPOINT_DIR,
        LEARNING_RATE,
        NUM_EPOCHS,
        RANDOM_SEED,
        RESULTS_DIR,
    )
    from .dataset import create_dataloaders
    from .model import build_model, count_parameters
    from .utils import weighted_l1_loss
except ImportError:
    from config import (
        BATCH_SIZE,
        CHECKPOINT_DIR,
        LEARNING_RATE,
        NUM_EPOCHS,
        RANDOM_SEED,
        RESULTS_DIR,
    )
    from dataset import create_dataloaders
    from model import build_model, count_parameters
    from utils import weighted_l1_loss


PROJECT_DIR = Path(__file__).resolve().parents[1]


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    # Project adaptation: support NVIDIA CUDA, Apple Silicon MPS, and CPU.
    # The official DIR training scripts target CUDA environments.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model,data_loader,criterion,optimizer,device,use_lds=False):
    # DIR 对应：train() 将 Dataset 返回的样本权重传入 weighted_l1_loss。
    # 本项目改动：兼容 baseline 两项 batch 和 LDS 三项 batch，分开记录 loss 与 MAE。
    model.train()
    total_loss = 0.0
    total_absolute_error = 0.0
    total_samples = 0

    for batch in data_loader:
        if use_lds:
            images, temperatures, weights = batch
            weights = weights.to(device)
        else:
            images, temperatures = batch

        images = images.to(device)
        temperatures = temperatures.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        # LDS 沿用官方的逐样本损失加权思路；普通 MAE 另算，用于直接比较两种训练。
        mae = criterion(predictions, temperatures)
        loss = (
            weighted_l1_loss(predictions, temperatures, weights)
            if use_lds
            else mae
        )
        if not torch.isfinite(loss).item():
            raise RuntimeError("Training loss became NaN or infinity.")
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_absolute_error += mae.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("The training DataLoader did not produce any samples.")

    return {
        "loss": total_loss / total_samples,
        "mae": total_absolute_error / total_samples,
    }


def validate_one_epoch(model,data_loader,criterion,device):
    # 沿用官方 validate() 的未加权 L1 评估；验证误差不乘训练样本权重。
    model.eval()
    total_absolute_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, temperatures in data_loader:
            images = images.to(device)
            temperatures = temperatures.to(device)

            predictions = model(images)
            loss = criterion(predictions, temperatures)

            batch_size = images.size(0)
            total_absolute_error += loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("The validation DataLoader did not produce any samples.")

    return total_absolute_error / total_samples


def save_checkpoint(path, model, optimizer, epoch, val_mae, run_config=None):
    # 本项目保存格式：额外记录 run_config，便于核对 LDS 是否启用及实验参数。
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_mae": val_mae,
            "run_config": run_config or {},
        },
        path,
    )


def save_history(path, history):
    # 本项目新增：逐轮 CSV，同时保存加权训练 loss、普通训练 MAE 和验证 MAE。
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["epoch", "train_loss", "train_mae", "val_mae"],
        )
        writer.writeheader()
        writer.writerows(history)


def train_model(
    model,train_loader,val_loader,criterion,optimizer,device,
    num_epochs,checkpoint_path,history_path,use_lds=False,run_config=None,
):
    history = []
    best_val_mae = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(
            model,train_loader,criterion,optimizer,device,use_lds=use_lds,
        )
        train_loss = train_metrics["loss"]
        train_mae = train_metrics["mae"]
        # 与官方 L1 实验的选择标准一致：按未加权验证 MAE 保存最佳模型。
        val_mae = validate_one_epoch(model,val_loader,criterion,device)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "val_mae": val_mae,
        }
        history.append(epoch_metrics)
        save_history(history_path, history)

        improved = val_mae < best_val_mae
        if improved:
            best_val_mae = val_mae
            save_checkpoint(
                checkpoint_path,model,optimizer,epoch,val_mae,
                run_config=run_config,
            )

        marker = "  <-- saved best model" if improved else ""
        loss_label = "weighted train loss" if use_lds else "train loss"
        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} | "
            f"{loss_label}: {train_loss:.3f} | "
            f"train MAE: {train_mae:.3f} | "
            f"val MAE: {val_mae:.3f}{marker}"
        )

    return history, best_val_mae


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a SkyFinder ResNet-18 baseline or an LDS-weighted model."
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not load ImageNet pretrained weights.",
    )
    parser.add_argument(
        "--lds",
        action="store_true",
        help="Use training-only LDS weights and weighted L1 loss.",
    )
    # 本项目新增温度分箱宽度；kernel-size/sigma 对应官方 lds_ks/lds_sigma 的作用。
    # 当前默认值属于本项目实验设置，不代表复现了官方全部超参数。
    parser.add_argument("--lds-bin-width", type=float, default=1.0)
    parser.add_argument("--lds-kernel-size", type=int, default=5)
    parser.add_argument("--lds-sigma", type=float, default=2.0)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional output suffix using letters, digits, underscores or hyphens.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed()
    device = get_device()

    if args.epochs <= 0:
        raise ValueError("epochs must be greater than zero.")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be greater than zero.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and greater than zero.")
    if args.num_workers < 0:
        raise ValueError("num-workers must not be negative.")
    if args.run_name is not None and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name
    ):
        raise ValueError(
            "run-name must start with a letter or digit and contain only "
            "letters, digits, underscores or hyphens."
        )

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_lds=args.lds,
        lds_bin_width=args.lds_bin_width,
        lds_kernel_size=args.lds_kernel_size,
        lds_sigma=args.lds_sigma,
    )
    model = build_model(pretrained=not args.no_pretrained).to(device)
    criterion = nn.L1Loss()
    # 本项目选择 AdamW；官方 imdb-wiki-dir/train.py 提供 Adam 或 SGD。
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    num_epochs = args.epochs
    # 本项目改动：baseline/LDS 使用独立文件名，--run-name 用于保留不同实验。
    run_name = args.run_name if args.run_name is not None else (
        "lds" if args.lds else ""
    )
    suffix = f"_{run_name}" if run_name else ""
    checkpoint_name = f"best_resnet18{suffix}.pt"
    history_name = f"training_history{suffix}.csv"

    checkpoint_path = PROJECT_DIR / CHECKPOINT_DIR / checkpoint_name
    history_path = PROJECT_DIR / RESULTS_DIR / history_name

    run_config = {
        **vars(args),
        "random_seed": RANDOM_SEED,
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "loss": "lds_weighted_l1" if args.lds else "l1",
        "torch_version": str(torch.__version__),
    }

    print(f"Device: {device}")
    print(f"Training mode: {'ResNet + LDS' if args.lds else 'Baseline'}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    if args.lds:
        weights = train_loader.dataset.weights
        info = train_loader.dataset.lds_info
        run_config["lds_bin_start"] = info["bin_start"]
        run_config["lds_kernel"] = info["kernel"].tolist()
        print(
            f"LDS: bin width={args.lds_bin_width} °C, "
            f"kernel size={args.lds_kernel_size}, sigma={args.lds_sigma}"
        )
        print(
            f"Training weights min/mean/max: {weights.min().item():.3f} / "
            f"{weights.mean().item():.3f} / {weights.max().item():.3f}"
        )
    print(f"Checkpoint output: {checkpoint_path}")
    print(f"History output: {history_path}")
    if checkpoint_path.exists() or history_path.exists():
        print(
            "Warning: this run will overwrite existing outputs with these names. "
            "Use a different --run-name to keep previous runs."
        )

    _, best_val_mae = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        use_lds=args.lds,
        run_config=run_config,
    )

    print(f"Best validation MAE: {best_val_mae:.3f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
