# DIR 方法来源：Yang et al., Delving into Deep Imbalanced Regression, ICML 2021。
# 官方仓库：https://github.com/YyzHarry/imbalanced-regression
# 下列函数对应官方 imdb-wiki-dir 的 LDS 与加权 L1 实现，按本项目接口重新实现。
# 当前只实现 LDS；未移植 fds.py 或特征均值、方差校准代码。
import numpy as np
import torch

def get_lds_kernel_window(kernel_size=5, sigma=2.0):
    # DIR 对应：imdb-wiki-dir/utils.py::get_lds_kernel_window。
    # https://github.com/YyzHarry/imbalanced-regression/blob/main/imdb-wiki-dir/utils.py
    # 本项目改动：仅保留 Gaussian，增加参数校验，用 NumPy 直接计算离散高斯。
    # 官方用 gaussian_filter1d 平滑脉冲并按峰值归一化；这里按核总和归一化。
    # 官方滤波的边界处理也影响核形状，因此同样的 size/sigma 不保证数值等同。
    if(
        not isinstance(kernel_size, int)
        or kernel_size % 2 == 0
        or kernel_size <= 0
    ):
        raise ValueError("kernel_size must be a positive odd integer")

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a finite number and greater than 0")

    half_size = kernel_size // 2
    offsets = np.arange(-half_size, half_size + 1,dtype=np.float64)

    kernel = np.exp(
        -0.5 * (offsets / sigma) ** 2
    )

    return kernel / kernel.sum()

def compute_lds_weights(
    temperatures,
    bin_width=1.0,
    kernel_size=5,
    sigma=2.0,
):
    # DIR 对应：imdb-wiki-dir/datasets.py::IMDBWIKI._prepare_weights。
    # https://github.com/YyzHarry/imbalanced-regression/blob/main/imdb-wiki-dir/datasets.py
    # 沿用 LDS 流程：统计标签频数 -> 平滑密度 -> 取倒数 -> 样本权重均值归一化。
    # 本项目改动：从 Dataset 拆出独立函数，将年龄桶改为支持负温度的可调宽度分箱。
    temperatures = np.asarray(temperatures, dtype=np.float64)
    if temperatures.ndim != 1 or temperatures.size == 0:
        raise ValueError(
            "temperatures must be a non-empty 1-D array."
        )

    if not np.isfinite(temperatures).all():
        raise ValueError(
            "temperatures contain NaN or infinity."
        )

    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError(
            "bin_width must be finite and greater than zero."
        )

    kernel = get_lds_kernel_window(kernel_size = kernel_size, sigma = sigma)

    # 本项目改动：官方年龄标签使用固定 0..120 桶及 int(label)；这里动态确定范围。
    # 分箱起点向下对齐到 bin_width 的整数倍，负温度也可得到非负桶索引。
    bin_start = np.floor(temperatures.min() / bin_width) * bin_width
    bin_indices = np.floor((temperatures - bin_start) / bin_width).astype(np.int64)
    counts = np.bincount(bin_indices).astype(np.float64)
    # 本项目实现：零填充 + np.convolve；对应官方 convolve1d(..., mode='constant')。
    # 两者在相同对称核和频数下具有相同的平滑含义，但本项目核与计数预处理不同。
    half_size = kernel_size // 2
    padded_counts = np.pad(
        counts,
        (half_size, half_size),
        mode="constant",
    )

    smoothed_counts = np.convolve(
        padded_counts,
        kernel,
        mode="valid",
    )

    # DIR 思路：将平滑后的桶密度映射回样本，再取倒数，稀疏区间获得更大权重。
    # 差异：官方 inverse 分支先把频数裁剪到 [5, 1000]，sqrt_inv 分支先开平方。
    # 当前直接平滑原始频数，只加 1e-8 防除零；没有计数裁剪或样本权重上限。
    sample_density = smoothed_counts[bin_indices]
    weights = 1.0 / np.maximum(sample_density, 1e-8)

    # 沿用官方按样本归一化的结果：除以均值等价于乘以 N / sum(weights)。
    weights = weights / weights.mean()

    # 本项目新增：返回分箱与平滑信息，供日志、checkpoint 和实验分析使用。
    info = {
        "bin_start": float(bin_start),
        "bin_width": float(bin_width),
        "bin_indices": bin_indices,
        "counts": counts,
        "smoothed_counts": smoothed_counts,
        "kernel": kernel,
    }

    return weights.astype(np.float32), info


def weighted_l1_loss(predictions, targets, weights):
    # DIR 对应：imdb-wiki-dir/loss.py::weighted_l1_loss。
    # https://github.com/YyzHarry/imbalanced-regression/blob/main/imdb-wiki-dir/loss.py
    # 沿用数学形式 mean(weight * abs(prediction - target))，通过本项目接口重写。
    # 本项目改动：统一要求 [batch_size]，增加形状校验和权重 device/dtype 对齐。
    # 官方允许 weights=None 并用 expand_as 广播；这里要求显式权重以防意外广播。
    if predictions.ndim != 1 or predictions.numel() == 0:
        raise ValueError(
            "predictions must have shape [batch_size]."
        )

    if (
        predictions.shape != targets.shape
        or predictions.shape != weights.shape
    ):
        raise ValueError(
            "predictions, targets, and weights "
            "must have identical shapes."
        )

    weights = weights.to(
        device=predictions.device,
        dtype=predictions.dtype,
    )

    errors = torch.abs(predictions - targets)

    return (weights * errors).mean()
