# DIR method source: Yang et al., Delving into Deep Imbalanced Regression,
# ICML 2021.
# Official repository: https://github.com/YyzHarry/imbalanced-regression
# The functions below correspond to the official imdb-wiki-dir LDS and weighted
# L1 implementations, reimplemented for this project's interface.
# Only LDS is implemented; fds.py and feature mean/variance calibration are not
# ported.
import numpy as np
import torch

def get_lds_kernel_window(kernel_size=5, sigma=2.0):
    # DIR correspondence: imdb-wiki-dir/utils.py::get_lds_kernel_window.
    # https://github.com/YyzHarry/imbalanced-regression/blob/main/imdb-wiki-dir/utils.py
    # Project change: retain only the Gaussian kernel, add input validation, and
    # calculate the discrete Gaussian directly with NumPy.
    # The official code smooths an impulse with gaussian_filter1d and normalizes
    # by the peak; this implementation normalizes by the kernel sum.
    # Official boundary handling also affects kernel shape, so identical
    # size/sigma values do not guarantee numerically identical kernels.
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
    # DIR correspondence: imdb-wiki-dir/datasets.py::IMDBWIKI._prepare_weights.
    # https://github.com/YyzHarry/imbalanced-regression/blob/main/imdb-wiki-dir/datasets.py
    # Follow the LDS workflow: count labels -> smooth density -> invert density
    # -> normalize sample weights to mean 1.
    # Project change: separate weighting from the Dataset and replace age buckets
    # with adjustable-width bins that support negative temperatures.
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

    # Project change: official age labels use fixed 0..120 buckets and
    # int(label); this implementation determines the range dynamically.
    # Align the starting edge downward to a multiple of bin_width so negative
    # temperatures still map to non-negative bin indices.
    bin_start = np.floor(temperatures.min() / bin_width) * bin_width
    bin_indices = np.floor((temperatures - bin_start) / bin_width).astype(np.int64)
    counts = np.bincount(bin_indices).astype(np.float64)
    # Project implementation: zero padding plus np.convolve, corresponding to
    # the official convolve1d(..., mode='constant').
    # Both have the same smoothing interpretation for identical symmetric
    # kernels and counts, but this project uses different kernel normalization
    # and count preprocessing.
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

    # DIR idea: map the smoothed bin density back to each sample and invert it so
    # sparse intervals receive larger weights.
    # Difference: the official inverse branch clips counts to [5, 1000], while
    # the sqrt_inv branch takes the square root first.
    # This implementation smooths raw counts directly and adds only 1e-8 to
    # prevent division by zero; it does not clip counts or cap sample weights.
    sample_density = smoothed_counts[bin_indices]
    weights = 1.0 / np.maximum(sample_density, 1e-8)

    # Preserve the official per-sample normalization: dividing by the mean is
    # equivalent to multiplying by N / sum(weights).
    weights = weights / weights.mean()

    # Project addition: return binning and smoothing details for logs,
    # checkpoints, and experiment analysis.
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
    # DIR correspondence: imdb-wiki-dir/loss.py::weighted_l1_loss.
    # https://github.com/YyzHarry/imbalanced-regression/blob/main/imdb-wiki-dir/loss.py
    # Preserve the mathematical form mean(weight * abs(prediction - target)),
    # reimplemented for this project's interface.
    # Project change: require [batch_size], validate shapes, and align weight
    # device/dtype with the predictions.
    # The official function accepts weights=None and broadcasts with expand_as;
    # this implementation requires explicit weights to prevent unintended
    # broadcasting.
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
