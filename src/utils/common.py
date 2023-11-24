import math
import os
import random
import sys
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import psutil


@contextmanager
def trace(title):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info().rss / 2.0**30
    yield
    m1 = p.memory_info().rss / 2.0**30
    delta = m1 - m0
    sign = "+" if delta >= 0 else "-"
    delta = math.fabs(delta)
    print(f"[{m1:.1f}GB({sign}{delta:.1f}GB):{time.time() - t0:.1f}sec] {title} ", file=sys.stderr)


def pad_if_needed(x: np.ndarray, max_len: int, pad_value: float = 0.0) -> np.ndarray:
    if len(x) == max_len:
        return x
    num_pad = max_len - len(x)
    n_dim = len(x.shape)
    pad_widths = [(0, num_pad)] + [(0, 0) for _ in range(n_dim - 1)]
    return np.pad(x, pad_width=pad_widths, mode="constant", constant_values=pad_value)


def nearest_valid_size(input_size: int, downsample_rate: int) -> int:
    """
    (x // hop_length) % 32 == 0
    を満たすinput_sizeに最も近いxを返す
    """

    while (input_size // downsample_rate) % 32 != 0:
        input_size += 1
    assert (input_size // downsample_rate) % 32 == 0

    return input_size


def random_crop(pos: int, duration: int, max_end) -> tuple[int, int]:
    """Randomly crops with duration length including pos.
    However, 0<=start, end<=max_end
    """
    start = random.randint(max(0, pos - duration), min(pos, max_end - duration))
    end = start + duration
    return start, end


def negative_sampling(this_event_df: pd.DataFrame, num_steps: int) -> int:
    """negative sampling

    Args:
        this_event_df (pd.DataFrame): event df
        num_steps (int): number of steps in this series

    Returns:
        int: negative sample position
    """
    # onsetとwakupを除いた範囲からランダムにサンプリング
    positive_positions = set(this_event_df[["onset", "wakeup"]].to_numpy().flatten().tolist())
    negative_positions = list(set(range(num_steps)) - positive_positions)
    return random.sample(negative_positions, 1)[0]

onset_features = {
    'w1': 0.97292814,
    'mu1': 2.12861738,
    'sigma1': 1.699476,
    'mu2': 22.7643724,
    'sigma2': 0.42483832
}

awake_features = {
    'w1': 0.47799486,
    'mu1': 10.89539674,
    'sigma1': 0.87151052,
    'mu2': 11.82689931,
    'sigma2': 2.06792452
}


# ref: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/360236#2004730
# def gaussian_kernel(length: int, sigma: int = 3) -> np.ndarray:
#     x = np.ogrid[-length : length + 1]
#     h = np.exp(-(x**2) / (2 * sigma * sigma))  # type: ignore
#     h[h < np.finfo(h.dtype).eps * h.max()] = 0
#     return h

def gaussian_kernel(length: int, w1: float, mu1: float, sigma1: float, mu2: float, sigma2: float) -> np.ndarray:
    x = np.ogrid[-length: length + 1]
    gaussian1 = np.exp(-(x - mu1) ** 2 / (2 * sigma1 ** 2))
    gaussian2 = np.exp(-(x - mu2) ** 2 / (2 * sigma2 ** 2))
    h = w1 * gaussian1 + (1 - w1) * gaussian2
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


# def gaussian_label(label: np.ndarray, offset: int, sigma: int) -> np.ndarray:
#     num_events = label.shape[1]
#     for i in range(num_events):
#         label[:, i] = np.convolve(label[:, i], gaussian_kernel(offset, sigma), mode="same")

#     return label

def gaussian_label(label: np.ndarray, offset: int, onset_features: dict, awake_features: dict) -> np.ndarray:
    num_events = label.shape[1]
    for i in range(num_events):
        if i == 0:
            # onset_features を使用
            kernel = gaussian_kernel(offset, **onset_features)
        else:
            # awake_features を使用
            kernel = gaussian_kernel(offset, **awake_features)

        label[:, i] = np.convolve(label[:, i], kernel, mode="same")

    return label

