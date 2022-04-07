"""Augmentations pipeline for self-supervised pre-training. 

Two variations of augmentations are used. Jittering, where random uniform noise is added to the EEG signal depending on its peak-to-peak values, 
along with masking, where signals are masked randomly. Flipping, where the EEG signal is horizontally flipped randomly, and scaling, where EEG 
signal is scaled with Gaussian noise.

This file can also be imported as a module and contains the following functions:

    * add_noise - Adds low-frequency and high-frequency noise to the EEG signal.
    * jitter - Applies random uniform noise to the EEG signal.
    * scaling - Applies Gaussian noise to the EEG signal.
    * masking - Masks a single segment of the EEG signal randomly.
    * multi_masking - Masks multiple segments of the EEG signal randomly.
    * flip - Applies a horizontal flip to the EEG signal.
    * augment - Builds the augmentations pipeline.

"""
__author__ = "Likith Reddy, Vamsi Kumar"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com"

from typing import Optional, Tuple

import numpy as np
import torch
from scipy.interpolate import interp1d


def add_noise(ts: np.ndarray, degree: float):

    len_ts = len(ts)
    num_range = np.ptp(ts) + 1e-4  # add a small number for flat signal

    noise1 = degree * num_range * (2 * np.random.rand(len_ts) - 1)
    noise2 = degree * num_range * (2 * np.random.rand(len_ts // 100) - 1)
    x_old = np.linspace(0, 1, num=len_ts // 100, endpoint=True)
    x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
    f = interp1d(x_old, noise2, kind="linear")
    noise2 = f(x_new)
    out_ts = ts + noise1 + noise2

    return out_ts


def jitter(x: np.ndarray, config, degree: float = 1.0):

    ret = []
    for chan in range(x.shape[0]):
        ret.append(add_noise(x[chan], config.degree * degree))
    ret = np.vstack(ret)
    ret = torch.from_numpy(ret)
    return ret


def scaling(x: np.ndarray, config, degree: float = 2.):

    ret = np.zeros_like(x)
    degree = config.degree * (degree + np.random.rand())
    factor = 2. * np.random.normal(size=x.shape[1]) - 1
    factor = 1.5 + (2. * np.random.rand()) + degree * factor
    for i in range(x.shape[0]):
        ret[i] = x[i] * factor
    ret = torch.from_numpy(ret)
    return ret


def masking(x: np.ndarray, config):

    segments = config.mask_min_points + int(
        np.random.rand() * (config.mask_max_points - config.mask_min_points)
    )
    points = np.random.randint(0, 3000 - segments)
    ret = x.detach().clone()
    for i, k in enumerate(x):
        ret[i, points : points + segments] = 0

    return ret


def multi_masking(
    x: np.ndarray,
    mask_min: int = 40,
    mask_max: int = 10,
    min_seg: int = 8,
    max_seg: int = 14,
):

    fin_masks = []
    segments = min_seg + int(np.random.rand() * (max_seg - min_seg))
    for seg in range(segments):
        fin_masks.append(mask_min + int(np.random.rand() * (mask_max - mask_min)))
    points = np.random.randint(0, 3000 - segments, size=segments)
    ret = x.clone()
    for i, k in enumerate(x):
        for seg in range(segments):
            ret[i, points[seg] : points[seg] + fin_masks[seg]] = 0
    return ret


def flip(x: np.ndarray):

    if np.random.rand() > 0.5:
        return torch.tensor(np.flip(x.numpy(), 1).copy())
    else:
        return x


def augment(x: torch.Tensor, config, masking_type: Optional[str] = None, degree: float = 3.0) -> Tuple[torch.Tensor, torch.Tensor]:

    if masking_type is None:
        weak_augment = masking(jitter(x, config), config)
    else:
        weak_augment = multi_masking(jitter(x, config), config)
    strong_augment = scaling(flip(x, config), config, degree=degree)
    return weak_augment, strong_augment

