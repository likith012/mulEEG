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
__email__ = "likith012@gmail.com, vamsi81523@gmail.com"


from typing import Optional, Tuple

import numpy as np
import torch
from scipy.interpolate import interp1d


def add_noise(x: torch.Tensor, degree: float) -> torch.Tensor:
    """Adds low-frequency and high-frequency noise to the EEG signal.

    Parameters
    ----------
    x: torch.Tensor
        Input EEG signals.
    degree: float
        Degree of low and high frequency noise to be added.     

    Returns
    -------
    torch.Tensor
        Augmented EEG signals.
    """

    len_x = len(x)
    num_range = np.ptp(x) + 1e-4  # add a small number for flat signal

    noise_high_frequency = degree * num_range * (2.0 * np.random.rand(len_x) - 1)
    noise_low_frequency = degree * num_range * (2.0 * np.random.rand(len_x // 100) - 1)
    x_old = np.linspace(0, 1, num=len_x // 100, endpoint=True)
    x_new = np.linspace(0, 1, num=len_x, endpoint=True)
    interpolation = interp1d(x_old, noise_low_frequency, kind="linear")
    noise2 = interpolation(x_new)
    out_x = x + noise2 + noise_high_frequency
    return out_x


def jitter(x: torch.Tensor, config, degree: float = 1.0) -> torch.Tensor:
    """Applies random uniform noise to the EEG signal.

    Parameters
    ----------
    x: torch.Tensor
        Input EEG signals.
    config
        Configuration object.
    degree: float, optional
        Degree of uniform noise to be added.

    Returns
    -------
    torch.Tensor
        Augmented EEG signals.
    """

    ret = []
    for ch in range(x.shape[0]):
        ret.append(add_noise(x[ch], config.degree * degree))
    ret = np.vstack(ret)
    ret = torch.from_numpy(ret)
    return ret


def scaling(x: torch.Tensor, config, degree: float = 2.0) -> torch.Tensor:
    """Applies Gaussian noise to the EEG signal.

    Parameters
    ----------
     x: torch.Tensor
        Input EEG signals.
    config
        Configuration object.
    degree: float, optional
        Degree of Gaussian noise to be added.

    Returns
    -------
    torch.Tensor
        Augmented EEG signals.
    """

    ret = np.zeros_like(x)
    degree = config.degree * (degree + np.random.rand())
    factor = 2.0 * np.random.normal(size=x.shape[1]) - 1
    factor = 1.5 + (2.0 * np.random.rand()) + degree * factor
    for i in range(x.shape[0]):
        ret[i] = x[i] * factor
    ret = torch.from_numpy(ret)
    return ret


def masking(x: torch.Tensor, config) -> torch.Tensor:
    """Masks a single segment of the EEG signal randomly.

    Parameters
    ----------
    x: torch.Tensor
        Input EEG signals.
    config
        Configuration object.

    Returns
    -------
    torch.Tensor
        Augmented EEG signals.
    """

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
) -> torch.Tensor:
    """Masks multiple segments of the EEG signal randomly.

    Parameters
    ----------
    x: torch.Tensor
        Input EEG signals.
    mask_min: int, optional
        Minimum number of points to be masked in a segment.
    mask_max: int, optional
        Maximum number of points to be masked in a segment.
    min_seg: int, optional
        Minimum number of points segments.
    max_seg: int, optional
        Maximum number of points segments.

    Returns
    -------
    torch.Tensor
        Augmented EEG signals.
    """

    fin_masks = []
    segments = min_seg + int(np.random.rand() * (max_seg - min_seg))
    for seg in range(segments):
        fin_masks.append(mask_min + int(np.random.rand() * (mask_max - mask_min)))
    points = np.random.randint(0, 3000 - segments, size=segments)
    ret = x.clone()
    for i, _ in enumerate(x):
        for seg in range(segments):
            ret[i, points[seg] : points[seg] + fin_masks[seg]] = 0
    return ret


def flip(x: torch.Tensor) -> torch.Tensor:
    """Flips the input EEG Signal.

    Parameters
    ----------
    inputs: torch.Tensor
        Input EEG signal.

    Returns
    -------
    torch.Tensor
        Flipped EEG signal.

    """

    if np.random.rand() > 0.5:
        return torch.tensor(np.flip(x.numpy(), 1).copy())
    else:
        return x


def augment(
    x: torch.Tensor, config, masking_type: Optional[str] = None, degree: float = 3.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies strong and weak kind of augmentations on the EEG Signal.

    Parameters
    ----------
    inputs: torch.tensor
        Input EEG signal.
    config
        Configuration object.
    masking_type: str, optional
        Type of masking to be applied.
    degree: float, optional
        Degree of noise to be added.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
       Two different augmented views of the input EEG Signal.

    """

    if masking_type is None:
        weak_augment = masking(jitter(x, config), config)
    else:
        weak_augment = multi_masking(jitter(x, config), config)
    strong_augment = scaling(flip(x, config), config, degree=degree)
    return weak_augment, strong_augment

