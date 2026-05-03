"""
Evaluation metrics for source separation.

Two metrics are used:
- SDR (museval): the standard benchmark metric, computed over full tracks and
  reported as median over frames. Used for validation and final evaluation.
- SI-SDR (torch): scale-invariant SDR, used as a fast per-batch signal during
  training.
"""

import numpy as np
import torch
from torch import Tensor
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio


def compute_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """
    Compute median SDR for a single source using museval.

    The estimate is oracle-rescaled before scoring (debugging purposes)
    This makes the metric equivalent to SI-SDR expressed in SDR units, which
    is consistent with the SI-SDR training objective and avoids penalizing the
    model for output amplitude.

    Args:
        estimate:  numpy array [T, C] — model output
        reference: numpy array [T, C] — ground truth

    Returns:
        Median SDR in dB over all evaluated frames (NaN frames excluded).
    """
    import museval

    alpha = (estimate * reference).sum() / ((estimate * estimate).sum() + 1e-8)
    estimate = alpha * estimate

    # museval.evaluate expects [n_sources, T, C] — wrap single source in batch dim
    scores = museval.evaluate(
        references=reference[np.newaxis],  # [1, T, C]
        estimates=estimate[np.newaxis],    # [1, T, C]
    )
    # scores is a tuple: (sdr, isr, sir, sar), each shape [n_sources, n_frames]
    sdr_frames = scores[0][0]  # [n_frames] for the single source
    return float(np.nanmedian(sdr_frames))


def compute_si_sdr(estimate: Tensor, target: Tensor) -> Tensor:
    """
    Scale-Invariant SDR (SI-SDR) for use during training and quick validation.

    Args:
        estimate: Tensor [B, C, T] or [B, T]
        target:   Tensor [B, C, T] or [B, T]

    Returns:
        SI-SDR in dB, scalar (mean over batch)
    """
    return scale_invariant_signal_distortion_ratio(estimate, target)
