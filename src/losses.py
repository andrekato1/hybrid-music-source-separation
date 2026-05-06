"""
Training losses for source separation.
"""

import torch.nn as nn
from torch import Tensor

from .metrics import compute_si_sdr


class SISDRLoss(nn.Module):
    """
    Negative SI-SDR loss. Minimising this maximises SI-SDR.

    Averages over the batch dimension, so the scale is consistent
    regardless of batch size.

    Usage:
        loss_fn = SISDRLoss()
        loss = loss_fn(estimate, target)  # estimate, target: [B, C, T]
        loss.backward()
    """

    def forward(self, estimate: Tensor, target: Tensor) -> Tensor:
        return -compute_si_sdr(estimate, target).mean()


class SISDRWithMagnitudeLoss(nn.Module):
    def __init__(self, mag_weight: float = 0.1) -> None:
        super().__init__()
        self.mag_weight = mag_weight

    def forward(self, estimate: Tensor, target: Tensor) -> Tensor:
        si_sdr_loss = -compute_si_sdr(estimate, target).mean()
        rms_est = estimate.pow(2).mean(dim=-1).sqrt()
        rms_tgt = target.pow(2).mean(dim=-1).sqrt()
        mag_loss = (rms_est - rms_tgt).pow(2).mean()
        return si_sdr_loss + self.mag_weight * mag_loss
