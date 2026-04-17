"""
Training losses for source separation.

Using negative SI-SDR loss, which is directly optimising the metric we care about.
Alternative could be to use L1 loss
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
