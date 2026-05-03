"""
Training losses for source separation.
"""

import torch.nn as nn
import torch.nn.functional as F
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

class MixedLoss(nn.Module):
    """
    Mixed SI-SDR + L1 loss for scale-aware source separation.

    SI-SDR alone is scale-invariant: the model can achieve low loss while
    outputting estimates at an arbitrary amplitude. The L1 term should anchor the
    output scale to match the target amplitude.

    Args:
        alpha: weight on the SI-SDR term. The remaining (1 - alpha) weight goes to L1.
    """

    def __init__(self, alpha: float = 0.9) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, estimate: Tensor, target: Tensor) -> Tensor:
        si_sdr_loss = -compute_si_sdr(estimate, target).mean()
        l1_loss = F.l1_loss(estimate, target)
        return self.alpha * si_sdr_loss + (1 - self.alpha) * l1_loss
