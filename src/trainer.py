"""
trainer.py
----------
Generic training loop shared across all three model branches.

Separation of concerns:
- train()           : full training run, handles logging + checkpointing
- train_one_epoch() : one pass over the training segments
- evaluate()        : full-track evaluation (used for val/test)

All three models (waveform, spectrogram, hybrid) use the same trainer
since they share the BaseSeparator interface: forward() takes a mixture
tensor [B, C, T] and returns a dict of source estimates.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import SISDRLoss
from .metrics import compute_sdr
from .experiment import ExperimentConfig, ExperimentLogger, count_parameters


# ---------------------------------------------------------------------------
# Single epoch: training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    target_source: str = "vocals",
    grad_clip: float = 5.0,
) -> float:
    """
    One pass over the training DataLoader (random 4s segments).
    Returns mean loss over all batches.
    """
    model.train()
    total_loss = 0.0

    for mixture, targets in tqdm(loader, desc="  train", leave=False):
        mixture = mixture.to(device)               # [B, C, T]
        target = targets[target_source].to(device) # [B, C, T]

        optimizer.zero_grad()
        estimates = model(mixture)                 # dict[str, Tensor]
        loss = loss_fn(estimates[target_source], target)
        loss.backward()

        # Gradient clipping prevents exploding gradients during early training
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Chunked inference for the spectrogram model
# ---------------------------------------------------------------------------

@torch.no_grad()
def separate_in_chunks(
    model: nn.Module,
    mixture: Tensor,
    chunk_samples: int,
    overlap_samples: int,
    device: torch.device,
) -> dict[str, Tensor]:
    """
    Run model on a full-length mixture [1, C, T] using overlap-add chunking.
    Returns a dict of source estimates on CPU with the same shape as mixture.
    """
    _, C, T = mixture.shape
    stride = chunk_samples - overlap_samples
    fade_in  = torch.linspace(0.0, 1.0, overlap_samples)
    fade_out = torch.linspace(1.0, 0.0, overlap_samples)

    out: dict[str, Tensor] | None = None
    chunk_start = 0
    chunk_idx = 0

    while chunk_start < T:
        chunk_end = min(chunk_start + chunk_samples, T)
        actual = chunk_end - chunk_start
        chunk = mixture[:, :, chunk_start:chunk_end]
        if actual < chunk_samples:
            chunk = F.pad(chunk, (0, chunk_samples - actual))

        # Normalise each chunk to RMS=1 to match training-time SegmentDataset
        # normalization. The model was never trained on inputs outside this scale.
        chunk_rms = chunk.pow(2).mean().sqrt().clamp(min=1e-8)
        estimates = model((chunk / chunk_rms).to(device))

        if out is None:
            out = {src: torch.zeros(1, C, T) for src in estimates}

        is_first = chunk_idx == 0
        is_last  = chunk_end >= T

        for src, est in estimates.items():
            seg = (est.cpu() * chunk_rms)[:, :, :actual].clone()
            if not is_first and overlap_samples > 0:
                ov = min(overlap_samples, actual)
                seg[:, :, :ov] *= fade_in[:ov].reshape(1, 1, -1)
            if not is_last and overlap_samples > 0:
                ov = min(overlap_samples, actual)
                seg[:, :, actual - ov:] *= fade_out[:ov].reshape(1, 1, -1)
            out[src][:, :, chunk_start:chunk_end] += seg

        chunk_idx += 1
        chunk_start += stride

    return out


# ---------------------------------------------------------------------------
# Evaluation: full tracks
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_source: str = "vocals",
    chunk_samples: int = 4 * 44100,
    overlap_samples: int | None = None,
) -> float:
    """
    Evaluate on full tracks (val or test set).
    Returns median SDR in dB over all tracks.

    Note: loader must have batch_size=1 since tracks have variable length.
    Each batch is a single full song.
    """
    if overlap_samples is None:
        overlap_samples = chunk_samples // 4

    model.eval()
    sdrs = []

    for mixture, targets, track_name in tqdm(loader, desc="  eval", leave=False):
        # mixture stays on CPU; separate_in_chunks handles device placement
        target = targets[target_source]  # [1, C, T], kept on CPU for museval

        estimates = separate_in_chunks(model, mixture, chunk_samples, overlap_samples, device)
        estimate = estimates[target_source]  # already on CPU

        # museval expects numpy arrays of shape [T, C]
        est_np = estimate[0].T.numpy()   # [T, C]
        ref_np = target[0].T.numpy()     # [T, C]

        sdr = compute_sdr(est_np, ref_np)
        sdrs.append(sdr)

    return float(np.nanmedian(sdrs))


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    experiment_name: str,
    n_epochs: int = 100,
    lr: float = 3e-4,
    target_source: str = "vocals",
    val_every_n_epochs: int = 5,
    device: Optional[torch.device] = None,
    seed: int = 42,
    notes: str = "",
    loss_fn: Optional[nn.Module] = None,
) -> ExperimentLogger:
    """
    Full training loop with logging and checkpointing.

    Args:
        model:               any BaseSeparator subclass
        train_loader:        yields (mixture, targets) — segments, batch_size > 1
        val_loader:          yields (mixture, targets, name) — full tracks, batch_size=1
        experiment_name:     name for the run directory under runs/
        n_epochs:            total training epochs
        lr:                  Adam learning rate
        target_source:       which stem to separate (default: "vocals")
        val_every_n_epochs:  run full-track eval every N epochs (expensive)
        device:              defaults to CUDA if available
        notes:               free-text note saved in config.json
        loss_fn:             training loss module; defaults to SISDRLoss()

    Returns:
        ExperimentLogger with full training history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if loss_fn is None:
        loss_fn = SISDRLoss()

    config = ExperimentConfig(
        model_name=model.__class__.__name__,
        parameter_count=count_parameters(model),
        hyperparameters={
            "lr": lr,
            "n_epochs": n_epochs,
            "target_source": target_source,
            "val_every_n_epochs": val_every_n_epochs,
            "batch_size": train_loader.batch_size,
            "seed": seed,
            "loss_fn": loss_fn.__class__.__name__,
        },
        notes=notes,
    )
    logger = ExperimentLogger(experiment_name, config)

    chunk_samples = train_loader.dataset.segment_length

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, target_source
        )

        # Full-track validation is slow — run every N epochs and at the end
        if epoch % val_every_n_epochs == 0 or epoch == n_epochs:
            val_sdr = evaluate(model, val_loader, device, target_source, chunk_samples)
            logger.log_epoch(epoch, train_loss, val_sdr)
            logger.maybe_save_best(model, optimizer, epoch, val_sdr)
        else:
            logger.log_epoch(epoch, train_loss, float("nan"))

    logger.finish(model, optimizer, n_epochs)
    return logger
