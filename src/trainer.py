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
# Evaluation: full tracks
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_source: str = "vocals",
) -> float:
    """
    Evaluate on full tracks (val or test set).
    Returns median SDR in dB over all tracks.

    Note: loader must have batch_size=1 since tracks have variable length.
    Each batch is a single full song.
    """
    model.eval()
    sdrs = []

    chunk_samples = int(30 * 44100)  # 30s chunks to avoid cuDNN LSTM sequence length limit

    for mixture, targets, track_name in tqdm(loader, desc="  eval", leave=False):
        mixture = mixture.to(device)  # [1, C, T]
        target = targets[target_source]  # [1, C, T], kept on CPU for museval
        T = mixture.shape[-1]

        # Per-chunk RMS normalization matches training conditions: SegmentDataset
        # feeds the model unit-RMS segments, so chunks here must also be unit-RMS.
        # Output is rescaled by the same factor to stay on the global-track scale
        # of the reference target.
        chunks = [mixture[..., i:i + chunk_samples] for i in range(0, T, chunk_samples)]
        estimates = []
        for chunk in chunks:
            rms = chunk.pow(2).mean().sqrt().clamp(min=1e-8)
            est = model(chunk / rms)[target_source] * rms
            estimates.append(est)
        estimate = torch.cat(estimates, dim=-1).cpu()  # [1, C, T]

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

    Returns:
        ExperimentLogger with full training history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr / 6)
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
        },
        notes=notes,
    )
    logger = ExperimentLogger(experiment_name, config)

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, target_source
        )

        # Full-track validation is slow — run every N epochs and at the end
        if epoch % val_every_n_epochs == 0 or epoch == n_epochs:
            val_sdr = evaluate(model, val_loader, device, target_source)
            logger.log_epoch(epoch, train_loss, val_sdr)
            logger.maybe_save_best(model, optimizer, epoch, val_sdr)
        else:
            logger.log_epoch(epoch, train_loss, float("nan"))

        scheduler.step()

    logger.finish(model, optimizer, n_epochs)
    return logger
