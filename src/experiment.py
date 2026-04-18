"""
experiment tracking for training runs.

Each run is saved to:
    runs/<experiment_name>/
        config.json     — model name, parameter count, hyperparameters, notes
        metrics.json    — per-epoch train loss and val SDR
        best_model.pt   — checkpoint saved whenever val SDR improves
        final_model.pt  — checkpoint saved at the end of training

Usage:
    config = ExperimentConfig(
        model_name="WaveformModel",
        parameter_count=count_parameters(model),
        hyperparameters={"lr": 3e-4, "batch_size": 8, "segment_duration": 4.0},
        notes="baseline run, no augmentation",
    )
    logger = ExperimentLogger("waveform_run_01", config)

    for epoch in range(1, n_epochs + 1):
        train_loss = ...
        val_sdr = ...
        logger.log_epoch(epoch, train_loss, val_sdr)
        logger.maybe_save_best(model, optimizer, epoch, val_sdr)

    logger.finish(model, optimizer, n_epochs)
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Captures everything needed to reproduce a run."""
    model_name: str
    parameter_count: int
    hyperparameters: dict[str, Any]
    notes: str = ""


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_sdr: float           # dB; float("nan") on epochs where val was skipped
    elapsed_seconds: float   # wall-clock time since training started


# ---------------------------------------------------------------------------
# Experiment logger
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """Tracks one training run end-to-end."""

    def __init__(
        self,
        experiment_name: str,
        config: ExperimentConfig,
        runs_dir: str = "runs",
    ) -> None:
        self.experiment_name = experiment_name
        self.config = config
        self.run_dir = Path(runs_dir) / experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.history: list[EpochRecord] = []
        self.best_val_sdr: float = float("-inf")
        self._start_time: float = time.time()

        # Write config immediately so it's there even if training crashes
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)

        print(f"Experiment '{experiment_name}' started.")
        print(f"  Model:      {config.model_name}")
        print(f"  Parameters: {config.parameter_count:,}")
        print(f"  Run dir:    {self.run_dir.resolve()}")

    # ------------------------------------------------------------------
    # Per-epoch logging
    # ------------------------------------------------------------------

    def log_epoch(self, epoch: int, train_loss: float, val_sdr: float) -> None:
        """Record metrics for one epoch and persist to disk."""
        elapsed = time.time() - self._start_time
        record = EpochRecord(epoch, train_loss, val_sdr, elapsed)
        self.history.append(record)

        # Overwrite metrics file each epoch so it's always up to date
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump([asdict(r) for r in self.history], f, indent=2)

        val_str = f"{val_sdr:.2f} dB" if val_sdr == val_sdr else "—"  # nan check
        best_marker = " ↑ best" if val_sdr > self.best_val_sdr else ""
        print(f"  Epoch {epoch:03d} | loss {train_loss:.4f} | val SDR {val_str}{best_marker}")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        tag: str = "best",
    ) -> None:
        """Save model + optimizer state to <tag>_model.pt."""
        path = self.run_dir / f"{tag}_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_sdr": self.best_val_sdr,
                "config": asdict(self.config),
            },
            path,
        )

    def maybe_save_best(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_sdr: float,
    ) -> bool:
        """Save checkpoint if this is the best val SDR so far. Returns True if saved."""
        if val_sdr > self.best_val_sdr:
            self.best_val_sdr = val_sdr
            self.save_checkpoint(model, optimizer, epoch, tag="best")
            return True
        return False

    # ------------------------------------------------------------------
    # End of training
    # ------------------------------------------------------------------

    def finish(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> None:
        """Save final model and print training summary."""
        self.save_checkpoint(model, optimizer, epoch, tag="final")
        total_minutes = (time.time() - self._start_time) / 60

        print(f"\n{'=' * 55}")
        print(f"Experiment : {self.experiment_name}")
        print(f"Model      : {self.config.model_name}")
        print(f"Parameters : {self.config.parameter_count:,}")
        print(f"Best val SDR: {self.best_val_sdr:.2f} dB")
        print(f"Total time  : {total_minutes:.1f} min")
        print(f"Saved to    : {self.run_dir.resolve()}")
        print(f"{'=' * 55}\n")


# ---------------------------------------------------------------------------
# Standalone helpers (shared across the project)
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters. Call this to verify parameter budget."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, float | None]:
    """
    Load a saved checkpoint into model (and optionally optimizer).

    Returns:
        (epoch, val_sdr) from the checkpoint.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint.get("val_sdr")
