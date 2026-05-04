"""
Train the spectrogram branch of Hybrid Demucs.

Usage (run from the project root):
    uv run python -m src.train_spectrogram
    uv run python -m src.train_spectrogram --config configs/spectrogram.yaml
"""

import argparse
import sys
from pathlib import Path

# Make `from src.*` imports work when the script is run directly as a file
# (python src/train_spectrogram.py) in addition to the module form (-m src.train_spectrogram).
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import yaml

from src.data.dataset import get_dataloaders
from src.models.spectrogram.model import SpectrogramBranch
from src.trainer import train

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["experiment"]["seed"])

    train_loader, val_loader, _ = get_dataloaders(**cfg["data"])
    model = SpectrogramBranch(**cfg["model"])

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name=cfg["experiment"]["name"],
        notes=cfg["experiment"]["notes"],
        seed=cfg["experiment"]["seed"],
        loss_fn=nn.L1Loss(),
        **cfg["training"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/spectrogram.yaml",
        help="Path to the YAML config file (default: configs/spectrogram.yaml)",
    )
    args = parser.parse_args()
    main(args.config)
