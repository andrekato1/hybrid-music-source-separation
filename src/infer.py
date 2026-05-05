"""
Inference script: load a checkpoint, separate a test track, save WAVs.

Usage (from project root):
    uv run python -m src.infer \
        --checkpoint runs/spectrogram_run_01/best_model.pt \
        --config configs/spectrogram.yaml \
        --out_dir out/run_01 \
        --n_tracks 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torchaudio
import yaml

from src.data.dataset import SAMPLE_RATE, get_dataloaders
from src.metrics import compute_sdr
from src.models.spectrogram.model import SpectrogramBranch
from src.trainer import separate_in_chunks


def main(checkpoint_path: str, config_path: str, out_dir: str, n_tracks: int) -> None:
    sdrs: list[float] = []
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SpectrogramBranch(**cfg["model"])
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    epoch = ckpt.get("epoch", "?")
    val_sdr = ckpt.get("val_sdr", float("nan"))
    print(f"Loaded checkpoint — epoch {epoch}, val SDR {val_sdr:.2f} dB")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    data_cfg = {**cfg["data"], "num_workers": 0}
    _, _, test_loader = get_dataloaders(**data_cfg)

    sample_rate = cfg["data"].get("sample_rate", SAMPLE_RATE)
    chunk_samples = int(cfg["data"]["segment_duration"] * sample_rate)
    target_source = cfg["training"]["target_source"]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for i, (mixture, targets, track_name) in enumerate(test_loader):
        if i >= n_tracks:
            break

        name = track_name[0]
        print(f"\nTrack: {name}")
        print(f"  mixture shape : {tuple(mixture.shape)}   "
              f"({mixture.shape[-1] / sample_rate:.1f}s)")

        with torch.no_grad():
            estimates = separate_in_chunks(
                model, mixture, chunk_samples, chunk_samples // 4, device
            )

        estimate = estimates[target_source]   # [1, C, T]
        reference = targets[target_source]    # [1, C, T]

        # Print amplitude stats so we can tell immediately if output is near-silence
        est_rms = estimate.pow(2).mean().sqrt().item()
        ref_rms = reference.pow(2).mean().sqrt().item()
        mix_rms = mixture.pow(2).mean().sqrt().item()
        print(f"  mixture RMS   : {mix_rms:.4f}")
        print(f"  reference RMS : {ref_rms:.4f}")
        print(f"  estimate RMS  : {est_rms:.4f}")

        est_np = estimate[0].T.numpy()   # [T, C]
        ref_np = reference[0].T.numpy()  # [T, C]
        sdr = compute_sdr(est_np, ref_np)
        sdrs.append(sdr)
        print(f"  SDR           : {sdr:.2f} dB")

        track_dir = out_path / name
        track_dir.mkdir(exist_ok=True)

        def save(tensor: torch.Tensor, fname: str) -> None:
            audio = tensor[0].clamp(-1.0, 1.0)
            torchaudio.save(str(track_dir / fname), audio, sample_rate)

        save(mixture,   "mixture.wav")
        save(reference, f"{target_source}_reference.wav")
        save(estimate,  f"{target_source}_estimate.wav")

        print(f"  Saved to {track_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", default="configs/spectrogram.yaml",
                        help="YAML config whose model section matches the checkpoint")
    parser.add_argument("--out_dir", default="out/", help="Where to write WAV files")
    parser.add_argument("--n_tracks", type=int, default=1,
                        help="Number of test tracks to process")
    args = parser.parse_args()
    main(args.checkpoint, args.config, args.out_dir, args.n_tracks)
