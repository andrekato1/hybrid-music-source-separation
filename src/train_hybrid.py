import torch
import torch.nn as nn

from src.data.dataset import get_dataloaders
from src.models.hybrid.model import HybridConcatModel
from src.trainer import train


# ---------------------------------------------------------------------------
# Config — edit these before each run
# ---------------------------------------------------------------------------

ROOT            = "data/"
EXPERIMENT_NAME = "hybrid_concat_drums_4s_b8_s50_50ep_10M_l1"
N_EPOCHS        = 50
LEARNING_RATE   = 3e-4
BATCH_SIZE      = 8
SEGMENT_DURATION = 4.0   # seconds
SAMPLES_PER_TRACK = 50
NUM_WORKERS     = 4       # set to 0 on Windows
TARGET_SOURCE   = "drums"
BASE_CHANNELS   = 24      # waveform branch: ~4.85M params (bottleneck = 16*24 = 384)
LSTM_DIM        = 192
SPEC_CHANNELS   = 24      # spectrogram branch: ~5.64M params (bottleneck channels = 24*16 = 384)
SPEC_DEPTH      = 5
NOTES           = "balanced 10M-param concat hybrid (waveform ~4.85M + spectrogram ~5.64M), pure L1 loss, no waveform output gate"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    train_loader, val_loader, test_loader = get_dataloaders(
        root=ROOT,
        segment_duration=SEGMENT_DURATION,
        batch_size=BATCH_SIZE,
        samples_per_track=SAMPLES_PER_TRACK,
        num_workers=NUM_WORKERS,
        sources=[TARGET_SOURCE],
    )

    model = HybridConcatModel(
        target_source=TARGET_SOURCE,
        base_channels=BASE_CHANNELS,
        lstm_dim=LSTM_DIM,
        spec_channels=SPEC_CHANNELS,
        spec_depth=SPEC_DEPTH,
    )

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name=EXPERIMENT_NAME,
        n_epochs=N_EPOCHS,
        lr=LEARNING_RATE,
        target_source=TARGET_SOURCE,
        val_every_n_epochs=5,
        notes=NOTES,
        loss_fn=nn.L1Loss(),
    )
