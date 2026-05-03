import torch
#print(f"CUDA available: {torch.cuda.is_available()}")
#print(f"Device: {torch.cuda.get_device_name(0)}")
from src.data.dataset import get_dataloaders
from src.models.waveform.model import WaveformModel
from src.trainer import train


# ---------------------------------------------------------------------------
# Placeholder model — kept for pipeline smoke testing
# ---------------------------------------------------------------------------

# class PlaceholderModel(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self._dummy = torch.nn.Parameter(torch.zeros(1))
#
#     def forward(self, mixture):
#         out = mixture + self._dummy * 0
#         return {"drums": out, "bass": out, "other": out, "vocals": out}


# ---------------------------------------------------------------------------
# Config — edit these before each run
# ---------------------------------------------------------------------------

ROOT            = "data/"
EXPERIMENT_NAME = "waveform_blstm_5lvl_drums_4s_b32_s50_50ep_tanhgate"
N_EPOCHS        = 50
LEARNING_RATE   = 3e-4
BATCH_SIZE      = 8
SEGMENT_DURATION = 4.0   # seconds
SAMPLES_PER_TRACK = 50
NUM_WORKERS     = 4       # set to 0 on Windows
TARGET_SOURCE   = "drums"
BASE_CHANNELS   = 32      # 5-level encoder → bottleneck channels = 16*32 = 512 (matches spectrogram branch)
LSTM_DIM        = 320     # tuned to bring total param count near spectogram 10.6M
NOTES           = "drums, 5-level encoder, plain SI-SDR loss; output gated as tanh(decoder_out) * mixture to anchor output amplitude to mixture amplitude"


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

    model = WaveformModel(base_channels=BASE_CHANNELS, lstm_dim=LSTM_DIM, target_source=TARGET_SOURCE)

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
    )
