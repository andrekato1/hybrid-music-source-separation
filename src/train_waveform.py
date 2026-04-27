import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
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
EXPERIMENT_NAME = "waveform_run_01"  # update per run
N_EPOCHS        = 100
LEARNING_RATE   = 3e-4
BATCH_SIZE      = 32
SEGMENT_DURATION = 6.0   # seconds
SAMPLES_PER_TRACK = 50
NUM_WORKERS     = 4       # set to 0 on Windows
TARGET_SOURCE   = "vocals"
NOTES           = ""


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
        sources=["vocals"],
    )

    model = WaveformModel()

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name=EXPERIMENT_NAME,
        n_epochs=N_EPOCHS,
        lr=LEARNING_RATE,
        target_source=TARGET_SOURCE,
        notes=NOTES,
    )
