import torch

from src.data.dataset import get_dataloaders
from src.trainer import train


# ---------------------------------------------------------------------------
# Placeholder model — replace with real WaveformModel once implemented
# ---------------------------------------------------------------------------

class PlaceholderModel(torch.nn.Module):
    """
    Passthrough model that returns the mixture as its estimate for every source.
    Useful for verifying the data pipeline and training loop end-to-end
    before the real architecture exists.
    The dummy parameter avoid optimizer crash when testing.
    """

    def __init__(self):
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, mixture):
        # mixture: [B, C, T]
        # Adding dummy * 0 connects the output to the parameter graph so
        # loss.backward() has a path to follow — doesn't change the values
        out = mixture + self._dummy * 0
        return {"drums": out, "bass": out, "other": out, "vocals": out}


# ---------------------------------------------------------------------------
# Config — edit these before each run
# ---------------------------------------------------------------------------

ROOT            = "data/"
EXPERIMENT_NAME = "waveform_run_01"
N_EPOCHS        = 100
LEARNING_RATE   = 3e-4
BATCH_SIZE      = 8
SEGMENT_DURATION = 6.0   # seconds
SAMPLES_PER_TRACK = 2
TARGET_SOURCE   = "vocals"
NOTES           = "placeholder model — pipeline smoke test"


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
    )

    model = PlaceholderModel()

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
