import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import musdb

SOURCES: list[str] = ["drums", "bass", "other", "vocals"]
SAMPLE_RATE: int = 44100
SEGMENT_DURATION: float = 4.0  # seconds

# 16 randomly chosen indices held out from the 100 musdb18 train tracks for validation
_VAL_INDICES: frozenset[int] = frozenset(
    [14, 3, 27, 8, 41, 19, 6, 33, 52, 11, 24, 39, 2, 46, 17, 30]
)


def load_musdb(root: str = "data/", is_wav: bool = True) -> tuple[list, list, list]:
    """Return (train_tracks, val_tracks, test_tracks) from the musdb18 root directory.

    Args:
        root:    path to the musdb18 folder containing train/ and test/ subdirectories.
        is_wav:  set True if the data has been pre-converted to WAV format (much faster
                 loading). Set False (default) to read directly from .stem.mp4 files.
                 See convert_to_wav.py for the one-time conversion step.
    """
    db = musdb.DB(root=root, is_wav=is_wav)
    all_train = db.load_mus_tracks(subsets="train")
    test_tracks = db.load_mus_tracks(subsets="test")
    train_tracks = [t for i, t in enumerate(all_train) if i not in _VAL_INDICES]
    val_tracks = [t for i, t in enumerate(all_train) if i in _VAL_INDICES]
    return train_tracks, val_tracks, test_tracks


# --- Dataset classes ---
# PyTorch's DataLoader requires objects that subclass Dataset, so these must be classes.
# Each class only needs __len__ (how many items) and __getitem__ (get item by index).


class SegmentDataset(Dataset):
    """Training dataset: loads a random fixed-length segment from each track.

    samples_per_track controls how many random crops are drawn per track per epoch.
    With 84 training tracks, samples_per_track=50 gives 4,200 segments per epoch —
    a reasonable training set size. samples_per_track=1 would give only 84 steps/epoch,
    which is too few for meaningful learning.
    """

    def __init__(
        self,
        tracks: list,
        segment_duration: float = SEGMENT_DURATION,
        sample_rate: int = SAMPLE_RATE,
        sources: list[str] = SOURCES,
        samples_per_track: int = 50,
        normalize: bool = True,
    ) -> None:
        self.tracks = tracks
        self.segment_length = int(segment_duration * sample_rate)
        self.sample_rate = sample_rate
        self.sources = sources
        self.samples_per_track = samples_per_track
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.tracks) * self.samples_per_track

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor]]:
        track = self.tracks[idx % len(self.tracks)]
        total_samples = int(track.duration * self.sample_rate)

        # torch.randint is seeded by torch.manual_seed, keeping runs reproducible
        start = torch.randint(0, total_samples - self.segment_length, (1,)).item()

        # Load only the segment we need — avoids reading the full track into memory.
        # Note: chunk_start/chunk_duration are read by musdb before calling track.audio,
        # so we set them here and immediately read the audio before any other worker
        # can overwrite them. Safe with num_workers=0; see note in get_dataloaders.
        track.chunk_start = start / self.sample_rate
        track.chunk_duration = self.segment_length / self.sample_rate

        mixture = torch.from_numpy(track.audio.T).float()  # [C, T]
        targets = {
            src: torch.from_numpy(track.targets[src].audio.T).float()
            for src in self.sources
        }

        if self.normalize:
            # RMS-normalise using the mixture's energy.
            # The same scale factor is applied to all targets so the
            # mixture-to-source relationship the model must learn is preserved.
            rms = mixture.pow(2).mean().sqrt().clamp(min=1e-8)
            mixture = mixture / rms
            targets = {src: t / rms for src, t in targets.items()}

        return mixture, targets


class FullTrackDataset(Dataset):
    """Validation / test dataset: loads complete tracks (variable length)."""

    def __init__(
        self,
        tracks: list,
        sources: list[str] = SOURCES,
        normalize: bool = True,
    ) -> None:
        self.tracks = tracks
        self.sources = sources
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor], str]:
        track = self.tracks[idx]
        track.chunk_start = 0
        track.chunk_duration = track.duration

        mixture = torch.from_numpy(track.audio.T).float()  # [C, T]
        targets = {
            src: torch.from_numpy(track.targets[src].audio.T).float()
            for src in self.sources
        }

        if self.normalize:
            rms = mixture.pow(2).mean().sqrt().clamp(min=1e-8)
            mixture = mixture / rms
            targets = {src: t / rms for src, t in targets.items()}

        return mixture, targets, track.name


def get_dataloaders(
    root: str = "data/",
    is_wav: bool = True,
    segment_duration: float = SEGMENT_DURATION,
    sample_rate: int = SAMPLE_RATE,
    sources: list[str] = SOURCES,
    batch_size: int = 8,
    samples_per_track: int = 50,
    normalize: bool = True,
    num_workers: int = 0,  # keep at 0 on Windows; increase on Linux
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader).

    Train loader yields batched segments.
    Val/test loaders yield one full track at a time (songs have variable length).

    For faster loading, convert the dataset to WAV first (see convert_to_wav.py)
    and pass is_wav=True. This is strongly recommended for real training runs.

    Warning: num_workers > 0 is unsafe with the current chunk_start/chunk_duration
    approach since multiple workers share the same track objects. On Linux, keep
    num_workers=0 or refactor to load audio without mutating the track.
    """
    train_tracks, val_tracks, test_tracks = load_musdb(root, is_wav)

    train_loader = DataLoader(
        SegmentDataset(train_tracks, segment_duration, sample_rate, sources, samples_per_track, normalize),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        FullTrackDataset(val_tracks, sources, normalize),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        FullTrackDataset(test_tracks, sources, normalize),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader
