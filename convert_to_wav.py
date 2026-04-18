"""
convert_to_wav.py
-----------------
One-time conversion of MUSDB18 .stem.mp4 files to WAV stems.

Run once before training, then set is_wav=True in dataset.py for faster loading.

Usage:
    uv run python convert_to_wav.py
"""

from pathlib import Path
import numpy as np
import soundfile as sf
import musdb

ROOT = Path("data/")
SAMPLE_RATE = 44100
SOURCES = ["drums", "bass", "other", "vocals"]


def convert_subset(subset: str) -> None:
    mus = musdb.DB(root=str(ROOT), is_wav=False)
    tracks = mus.load_mus_tracks(subsets=subset)
    print(f"\nConverting {len(tracks)} {subset} tracks...")

    for i, track in enumerate(tracks):
        out_dir = ROOT / subset / track.name
        out_dir.mkdir(exist_ok=True)

        # Load full track audio (no chunk — we want everything)
        track.chunk_start = 0
        track.chunk_duration = track.duration

        # Write mixture
        sf.write(out_dir / "mixture.wav", track.audio, SAMPLE_RATE)

        # Write each stem
        for source in SOURCES:
            sf.write(out_dir / f"{source}.wav", track.targets[source].audio, SAMPLE_RATE)

        print(f"  [{i+1}/{len(tracks)}] {track.name}")


if __name__ == "__main__":
    print("Converting MUSDB18 stems to WAV format (one-time setup)...")
    convert_subset("train")
    convert_subset("test")
    print("\nDone. Set is_wav=True in train.py to use the converted files.")
