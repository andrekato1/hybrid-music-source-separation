"""
convert_to_wav.py
-----------------
One-time conversion of MUSDB18 .stem.mp4 files to WAV stems.

Run this once before training to get much faster data loading.
WAV files are written alongside the original .stem.mp4 files in data/train/ and data/test/.

Usage:
    uv run python convert_to_wav.py

After conversion, pass is_wav=True to get_dataloaders() or set IS_WAV=True in train.py.
"""

import musdb

print("Converting MUSDB18 stems to WAV format (one-time setup)...")
print("This may take several minutes.\n")

mus = musdb.DB(root="data/", is_wav=False)

for subset in ("train", "test"):
    tracks = mus.load_mus_tracks(subsets=subset)
    print(f"Converting {len(tracks)} {subset} tracks...")
    for i, track in enumerate(tracks):
        track.stems  # triggers WAV extraction via stempeg
        print(f"  [{i+1}/{len(tracks)}] {track.name}")

print("\nDone. You can now use is_wav=True in get_dataloaders().")
