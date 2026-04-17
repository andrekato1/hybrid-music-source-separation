# Hybrid Music Source Separation

Implementation inspired on [Hybrid Demucs](https://arxiv.org/abs/2111.03600) paper for music source separation using the MUSDB18 dataset.

## Setup

### 1. Install uv

https://docs.astral.sh/uv/guides/install-python/

### 2. Install dependencies

```bash
uv sync
```

### 3. Install PyTorch manually

https://pytorch.org/get-started/locally/

### 4. Download the dataset

```bash
uv run python download_data.py
```

This downloads MUSDB18 into `data/` using the predefined 86/14 train/test split so the results are comparable to the literature.

## Running scripts

Either prefix commands with `uv run`:
```bash
uv run python download_data.py
```

Or activate the virtual environment once and use `python` directly:
```bash
# Windows
source .venv/Scripts/activate

# Linux
source .venv/bin/activate

python download_data.py
```
## Dataset

Download from https://zenodo.org/records/1117372, extract into the data/ folder and convert the files to wav format using:
```bash
uv run python convert_to_wav.py
```