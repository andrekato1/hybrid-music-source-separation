import math
from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor

from src.data.dataset import SOURCES
from src.models.base import BaseEncDecSeparator, Latent
from src.models.spectrogram.decoder import SpectrogramDecoder
from src.models.spectrogram.encoder import SpectrogramEncoder
from src.models.spectrogram.stft import STFT


@dataclass
class SpectrogramLatent(Latent):
    """Extends Latent with the metadata needed to invert the STFT."""
    n_freq_orig: int = 0    # F before frequency padding  (= n_fft // 2 + 1)
    wav_length: int = 0     # original waveform length T for torch.istft


class SpectrogramBranch(BaseEncDecSeparator):
    """Spectrogram branch of Hybrid Demucs.

    Waveform in > STFT > 2D U-Net (freq-only strides) + BiLSTM > iSTFT >
    four separated stereo waveforms out.

    Args:
        audio_channels: stereo=2
        n_sources:      number of sources to separate
        n_fft:          STFT FFT size
        hop_length:     STFT hop length
        channels:       base channel count for level 0
        depth:          number of encoder/decoder levels
        freq_stride:    frequency stride per level
        lstm_layers:    BiLSTM layers at the bottleneck
    """

    def __init__(
        self,
        audio_channels: int = 2,
        n_sources: int = 4,
        n_fft: int = 4096,
        hop_length: int = 1024,
        channels: int = 48,
        depth: int = 5,
        freq_stride: int = 2,
        lstm_layers: int = 2,
    ) -> None:
        super().__init__()
        self.audio_channels = audio_channels
        self.n_sources = n_sources
        self.n_fft = n_fft
        self.depth = depth
        self.freq_stride = freq_stride

        self.stft = STFT(n_fft=n_fft, hop_length=hop_length)
        self.encoder = SpectrogramEncoder(
            in_channels=audio_channels * 2,     # real + imag per audio channel
            channels=channels,
            depth=depth,
            freq_stride=freq_stride,
            n_fft=n_fft,
            lstm_layers=lstm_layers,
        )
        self.decoder = SpectrogramDecoder(
            channels=channels,
            out_channels=n_sources * audio_channels * 2,    # 16 for defaults
            depth=depth,
            freq_stride=freq_stride,
        )

    def _padded_freq(self) -> int:
        n_freq = self.n_fft // 2 + 1
        stride_total = self.freq_stride ** self.depth
        return math.ceil(n_freq / stride_total) * stride_total

    def encode(self, mixture: Tensor) -> SpectrogramLatent:
        """STFT > frequency padding > encoder

        Args:
            mixture: [B, C, T] raw waveform

        Returns:
            SpectrogramLatent carrying the bottleneck, skip connections,
            and the metadata required to invert the transform in decode.
        """
        wav_length = mixture.shape[-1]

        spec = self.stft.encode(mixture)            # [B, C*2, F, T_frames]
        n_freq_orig = spec.shape[2]

        # Pad the frequency axis so every level divides evenly
        pad_amt = self._padded_freq() - n_freq_orig
        if pad_amt > 0:
            # F.pad tuple is applied last-dim first: (T_left, T_right, F_left, F_right)
            spec = F.pad(spec, (0, 0, 0, pad_amt))

        bottleneck, skips = self.encoder(spec)      # [B, C_bn, F_small, T_frames]

        return SpectrogramLatent(
            bottleneck=bottleneck,
            skips=skips,
            n_freq_orig=n_freq_orig,
            wav_length=wav_length,
        )

    def decode(self, latent: Latent) -> dict[str, Tensor]:
        """Decoder > frequency crop > iSTFT > source waveforms

        Args:
            latent: SpectrogramLatent produced by encode

        Returns:
            dict mapping source name to [B, C, T] waveform
        """
        assert isinstance(latent, SpectrogramLatent)

        x = self.decoder(latent.bottleneck, latent.skips)
        # x: [B, n_sources * C*2, F_padded, T_frames]

        # Crop back to the original (unpadded) frequency bins
        x = x[:, :, :latent.n_freq_orig, :]
        # x: [B, n_sources * C*2, F_orig, T_frames]

        # Reshape so each source is a separate "batch item" for iSTFT
        B, _, F_orig, T_frames = x.shape
        x = x.reshape(B * self.n_sources, self.audio_channels * 2, F_orig, T_frames)
        waveforms = self.stft.decode(x, latent.wav_length)          # [B*n_sources, C, T]
        waveforms = waveforms.reshape(B, self.n_sources, self.audio_channels, latent.wav_length)

        return {
            source: waveforms[:, i]     # [B, C, T]
            for i, source in enumerate(SOURCES[: self.n_sources])
        }

    def forward(self, mixture: Tensor) -> dict[str, Tensor]:
        """End-to-end separation

        Args:
            mixture: [B, C, T]

        Returns:
            dict mapping source name to [B, C, T]
        """
        return self.decode(self.encode(mixture))
