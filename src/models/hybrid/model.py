from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.base import BaseEncDecSeparator, Latent
from src.models.waveform.model import WaveformModel, WaveformLatent
from src.models.spectrogram.model import SpectrogramBranch, SpectrogramLatent


@dataclass
class HybridLatent(Latent):
    """Carries both branches' latents through to decode time."""
    wav_latent: WaveformLatent | None = None
    spec_latent: SpectrogramLatent | None = None


class HybridConcatModel(BaseEncDecSeparator):
    """Hybrid waveform + spectrogram model with concatenation fusion.

    Both branches encode the mixture independently. At the bottleneck their
    feature maps are concatenated along the channel axis, then projected back
    to each branch's expected channel count via a 1x1 conv. Each decoder runs
    on its fused bottleneck and its own skip connections, producing a waveform.
    The two waveforms are averaged to give the final estimate.

    Args:
        target_source: which stem to extract ("vocals", "drums", "bass", "other")
        base_channels: waveform branch base channel width (bottleneck = 16 * base_channels)
        lstm_dim:      waveform branch BiLSTM hidden size
        spec_channels: spectrogram branch base channel count
        spec_depth:    spectrogram branch encoder/decoder depth
        n_sources:     number of sources the spectrogram branch is built for
    """

    def __init__(
        self,
        target_source: str = "drums",
        base_channels: int = 32,
        lstm_dim: int = 320,
        spec_channels: int = 48,
        spec_depth: int = 5,
        n_sources: int = 4,
    ) -> None:
        super().__init__()
        self.target_source = target_source

        self.wav = WaveformModel(
            base_channels=base_channels,
            lstm_dim=lstm_dim,
            target_source=target_source,
        )
        self.spec = SpectrogramBranch(
            n_sources=n_sources,
            channels=spec_channels,
            depth=spec_depth,
        )

        # Bottleneck channel counts at each branch's [B, C, T] bottleneck (after
        # squeezing the spectrogram's singleton frequency axis).
        wav_bn_ch = 16 * base_channels
        spec_bn_ch = spec_channels * (2 ** (spec_depth - 1))
        fused_ch = wav_bn_ch + spec_bn_ch

        # Project the concatenated representation back to each branch's
        # expected bottleneck channel count.
        self.fuse_to_wav = nn.Conv1d(fused_ch, wav_bn_ch, kernel_size=1)
        self.fuse_to_spec = nn.Conv1d(fused_ch, spec_bn_ch, kernel_size=1)

    def encode(self, mixture: Tensor) -> HybridLatent:
        """Encode both branches and fuse their bottlenecks.

        Args:
            mixture: [B, C, T] raw waveform

        Returns:
            HybridLatent carrying both branches' latents, with their
            bottlenecks already replaced by the fused projections.
        """
        wav_latent = self.wav.encode(mixture)
        spec_latent = self.spec.encode(mixture)

        wav_bn = wav_latent.bottleneck                 # [B, C_w, T_w]
        spec_bn = spec_latent.bottleneck.squeeze(2)    # [B, C_s, T_s]

        # Bottleneck time axes can differ by 1-2 frames between branches.
        # Each branch's decoder is built around its own native time length, so
        # we keep the natives and use linear interpolation only to align inputs
        # to the concatenation step, then interpolate back to each branch's
        # native length before handing to its decoder.
        T_w, T_s = wav_bn.shape[-1], spec_bn.shape[-1]
        spec_bn_aligned = F.interpolate(spec_bn, size=T_w, mode="linear", align_corners=False)
        fused = torch.cat([wav_bn, spec_bn_aligned], dim=1)    # [B, C_w + C_s, T_w]

        wav_latent.bottleneck = self.fuse_to_wav(fused)        # [B, C_w, T_w]
        fused_to_spec = self.fuse_to_spec(fused)               # [B, C_s, T_w]
        if T_s != T_w:
            fused_to_spec = F.interpolate(fused_to_spec, size=T_s, mode="linear", align_corners=False)
        spec_latent.bottleneck = fused_to_spec.unsqueeze(2)    # [B, C_s, 1, T_s]

        return HybridLatent(
            bottleneck=fused,
            skips=[],
            wav_latent=wav_latent,
            spec_latent=spec_latent,
        )

    def decode(self, latent: Latent) -> dict[str, Tensor]:
        """Run both decoders on the fused latents and average their outputs."""
        assert isinstance(latent, HybridLatent)

        wav_out = self.wav.decode(latent.wav_latent)[self.target_source]
        spec_out = self.spec.decode(latent.spec_latent)[self.target_source]

        # Output lengths can differ by a sample due to STFT vs valid_length
        # rounding; align before averaging.
        T = min(wav_out.shape[-1], spec_out.shape[-1])
        out = (wav_out[..., :T] + spec_out[..., :T]) / 2

        return {self.target_source: out}

    def forward(self, mixture: Tensor) -> dict[str, Tensor]:
        return self.decode(self.encode(mixture))


class HybridGatingModel(BaseEncDecSeparator):
    """Hybrid waveform + spectrogram model with sigmoid-gated fusion.

    Both branches encode the mixture independently. At the bottleneck both
    feature maps are projected to a common channel dimension. A sigmoid gate
    is then learned per feature from the concatenated projections; this gate
    weights the contributions of each branch element-wise:

        fused = gate * wav_proj + (1 - gate) * spec_proj

    The fused representation is then projected back to each branch's expected
    channel count via a 1x1 conv. Each decoder runs on its gated bottleneck
    and its own skip connections, producing a waveform. The two waveforms are
    averaged to give the final estimate.

    Args:
        target_source: which stem to extract ("vocals", "drums", "bass", "other")
        base_channels: waveform branch base channel width (bottleneck = 16 * base_channels)
        lstm_dim:      waveform branch BiLSTM hidden size
        spec_channels: spectrogram branch base channel count
        spec_depth:    spectrogram branch encoder/decoder depth
        n_sources:     number of sources the spectrogram branch is built for
        fused_ch:      shared fused-space channel dim. Defaults to max of the
                       two branches' bottleneck channel counts.
    """

    def __init__(
        self,
        target_source: str = "drums",
        base_channels: int = 32,
        lstm_dim: int = 320,
        spec_channels: int = 48,
        spec_depth: int = 5,
        n_sources: int = 4,
        fused_ch: int | None = None,
    ) -> None:
        super().__init__()
        self.target_source = target_source

        self.wav = WaveformModel(
            base_channels=base_channels,
            lstm_dim=lstm_dim,
            target_source=target_source,
        )
        self.spec = SpectrogramBranch(
            n_sources=n_sources,
            channels=spec_channels,
            depth=spec_depth,
        )

        wav_bn_ch = 16 * base_channels
        spec_bn_ch = spec_channels * (2 ** (spec_depth - 1))
        if fused_ch is None:
            fused_ch = max(wav_bn_ch, spec_bn_ch)

        self.proj_wav = nn.Conv1d(wav_bn_ch, fused_ch, kernel_size=1)
        self.proj_spec = nn.Conv1d(spec_bn_ch, fused_ch, kernel_size=1)
        self.gate = nn.Conv1d(2 * fused_ch, fused_ch, kernel_size=1)

        self.fuse_to_wav = nn.Conv1d(fused_ch, wav_bn_ch, kernel_size=1)
        self.fuse_to_spec = nn.Conv1d(fused_ch, spec_bn_ch, kernel_size=1)

    def encode(self, mixture: Tensor) -> HybridLatent:
        """Encode both branches and fuse their bottlenecks via a sigmoid gate.

        Args:
            mixture: [B, C, T] raw waveform

        Returns:
            HybridLatent carrying both branches' latents, with their
            bottlenecks replaced by the gated fusion.
        """
        wav_latent = self.wav.encode(mixture)
        spec_latent = self.spec.encode(mixture)

        wav_bn = wav_latent.bottleneck                 # [B, C_w, T_w]
        spec_bn = spec_latent.bottleneck.squeeze(2)    # [B, C_s, T_s]

        T_w, T_s = wav_bn.shape[-1], spec_bn.shape[-1]
        spec_bn_aligned = F.interpolate(spec_bn, size=T_w, mode="linear", align_corners=False)

        wav_proj = self.proj_wav(wav_bn)                       # [B, fused_ch, T_w]
        spec_proj = self.proj_spec(spec_bn_aligned)            # [B, fused_ch, T_w]
        gate = torch.sigmoid(self.gate(torch.cat([wav_proj, spec_proj], dim=1)))
        fused = gate * wav_proj + (1 - gate) * spec_proj       # [B, fused_ch, T_w]

        wav_latent.bottleneck = self.fuse_to_wav(fused)        # [B, C_w, T_w]
        fused_to_spec = self.fuse_to_spec(fused)               # [B, C_s, T_w]
        if T_s != T_w:
            fused_to_spec = F.interpolate(fused_to_spec, size=T_s, mode="linear", align_corners=False)
        spec_latent.bottleneck = fused_to_spec.unsqueeze(2)    # [B, C_s, 1, T_s]

        return HybridLatent(
            bottleneck=fused,
            skips=[],
            wav_latent=wav_latent,
            spec_latent=spec_latent,
        )

    def decode(self, latent: Latent) -> dict[str, Tensor]:
        """Run both decoders on the gated latents and average their outputs."""
        assert isinstance(latent, HybridLatent)

        wav_out = self.wav.decode(latent.wav_latent)[self.target_source]
        spec_out = self.spec.decode(latent.spec_latent)[self.target_source]

        T = min(wav_out.shape[-1], spec_out.shape[-1])
        out = (wav_out[..., :T] + spec_out[..., :T]) / 2

        return {self.target_source: out}

    def forward(self, mixture: Tensor) -> dict[str, Tensor]:
        return self.decode(self.encode(mixture))
