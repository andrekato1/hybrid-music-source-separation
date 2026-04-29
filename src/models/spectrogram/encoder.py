import math

import torch
import torch.nn as nn
from torch import Tensor


class EncoderBlock(nn.Module):
    """One spectrogram encoder level.

    Strides along the frequency axis only (time axis is always 1-stride).
    Pointwise Conv2d + GLU gate follows the strided conv for channel gating.

    With freq_stride=2, freq_kernel=4, padding=1:
        output_freq = floor(input_freq / 2)
    """

    def __init__(self, in_channels: int, out_channels: int, freq_stride: int = 2) -> None:
        super().__init__()
        freq_kernel = freq_stride * 2                   # 4 for stride=2
        padding = freq_kernel // 2 - 1                  # 1 for kernel=4

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(freq_kernel, 1),
            stride=(freq_stride, 1),
            padding=(padding, 0),
        )
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU()
        # Pointwise gate: doubles channels then GLU halves them back
        self.gate = nn.Conv2d(out_channels, out_channels * 2, 1)
        self.glu = nn.GLU(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.gate(x)
        return self.glu(x)


class SpectrogramEncoder(nn.Module):
    """Five-level spectrogram encoder.

    Downsamples the frequency axis by freq_stride at each level while
    leaving the time axis unchanged.  A learned frequency embedding is added
    after the first block to give the network explicit bin-position awareness.
    A two-layer BiLSTM processes the time axis at the bottleneck.

    Channel schedule (considering defaults below): 4 > 48 > 96 > 192 > 384 > 768

    The four intermediate outputs (all except the bottleneck) are returned as
    skip connections ordered shallowest-first so the decoder can reverse them.

    Args:
        in_channels:  input feature-map channels (audio_channels * 2 for re+im)
        channels:     base channel count for level 0 (default 48)
        depth:        number of encoder levels (default 5)
        freq_stride:  frequency stride per level (default 2)
        n_fft:        FFT size; used to compute the padded frequency dimension
        lstm_layers:  BiLSTM layers at the bottleneck (default 2)
    """

    def __init__(
        self,
        in_channels: int = 4,
        channels: int = 48,
        depth: int = 5,
        freq_stride: int = 2,
        n_fft: int = 4096,
        lstm_layers: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.freq_stride = freq_stride
        self.n_fft = n_fft

        # Encoder blocks: channels double every level
        self.blocks = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            out_ch = channels * (2 ** i)
            self.blocks.append(EncoderBlock(in_ch, out_ch, freq_stride))
            in_ch = out_ch

        bottleneck_ch = channels * (2 ** (depth - 1))   # 768 for channels=48, depth=5

        # Frequency embedding: added to the first-level output so the model
        # knows which frequency bin it is looking at after the first downsample.
        n_freq_l0 = self._padded_freq() // freq_stride
        self.freq_emb = nn.Embedding(n_freq_l0, channels)
        self.freq_emb_scale = nn.Parameter(torch.full((1,), 0.2))

        # BiLSTM: runs along the time axis at the bottleneck.
        # hidden_size = bottleneck_ch // 2 so that bidirectional output = bottleneck_ch
        self.lstm = nn.LSTM(
            input_size=bottleneck_ch,
            hidden_size=bottleneck_ch // 2,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

    def _padded_freq(self) -> int:
        """Smallest multiple of freq_stride**depth that is >= n_fft//2+1."""
        n_freq = self.n_fft // 2 + 1
        stride_total = self.freq_stride ** self.depth
        return math.ceil(n_freq / stride_total) * stride_total

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """
        Args:
            x: [B, in_channels, F, T]
            F must equal _padded_freq() (caller is responsible for padding)

        Returns:
            bottleneck: [B, C_bottleneck, F_small, T]
            skips:      list of [B, C_i, F_i, T], shallowest first
                        (length = depth - 1; the bottleneck itself is not a skip)
        """
        skips: list[Tensor] = []

        for i, block in enumerate(self.blocks):
            x = block(x)

            if i == 0:
                # Frequency positional embedding: [F_cur, C] > [1, C, F_cur, 1]
                F_cur = x.shape[2]
                freq_idx = torch.arange(F_cur, device=x.device)
                emb = self.freq_emb(freq_idx)                       # [F_cur, C]
                emb = emb.T.unsqueeze(0).unsqueeze(-1)              # [1, C, F_cur, 1]
                x = x + self.freq_emb_scale * emb

            if i < self.depth - 1:
                skips.append(x)
            # The output of the last block is the bottleneck (not stored as skip)

        # BiLSTM along the time axis
        # x: [B, C, F_small, T]
        B, C, F_small, T = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B * F_small, T, C)  # [B*F_small, T, C]
        x_flat, _ = self.lstm(x_flat)                              # [B*F_small, T, C]
        x = x_flat.reshape(B, F_small, T, C).permute(0, 3, 1, 2)   # [B, C, F_small, T]

        return x, skips
