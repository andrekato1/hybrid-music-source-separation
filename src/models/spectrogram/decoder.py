from typing import Sequence

import torch.nn as nn
from torch import Tensor


class DecoderBlock(nn.Module):
    """One spectrogram decoder level.

    A pointwise Conv2d + GLU gate precedes the transposed conv so the network
    can control information flow before upsampling.  Norm + activation follow
    the transposed conv at every level except the last output layer.

    Kernel/padding match the encoder so that the transposed conv exactly
    inverts a strided conv with the same stride:

        freq_kernel = 2 * freq_stride
        padding     = freq_stride // 2

    With these values, ``output_freq = input_freq * freq_stride`` exactly.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        freq_stride: int = 2,
        last: bool = False,
    ) -> None:
        super().__init__()
        freq_kernel = freq_stride * 2
        padding = freq_stride // 2

        # Pointwise gate before upsampling
        self.gate = nn.Conv2d(in_channels, in_channels * 2, 1)
        self.glu = nn.GLU(dim=1)

        self.conv_t = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=(freq_kernel, 1),
            stride=(freq_stride, 1),
            padding=(padding, 0),
        )
        self.last = last
        if not last:
            self.norm = nn.GroupNorm(1, out_channels)
            self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.gate(x)
        x = self.glu(x)
        x = self.conv_t(x)
        if not self.last:
            x = self.norm(x)
            x = self.act(x)
        return x


class SpectrogramDecoder(nn.Module):
    """Five-level spectrogram decoder.

    Mirrors the encoder: upsamples the frequency axis by the per-level stride
    at each level (in reverse order) and adds the corresponding encoder skip
    connection after each upsample step (additive).

    The final block outputs out_channels feature maps at the original
    (padded) frequency resolution; the caller crops and applies iSTFT.

    Args:
        channels:     base channel count matching the encoder
        out_channels: channels of the final output layer
                      = n_sources * audio_channels * 2
        depth:        number of decoder levels (must match encoder, default 5)
        freq_strides: encoder-side frequency strides (per level). The decoder
                      applies them in reverse order so the deepest decoder
                      block uses the deepest encoder stride. Either a single
                      int or a sequence of length depth.
    """

    def __init__(
        self,
        channels: int = 48,
        out_channels: int = 16,
        depth: int = 5,
        freq_strides: Sequence[int] | int = (4, 4, 4, 8, 8),
    ) -> None:
        super().__init__()
        self.depth = depth
        if isinstance(freq_strides, int):
            freq_strides_list = [freq_strides] * depth
        else:
            freq_strides_list = list(freq_strides)
            
        # Decoder applies encoder strides in reverse so the deepest decoder
        # block (which sees the bottleneck first) uses the last encoder stride.
        self.freq_strides = list(reversed(freq_strides_list))

        # Channel schedule mirrors the encoder in reverse:
        #   encoder: [48, 96, 192, 384, 768]
        #   decoder: 768 > 384 > 192 > 96 > 48 > out_channels
        ch_list = [channels * (2 ** i) for i in range(depth)]      # [48, 96, 192, 384, 768]
        ch_reversed = list(reversed(ch_list))                      # [768, 384, 192, 96, 48]

        self.blocks = nn.ModuleList()
        for i, in_ch in enumerate(ch_reversed):
            is_last = (i == depth - 1)
            next_ch = ch_reversed[i + 1] if not is_last else out_channels
            self.blocks.append(
                DecoderBlock(in_ch, next_ch, self.freq_strides[i], last=is_last)
            )

    def forward(self, bottleneck: Tensor, skips: list[Tensor]) -> Tensor:
        """
        Args:
            bottleneck: [B, C_bottleneck, F_small, T]
                        F_small is 1 when the encoder fully crushes frequency.
            skips:      encoder skip connections, shallowest first
                        (length = depth - 1)

        Returns:
            [B, out_channels, F_padded, T]
        """
        # Reverse so the deepest skip aligns with the first decoder block
        skips_rev = list(reversed(skips))
        x = bottleneck
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(skips_rev):
                x = x + skips_rev[i]
        return x
