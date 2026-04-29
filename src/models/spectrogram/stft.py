import torch
import torch.nn as nn
from torch import Tensor


class STFT(nn.Module):
    """Wraps torch.stft / torch.istft as an nn.Module.

    Registering the Hann window as a buffer ensures it moves to the correct
    device with the model and is included in state_dict.
    """

    def __init__(self, n_fft: int = 4096, hop_length: int = 1024) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

    # ------------------------------------------------------------------
    # Public interface uses encode/decode to avoid shadowing nn.Module.forward
    # ------------------------------------------------------------------

    def encode(self, waveform: Tensor) -> Tensor:
        """Convert a waveform to a stacked real/imaginary spectrogram.

        Args:
            waveform: ``[B, C, T]``

        Returns:
            ``[B, C*2, F, T_frames]`` where the C real and C imaginary parts
            are interleaved as ``[re_0, im_0, re_1, im_1, ...]``.
        """
        B, C, T = waveform.shape
        x = waveform.reshape(B * C, T)
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            normalized=False,
            pad_mode="reflect",
        )  # [B*C, F, T_frames]
        F_bins, T_frames = spec.shape[-2], spec.shape[-1]
        spec = spec.reshape(B, C, F_bins, T_frames)         # [B, C, F, T_frames] complex
        spec = torch.view_as_real(spec)                     # [B, C, F, T_frames, 2]
        spec = spec.permute(0, 1, 4, 2, 3).contiguous()    # [B, C, 2, F, T_frames]
        return spec.reshape(B, C * 2, F_bins, T_frames)    # [B, C*2, F, T_frames]

    def decode(self, spec: Tensor, length: int) -> Tensor:
        """Convert a stacked real/imaginary spectrogram back to a waveform.

        Args:
            spec:   ``[B, C*2, F, T_frames]``
            length: original waveform length (passed to ``torch.istft``)

        Returns:
            ``[B, C, T]``
        """
        B, C2, F_bins, T_frames = spec.shape
        C = C2 // 2
        spec = spec.reshape(B, C, 2, F_bins, T_frames)
        spec = spec.permute(0, 1, 3, 4, 2).contiguous()    # [B, C, F, T_frames, 2]
        spec = torch.view_as_complex(spec)                  # [B, C, F, T_frames] complex
        spec_flat = spec.reshape(B * C, F_bins, T_frames)
        wav = torch.istft(
            spec_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            length=length,
            normalized=False,
        )  # [B*C, T]
        return wav.reshape(B, C, length)
