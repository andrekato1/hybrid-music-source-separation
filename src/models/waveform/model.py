"""
Waveform-domain source separation model.

Model inspired by
@article{defossez2019music,
  title={Music Source Separation in the Waveform Domain},
  author={D{\'e}fossez, Alexandre and Usunier, Nicolas and Bottou, L{\'e}on and Bach, Francis},
  journal={arXiv preprint arXiv:1911.13254},
  year={2019}
}
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from .encoder import DemucsEncoder
from .decoder import DemucsDecoder

class BLSTM(nn.Module): ## Like Demuc
    def __init__(self, dim, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=dim,
                            hidden_size=dim,
                            num_layers = layers,
                            bidirectional=True
                            )
        self.linear = nn.Linear(2 * dim, dim) ## defaults following Demucs logic

    def forward(self, x):
        x = x.permute(2,0,1)
        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1,2,0)
        return x

class WaveformModel(nn.Module):
    def __init__(self, audio_channels=2, lstm_layers=2):
        """
        :param audio_channels: 1 for mono, 2 for stereo
        :param lstm_layers: number of bidirectiional LSTM layers in bottleneck
        """

        super().__init__()

        self.audio_channels = audio_channels
        self.encoder = DemucsEncoder(in_channels = audio_channels)
        self.lstm = BLSTM(dim=512, layers=lstm_layers)
        self.decoder = DemucsDecoder(in_channels=audio_channels)

    # Leveraged from Demucs: https://github.com/facebookresearch/demucs/blob/v2/demucs/model.py
    def valid_length(self, length):
        """Return the nearest valid length >= `length` so the decoder output matches exactly."""
        for _ in range(4):
            length = math.ceil((length - 8) / 4) + 1
            length += 2  # context - 1
        for _ in range(4):
            length = (length - 1) * 4 + 8
        return int(length)

    def forward(self, x):
        """

        Input: Mixed audio of shape (batch, audio_channels, time)
        Returns: Tensor separated sources of shape (batch, 1, audio_channels, time)
        """
        length = x.shape[-1]
        x = F.pad(x, (0, self.valid_length(length) - length))

        # encode
        x, skips = self.encoder(x)

        # bottleneck
        x = self.lstm(x)

        # decode
        x = self.decoder(x, skips)

        return {"vocals": x[..., :length]}

