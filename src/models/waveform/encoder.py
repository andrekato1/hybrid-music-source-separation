"""
Waveform encoder for music source separation.
Demucs has encoder and decoder depth = 6, reducing to 4.

Model inspired by
@article{defossez2019music,
  title={Music Source Separation in the Waveform Domain},
  author={D{\'e}fossez, Alexandre and Usunier, Nicolas and Bottou, L{\'e}on and Bach, Francis},
  journal={arXiv preprint arXiv:1911.13254},
  year={2019}
}
"""
from torch import nn


class DemucsEncoder(nn.Module):
    def __init__(self, in_channels=2, base_channels=64):
        super().__init__()
        C = base_channels

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, C, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(C, 2*C, kernel_size=1, stride=1),
            nn.GLU(dim=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(C, 2*C, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(2*C, 4*C, kernel_size=1, stride=1),
            nn.GLU(dim=1),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(2*C, 4*C, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(4*C, 8*C, kernel_size=1, stride=1),
            nn.GLU(dim=1),
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(4*C, 8*C, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(8*C, 16*C, kernel_size=1, stride=1),
            nn.GLU(dim=1),
        )

        # 5th level so the bottleneck reaches T/1024 (matches the spectrogram branch
        # for hybrid fusion). Output channels = 16*C.
        self.layer5 = nn.Sequential(
            nn.Conv1d(8*C, 16*C, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(16*C, 32*C, kernel_size=1, stride=1),
            nn.GLU(dim=1),
        )

    def forward(self, x):
        skip1 = self.layer1(x)
        skip2 = self.layer2(skip1)
        skip3 = self.layer3(skip2)
        skip4 = self.layer4(skip3)
        skip5 = self.layer5(skip4)

        return skip5, [skip1, skip2, skip3, skip4, skip5]

