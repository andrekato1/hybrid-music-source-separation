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
    def __init__(self, in_channels=2): # 2 channels for stereo
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1, stride=1),
            nn.GLU(dim=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1, stride=1),
            nn.GLU(dim=1),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=1, stride=1),
            nn.GLU(dim=1),
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(512, 1024, kernel_size=1, stride=1),
            nn.GLU(dim=1),
        )

    def forward(self, x):
        skip1 = self.layer1(x)
        skip2 = self.layer2(skip1)
        skip3 = self.layer3(skip2)
        skip4 = self.layer4(skip3)

        return skip4, [skip1, skip2, skip3, skip4]

